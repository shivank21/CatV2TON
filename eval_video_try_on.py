import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from tqdm import tqdm

from modules.cloth_masker import AutoMasker
from modules.pipeline import V2TONPipeline


def read_video_frames(video_path, start_frame, end_frame, normalize=True):
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    video = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
    if end_frame > video.size(0):  # end_frame is exclusive
        end_frame = video.size(0)
    video = video[start_frame:end_frame].permute(1, 0, 2, 3).unsqueeze(0)
    video = video.float() / 255.0
    if normalize:
        video = video * 2 - 1
    return video

class ViViDTestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.data = self.load_data()
        self.process_mask()
        
    def load_data(self):
        jsonl_path = os.path.join(self.args.data_root_path, 'test_data_180.jsonl' if self.args.eval_pair else 'test_data_180_unpaired.jsonl')
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = json.loads(line)
            result_path = os.path.join(self.args.output_dir, f'{self.args.dataset}-{self.args.height}', 'paired' if self.args.eval_pair else 'unpaired', line['video'])
            if os.path.exists(result_path): 
                continue
            video_path = os.path.join(self.args.data_root_path, line['subdir'], 'videos', line['video'])
            image_path = os.path.join(self.args.data_root_path, line['subdir'], 'images', line['image'])
            mask_path = os.path.join(self.args.data_root_path, line['subdir'], 'agnostic_mask_smoothed', line['video'])
            densepose_path = os.path.join(self.args.data_root_path, line['subdir'], 'densepose', line['video'])

            data.append({
                'video': video_path,
                'image': image_path,
                'mask': mask_path,
                'densepose': densepose_path,
                'front_seqs': line['front_seqs'],
                'subdir': line['subdir'],
                'category': {'upper_body': 'upper', 'lower_body': 'lower', 'dresses': 'overall'}[line['subdir']]
            })
        return data

    def process_mask(self):
        """
        Use AutoMasker to process videos without masks.
        Return the processed mask video.
        """
        # Initialize AutoMasker
        catvton_ckpt_path = snapshot_download(self.args.catvton_ckpt_path) if not os.path.exists(self.args.catvton_ckpt_path) else self.args.catvton_ckpt_path
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(catvton_ckpt_path, "DensePose"),
            schp_ckpt=os.path.join(catvton_ckpt_path, "SCHP"),
            device="cuda"
        )
        # Check if mask_path exists
        for idx in range(len(self.data)):
            data = self.data[idx]
            video_path = data['video']
            mask_path = data['mask']
            if not os.path.exists(mask_path):
                # Process video to generate mask
                preprocess_result = automasker.process_video(
                    mask_type=data['category'],  # 'upper', 'lower' or 'overall'
                    video_path=video_path,
                    densepose_colormap=cv2.COLORMAP_VIRIDIS,
                )
                # Get the generated mask
                mask = preprocess_result['mask']
                # If it's a video file
                if video_path.endswith('.mp4'):
                    mask = mask.permute(1, 2, 3, 0).cpu()
                    mask_dir = os.path.dirname(mask_path)
                    os.makedirs(mask_dir, exist_ok=True)
                    write_video(mask_path, mask, fps=25)
                else:
                    # If it's an image file
                    mask_dir = os.path.dirname(mask_path)
                    os.makedirs(mask_dir, exist_ok=True)
                    mask.save(mask_path.replace('.jpg', '.png'))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        video_path = data['video']
        image_path = data['image']
        mask_path = data['mask']
        densepose_path = data['densepose']
        front_seqs = data['front_seqs']
        subdir = data['subdir']
        
        # Check all files exist
        exist_status = [os.path.exists(video_path), os.path.exists(image_path), os.path.exists(mask_path), os.path.exists(densepose_path)]
        if not all(exist_status):
            print(f"{self.missing_files} {video_path} is missing some files: {exist_status}")
        
        video = read_video_frames(video_path, front_seqs[0], front_seqs[1]).squeeze(0)
        mask = read_video_frames(mask_path, front_seqs[0], front_seqs[1], normalize=False).squeeze(0)
        densepose = read_video_frames(densepose_path, front_seqs[0], front_seqs[1]).squeeze(0)
        image = Image.open(image_path)
        image = self.image_transforms(image).unsqueeze(1)
        
        return {
            'video': video,
            'image': image,
            'mask': mask,
            'densepose': densepose,
            'subdir': subdir,
            'name': os.path.basename(video_path)
        }

class VVTTestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.data = self.load_data()
        self.process_mask()
        
    def process_mask(self):
        """
        Use AutoMasker to process videos without masks.
        Since VVT dataset stores videos as frame images, the processing results also need to be saved as frame images.
        """
        # Initialize AutoMasker
        automasker = AutoMasker(
            densepose_ckpt=os.path.join("Models/CatVTON", "DensePose"),
            schp_ckpt=os.path.join("Models/CatVTON", "SCHP"),
            device="cuda"
        )
        
        # Check if mask_path exists for each sample
        for idx in tqdm(range(len(self.data)), desc="Processing VVT dataset masks"):
            data = self.data[idx]
            video_path = data['video']  # This is a directory path
            mask_path = data['mask']    # This is also a directory path
            
            if os.path.exists(video_path):
                # Create save directory
                os.makedirs(mask_path, exist_ok=True)
                for image_path in os.listdir(video_path):
                    if os.path.exists(os.path.join(mask_path, image_path)):
                        continue
                    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
                        continue
                    # Process video to generate mask
                    preprocess_result = automasker.process_video(
                        mask_type='upper',  # VVT dataset is all upper body clothing
                        video_path=os.path.join(video_path, image_path),
                        densepose_colormap=cv2.COLORMAP_VIRIDIS,
                    )
                    # Get the generated mask
                    mask = preprocess_result['mask']  # This is a tensor
                    
                    # Save mask as frame images
                    filename = os.path.basename(image_path)
                    save_path = os.path.join(mask_path, filename)
                    mask.save(save_path)

    def load_data(self):
        txt_path = os.path.join(self.args.data_root_path, 'test_person_clothes_pose_tuple.txt')   
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            person_id, clothes_id, _ = line.strip().split(' ')
            image_path = os.path.join(self.args.data_root_path, 'lip_clothes_person', person_id if self.args.eval_pair else clothes_id)
            image_path = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')][0] # Files ending with cloth_front.jpg under image_path
            
            densepose_path = os.path.join(self.args.data_root_path, 'test_densepose-magcianimate', person_id)
            mask_path = os.path.join(self.args.data_root_path, 'test_agn_mask_new', person_id)
            video_path = os.path.join(self.args.data_root_path, 'test_frames', person_id)
            
            all_exist = [os.path.exists(video_path), os.path.exists(image_path), os.path.exists(mask_path), os.path.exists(densepose_path)]
            if not os.path.exists(video_path):
                print(f"{person_id} is missing some files: {all_exist}")
                continue    
        
            exist_status = {
                "video": os.path.exists(video_path), 
                "image": os.path.exists(image_path), 
                "mask": os.path.exists(mask_path), 
                "densepose": os.path.exists(densepose_path)
            }
            if not all(exist_status.values()):
                print(f"{video_path} is missing some files: {exist_status}")

            data.append({
                'video': video_path,
                'image': image_path,
                'mask': mask_path,
                'densepose': densepose_path,
            })
        return data

    def __len__(self):
        return len(self.data)
    
    def read_image_folder_as_video(self, image_folder_path):
        images = sorted([os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))])
        height, width = 256, 192
        video = np.empty((len(images), height, width, 3), dtype=np.uint8)

        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (width, height))
            video[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video = torch.from_numpy(video).float().permute(3, 0, 1, 2) / 255.0
        video = video * 2 - 1  
        
        return video
    
    def __getitem__(self, idx):
        data = self.data[idx]
        video_path = data['video']
        image_path = data['image']
        mask_path = data['mask']
        densepose_path = data['densepose']
        
        video = self.read_image_folder_as_video(video_path) 
        mask = self.read_image_folder_as_video(mask_path) * 0.5 + 0.5
        densepose = self.read_image_folder_as_video(densepose_path)
        image = Image.open(image_path).resize((192, 256))
        image = self.image_transforms(image).unsqueeze(1)
        
        return {
            'video': video,
            'image': image,
            'mask': mask,
            'densepose': densepose,
            'name': os.path.basename(video_path) + '.mp4'
        }
 
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="alibaba-pai/EasyAnimateV4-XL-2-InP",
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="zhengchong/CatV2TON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--catvton_ckpt_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of CatVTON."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vivid",
        choices=["vivid", "vvt"],
        help="The dataset to evaluate the model on.",
    )
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for evaluation."
    )
      
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--load_pose",
        action="store_true", 
        default=True,
        help="Whether to load the pose video."
    )
    parser.add_argument(
        "--use_adacn",
        action="store_true",
        default=True,
        help="Whether to use AdaCN."
    )   
    parser.add_argument(
        "--slice_frames",
        type=int,
        default=24,
        help="The number of frames to slice the video into."
    )
    parser.add_argument(
        "--pre_frames",
        type=int,
        default=8,
        help="The number of frames to preprocess the video."
    )
    parser.add_argument(
        "--eval_pair",
        action="store_true",
        help="Whether or not to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

from einops import rearrange


def repaint(
    person: torch.Tensor,
    mask: torch.Tensor,
    result: torch.Tensor,
    kernal_size: int=None,
    ):
    if kernal_size is None:
        h = person.size(-1)
        kernal_size = h // 50
        if kernal_size % 2 == 0:
            kernal_size += 1
    # Apply 2D average pooling on mask video
    # (B, C, F, H, W) -> (B*F, C, H, W)
    mask = rearrange(mask, 'b c f h w -> (b f) c h w')
    mask = torch.nn.functional.avg_pool2d(mask, kernal_size, stride=1, padding=kernal_size // 2)
    mask = rearrange(mask, '(b f) c h w -> b c f h w', b=person.size(0))
    # Use mask video to repaint result video
    result = person * (1 - mask) + result * mask
    return result

@torch.no_grad()
def main():
    args = parse_args()
    
    # Pipeline
    base_model_path = snapshot_download(args.base_model_path) if not os.path.exists(args.base_model_path) else args.base_model_path
    finetuned_model_path = snapshot_download(args.finetuned_model_path) if not os.path.exists(args.finetuned_model_path) else args.finetuned_model_path
    finetuned_model_path = os.path.join(finetuned_model_path, "512-64K" if args.dataset == "vivid" else "256-128K")
    pipeline = V2TONPipeline(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        load_pose=args.load_pose,
        torch_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
    )
    
    # Dataset
    if args.dataset == "vivid":
        dataset = ViViDTestDataset(args)

    elif args.dataset == "vvt":
        dataset = VVTTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")
    print(f"Dataset {args.dataset} loaded, total {len(dataset)} pairs.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    gt_path = os.path.join(args.output_dir, f"{args.dataset}-{args.height}", "gt")
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset}-{args.height}", "paired" if args.eval_pair else "unpaired")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader):
        num_frames = batch['video'].size(2)
        person_videos = batch['video']
        # save gt video
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        for i, video in enumerate(person_videos):
            if not os.path.exists(os.path.join(gt_path, batch['name'][i])):
                write_video(os.path.join(gt_path, batch['name'][i]), (video * 0.5 + 0.5).permute(1, 2, 3, 0).cpu() * 255, fps=24)
        mask_videos = batch['mask']
        densepose_videos = batch['densepose']
        cloth_images = batch['image']
        # Repeat the last frame to make the number of frames a multiple of 4
        original_num_frames = person_videos.size(2)
        if person_videos.size(2) % 4 != 0:
            person_videos = torch.cat([person_videos, person_videos[:, :, -1:].repeat(1, 1, 4 - person_videos.size(2) % 4, 1, 1)], dim=2)
            mask_videos = torch.cat([mask_videos, mask_videos[:, :, -1:].repeat(1, 1, 4 - mask_videos.size(2) % 4, 1, 1)], dim=2)
            densepose_videos = torch.cat([densepose_videos, densepose_videos[:, :, -1:].repeat(1, 1, 4 - densepose_videos.size(2) % 4, 1, 1)], dim=2)
        
        # Inference
        results = pipeline.video_try_on(
            source_video=person_videos,
            condition_image=cloth_images,
            mask_video=mask_videos,
            pose_video=densepose_videos if args.load_pose else None,
            num_inference_steps=args.num_inference_steps,
            slice_frames=args.slice_frames,
            pre_frames=args.pre_frames,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_adacn=args.use_adacn
        )  # [B, F, H, W, C]
        results = results.permute(0, 4, 1, 2, 3)  # [B, C, F, H, W]
        
        # Repaint
        if args.repaint:
            results = repaint(person_videos, mask_videos, results)
        if args.concat_eval_results:
            cloth_video = cloth_images.repeat(1, 1, person_videos.size(2), 1, 1)
        
        # Save results
        for i, result in enumerate(results):
            person_name = batch['name'][i]
            output_path = os.path.join(args.output_dir, person_name)
            
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
                
            if args.concat_eval_results:
                result = torch.cat([person_videos[i], cloth_video[i], mask_videos[i]], dim=-1)
                result = torch.cat([result, results[i]], dim=-1)
            
            result = result[:, :original_num_frames]
            result = result * 0.5 + 0.5
            result = (result.permute(1, 2, 3, 0).cpu() * 255).clamp(0, 255)
            # Cut the frames to the original number
            result = result[:num_frames, :, :, :]
            write_video(output_path, result, fps=24)
            

if __name__ == "__main__":
    main()