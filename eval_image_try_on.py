import argparse
import os

import cv2
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modules.densepose import DensePose, densepose_to_rgb
from modules.pipeline import V2TONPipeline


class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
        self.data = self.load_data()
    
    def load_data(self):
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask, pose = [Image.open(data[key]) for key in ['person', 'cloth', 'mask', 'pose']]
        pose = densepose_to_rgb(pose, colormap=cv2.COLORMAP_VIRIDIS)
        mask = mask.convert('RGB')
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0].unsqueeze(1),
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0].unsqueeze(1),
            'pose': self.vae_processor.preprocess(pose, self.args.height, self.args.width)[0].unsqueeze(1),
            'mask': self.vae_processor.preprocess(mask, self.args.height, self.args.width)[0].unsqueeze(1)    
        }

class VITONHDTestDataset(InferenceDataset):
    def load_data(self):
        assert os.path.exists(pair_txt:=os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
        output_dir = os.path.join(self.args.output_dir, f"vitonhd-{self.args.height}", 'unpaired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")
            if not os.path.exists(os.path.join(output_dir, person_img)):
                if self.args.eval_pair:
                    cloth_img = person_img
                data.append({
                    'person_name': person_img,
                    'person': os.path.join(self.args.data_root_path, 'image', person_img),
                    'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                    'pose': os.path.join(self.args.data_root_path, 'densepose_gray', person_img.replace('.jpg', '.png')),
                    'mask': os.path.join(self.args.data_root_path, 'agnostic-mask-new', person_img.replace('.jpg', '.png')),
                })
        return data

class DressCodeTestDataset(InferenceDataset):
    def load_data(self):
        data = []
        for sub_folder in ['dresses', 'lower_body', 'upper_body',]:
            assert os.path.exists(os.path.join(self.args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
            pair_txt = os.path.join(self.args.data_root_path, sub_folder, 'test_pairs_paired.txt' if self.args.eval_pair else 'test_pairs_unpaired.txt')
            assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
            with open(pair_txt, 'r') as f:
                lines = f.readlines()

            output_dir = os.path.join(self.args.output_dir, f"dresscode-{self.args.height}", 
                                        'unpaired' if not self.args.eval_pair else 'paired', sub_folder)
            for line in lines:
                person_img, cloth_img = line.strip().split(" ")
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                data.append({
                    'person_name': os.path.join(sub_folder, person_img),
                    'person': os.path.join(self.args.data_root_path, sub_folder, 'images', person_img),
                    'cloth': os.path.join(self.args.data_root_path, sub_folder, 'images', cloth_img),
                    'pose': os.path.join(self.args.data_root_path, sub_folder, 'densepose_gray', person_img.replace('.jpg', '.png')),
                    'mask': os.path.join(self.args.data_root_path, sub_folder, 'agnostic_masks', person_img.replace('.jpg', '.png'))
                })
        return data

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
        "--resume_path",
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
        "--dataset_name",
        type=str,
        required=True,
        help="The datasets to use for evaluation, vitonhd or dresscode."
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
        "--batch_size", type=int, default=8, help="The batch size for evaluation."
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
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


def check_conditions(
    data_root_path,
    resume_path,
    dataset_name="vitonhd"
    ):
    
    from modules.cloth_masker import AutoMasker
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(resume_path, "DensePose"),
        schp_ckpt=os.path.join(resume_path, "SCHP"),
        device='cuda', 
    )
    
    if dataset_name == "vitonhd":
        image_paths = os.listdir(os.path.join(data_root_path, 'test', 'image'))
        for image_path in tqdm([_ for _ in image_paths if _.endswith('.jpg')], desc='Preprocessing conditions for VITONHD'):
            densepose_path = os.path.join(data_root_path, 'test', 'densepose_gray', image_path.replace('.jpg', '.png'))
            mask_path = os.path.join(data_root_path, 'test', 'agnostic-mask-new', image_path.replace('.jpg', '.png'))
            if not os.path.exists(os.path.dirname(densepose_path)):
                os.makedirs(os.path.dirname(densepose_path))
            if not os.path.exists(os.path.dirname(mask_path)):
                os.makedirs(os.path.dirname(mask_path))

            if not os.path.exists(mask_path):
                image = Image.open(os.path.join(data_root_path, 'test', 'image', image_path))
                conditions = automasker(image, mask_type='upper')
                conditions['densepose'].save(densepose_path)
                conditions['mask'].save(mask_path)
            elif not os.path.exists(densepose_path):
                image = Image.open(os.path.join(data_root_path, 'test', 'image', image_path))
                densepose_gray = automasker.densepose_processor(image, resize=512, colormap=None)
                densepose_gray.save(densepose_path)
                
    elif dataset_name == 'dresscode':
        for sub_folder in ['upper_body', 'lower_body', 'dresses']:
            part = {'upper_body': 'upper', 'lower_body': 'lower', 'dresses': 'overall'}[sub_folder]
            image_paths = os.listdir(os.path.join(data_root_path, sub_folder, 'images'))
            for image_path in tqdm([_ for _ in image_paths if _.endswith('0.jpg')], desc=f'Preprocessing conditions for DressCode/{sub_folder}'):
                densepose_path = os.path.join(data_root_path, sub_folder, 'densepose_gray', image_path.replace('.jpg', '.png'))
                mask_path = os.path.join(data_root_path, sub_folder, 'agnostic_masks', image_path.replace('.jpg', '.png'))
                if os.path.exists(densepose_path) and os.path.exists(mask_path):
                    continue
                if not os.path.exists(os.path.dirname(densepose_path)):
                    os.makedirs(os.path.dirname(densepose_path))
                if not os.path.exists(os.path.dirname(mask_path)):
                    os.makedirs(os.path.dirname(mask_path))

                if not os.path.exists(mask_path):
                    image = Image.open(os.path.join(data_root_path, sub_folder, 'images', image_path))
                    conditions = automasker(image, mask_type='upper')
                    conditions['densepose'].save(densepose_path)
                    conditions['mask'].save(mask_path)
                elif not os.path.exists(densepose_path):
                    image = Image.open(os.path.join(data_root_path, sub_folder, 'images', image_path))
                    densepose_gray = automasker.densepose_processor(image, resize=512, colormap=None)
                    densepose_gray.save(densepose_path)

def check_densepose(
    data_root_path,
    densepose_ckpt_path,
    dataset_name="vitonhd"
    ):
    """Preprocess densepose_gray for VITONHD dataset.
    Args:
        data_root_path (str): Path to the dataset to evaluate.
        densepose_ckpt_path (str): Path to the densepose checkpoint.
    """
    if dataset_name == "vitonhd":
        densepose_path = os.path.join(data_root_path, 'test', 'densepose_gray')
        if not os.path.exists(densepose_path):
            os.makedirs(densepose_path)
            densepose_model = DensePose(
                model_path=densepose_ckpt_path,
                device='cuda'
            )
            image_paths = os.listdir(os.path.join(data_root_path, 'test', 'image'))
            for image_path in tqdm([_ for _ in image_paths if _.endswith('.jpg')], desc='Preprocessing densepose_gray for VITONHD'):
                image = Image.open(os.path.join(data_root_path, 'test', 'image', image_path))
                densepose_gray = densepose_model(image, resize=512, colormap=None)
                densepose_gray.save(os.path.join(densepose_path, image_path.replace('.jpg', '.png')))
    elif dataset_name == 'dresscode':
        densepose_model = None
        for sub_folder in ['upper_body', 'lower_body', 'dresses']:
            densepose_path = os.path.join(data_root_path, sub_folder, 'densepose_gray')
            if not os.path.exists(densepose_path):
                os.makedirs(densepose_path)
                densepose_model = DensePose(
                    model_path=densepose_ckpt_path,
                    device='cuda'
                ) if densepose_model is None else densepose_model
                image_paths = os.listdir(os.path.join(data_root_path, sub_folder, 'images'))
                for image_path in tqdm([_ for _ in image_paths if _.endswith('0.jpg')], desc=f'Preprocessing densepose_gray for DressCode/{sub_folder}'):
                    image = Image.open(os.path.join(data_root_path, sub_folder, 'images', image_path))
                    densepose_gray = densepose_model(image, resize=512, colormap=None)
                    densepose_gray.save(os.path.join(densepose_path, image_path.replace('.jpg', '.png')))
    else:
        raise ValueError(f"Invalid dataset name {dataset_name}.")
        
def repaint(person, mask, result, use_gaussian_blur=True):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    if use_gaussian_blur:
        mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def main():
    args = parse_args()
    
    # Pipeline
    base_model_path = snapshot_download(args.base_model_path) if not os.path.exists(args.base_model_path) else args.base_model_path
    finetuned_model_path = snapshot_download(args.resume_path) if not os.path.exists(args.resume_path) else args.resume_path
    finetuned_model_path = os.path.join(finetuned_model_path, "512-64K")
    pipeline = V2TONPipeline(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        torch_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        load_pose=True
    )
    
    # Dataset
    catvton_ckpt_path = snapshot_download(args.catvton_ckpt_path) if not os.path.exists(args.catvton_ckpt_path) else args.catvton_ckpt_path
    densepose_ckpt_path = os.path.join(catvton_ckpt_path, 'DensePose')
    check_densepose(args.data_root_path, 
                    densepose_ckpt_path,
                    dataset_name=args.dataset_name)
    check_conditions(args.data_root_path, 
                    catvton_ckpt_path, 
                    dataset_name=args.dataset_name)
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTestDataset(args)
    elif args.dataset_name == "dresscode":
        dataset = DressCodeTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}-{args.height}", "paired" if args.eval_pair else "unpaired")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader):
        person_images = batch['person']
        cloth_images = batch['cloth']
        pose_images = batch['pose']
        masks = batch['mask']
        results = pipeline.image_try_on(
            source_image=person_images,
            source_mask=masks,
            conditioned_image=cloth_images,
            pose_image=pose_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        
        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images.squeeze(2))
            cloth_images = to_pil_image(cloth_images.squeeze(2))
            pose_images = to_pil_image(pose_images.squeeze(2))
            masks = to_pil_image(masks.squeeze(2))
        for i, result in enumerate(results):
            person_name = batch['person_name'][i]
            output_path = os.path.join(args.output_dir, person_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if args.repaint:
                person_path, mask_path = dataset.data[batch['index'][i]]['person'], dataset.data[batch['index'][i]]['mask']
                person_image= Image.open(person_path).resize(result.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result.size, Image.NEAREST).convert('RGB')
                result = repaint(person_image, mask, result)
            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new('RGB', (w*5, h))
                concated_result.paste(person_images[i], (0, 0))
                concated_result.paste(masks[i], (w, 0))
                concated_result.paste(pose_images[i], (w*2, 0))
                concated_result.paste(cloth_images[i], (w*3, 0))
                concated_result.paste(result, (w*4, 0))
                result = concated_result
            result.save(output_path)

if __name__ == "__main__":
    main()