import csv
import json
import os
import random
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video

from data.utils import densepose_to_rgb, read_video_frames, scan_files_in_dir


def mask_avg_pooling(mask: torch.Tensor, kernel_size: int=9) -> torch.Tensor:
    """Average pooling for mas video tensor in HXW dimension.
    Args:
        mask (torch.Tensor): The mask tensor with shape (T, C, H, W)
        kernel_size (int): The kernel size for avg pooling, default is 9
    Returns:
        torch.Tensor: The pooled mask tensor with the same shape (T, C, H, W)
    """
    # 确保 kernel_size 是奇数
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    # 进行平均池化
    pooled_mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    # 二值化
    pooled_mask = (pooled_mask > 0.5).float()
    return pooled_mask
    
def center_crop_and_resize_image(image, target_size):
    # 获取输入图像的宽高和目标尺寸的宽高
    img_width, img_height = image.size
    target_width, target_height = target_size
    if img_width == target_width and img_height == target_height:
        return image
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height
     
    # 按目标宽高比裁剪图像
    if img_ratio > target_ratio:
        # 图像宽于目标尺寸的比例，裁剪宽度
        new_width = int(img_height * target_ratio)
        left = (img_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, img_height))
    else:
        # 图像高于目标尺寸的比例，裁剪高度
        new_height = int(img_width / target_ratio)
        top = (img_height - new_height) // 2
        image = image.crop((0, top, img_width, top + new_height))
    
    # 调整图像大小到目标尺寸
    image = image.resize(target_size, Image.LANCZOS)
    return image

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        video_root=None,
        image_ann_path=None,
        batch_size=1,
        video_sample_size=(448, 672),
        video_sample_stride=2,
        video_sample_n_frames=24,
        enable_inpaint=False,
        load_pose=False,
        load_latent=False,
        **kwargs,
    ):
        # Video 和 Image 至少有一个不为空
        assert image_ann_path is not None or video_root is not None, "At least one of image_ann_path and video_root must be provided."
        # video_sample_size: (h, w) 和 image_sample_size: (h, w) 分别是视频和图像的采样尺寸，必须可以被16整除
        assert video_sample_size[0] % 16 == 0 and video_sample_size[1] % 16 == 0, "Video sample size must be divisible by 16."
        
        # 加载视频数据
        self.video_dataset = []
        if video_root is not None:
            for subdir in os.listdir(video_root):
                if '.' not in subdir:
                    with open(os.path.join(video_root, subdir, "train_data.jsonl"), "r") as f:
                        dataset = []
                        for line in f:
                            # {"video": "1222020_detail.mp4", "image": "1222020_in_xl.jpg", "type": "video", "front_seqs": [[35, 89]]}
                            item = json.loads(line)
                            # 判断视频 front_seqs 是否有小于 video_sample_n_frames 的序列   
                            front_seqs = []
                            for seq in item['front_seqs']:
                                if seq[1] - seq[0] >= video_sample_n_frames:
                                    front_seqs.append(seq)
                            if len(front_seqs) > 0:
                                item = {
                                    "file_path": os.path.join(video_root, subdir, "videos", item["video"]),
                                    "pose_path": os.path.join(video_root, subdir, "densepose", item["video"]),
                                    "mask_path": os.path.join(video_root, subdir, "agnostic_mask_smoothed", item["video"]),
                                    "garment_image_path": os.path.join(video_root, subdir, "images", item["image"]),
                                    "text": "",
                                    "type": "video",
                                    "frame_count": item['front_seqs'][0][1] - item['front_seqs'][0][0],
                                    'front_seqs': front_seqs,
                                }
                                dataset.append(item)
                    self.video_dataset.extend(dataset)

        # 加载图像数据
        self.image_dataset = []
        if image_ann_path is not None:
            if image_ann_path.endswith(".csv"):
                with open(image_ann_path, "r") as csvfile:
                    dataset = list(csv.DictReader(csvfile))
            elif image_ann_path.endswith(".json"):
                dataset = json.load(open(image_ann_path))
            elif image_ann_path.endswith(".jsonl"):
                dataset = []
                with open(image_ann_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        for k, v in item.items():
                            if 'path' in k:
                                item[k] = os.path.join(os.path.dirname(image_ann_path), v[1:])
                        dataset.append(item)
            else:
                raise ValueError(f"Unsupported file format of {image_ann_path}")
            self.image_dataset.extend(dataset)

        # 输出数据集大小
        assert len(self.video_dataset) > 0 or len(self.image_dataset) > 0, "No data found in the dataset."
        print(f"Video dataset size: {len(self.video_dataset)}, Image dataset size: {len(self.image_dataset)}")

        # Hyperparameters 
        self.enable_data_bucket = len(self.video_dataset) > 0 and len(self.image_dataset) > 0
        self.batch_size = batch_size
        self.batch_pos = 0
        self.video_batch_flag = len(self.video_dataset) > 0
        self.load_pose = load_pose
        self.load_latent = load_latent
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.image_sample_size = self.video_sample_size
        
        # Transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    def _switch_data_bucket(self):
        if self.enable_data_bucket:
            self.batch_pos += 1
            if self.batch_pos == (self.batch_size if self.video_batch_flag else self.batch_size * 16):
                self.batch_pos = 0
                self.video_batch_flag = not self.video_batch_flag
    
    @property
    def num_frames_of_video_dataset(self):
        # 遍历每个视频数据集，返回视频正面总帧数
        total_frames = 0
        for item in self.video_dataset:
            for seq in item['front_seqs']:
                total_frames += seq[1] - seq[0]
        return total_frames

    def __getitem__(self, idx):
        # 根据 idx 和 vidfeo_batch_flag 加载 video 或 image 数据
        if self.video_batch_flag:
            data_info = self.video_dataset[idx % len(self.video_dataset)]
        else:
            data_info = self.image_dataset[idx % len(self.image_dataset)]
            
        # 读取数据
        try:
            sample = {
                "idx": idx,
                "text": data_info["text"],
                "data_type": data_info["type"],
            }
            # Load and process Latent data
            if self.load_latent:
                frames = torch.load(data_info["file_path"].replace("-datasets", "-datasets-latent").replace(".mp4", ".pt").replace(".jpg", ".pt"), map_location='cpu')
                mask_frames = torch.load(data_info["mask_path"].replace("-datasets", "-datasets-latent").replace(".mp4", ".pt").replace(".jpg", ".pt"), map_location='cpu')
                masked_frames = torch.load(data_info['file_path'].replace("videos", "masked_videos").replace("-datasets", "-datasets-latent").replace(".mp4", ".pt").replace(".jpg", ".pt"), map_location='cpu')
                garment_image = torch.load(data_info["garment_image_path"].replace("-datasets", "-datasets-latent").replace(".mp4", ".pt").replace(".jpg", ".pt"), map_location='cpu')
                if self.load_pose:
                    assert False, "Pose is not supported for latent dataset yet."
                pose_frames = 0  # Not implemented yet
                sample.update({
                    "pixel_values": frames,
                    'masked_video': masked_frames,
                    "video_mask": mask_frames,
                    "video_pose": pose_frames,
                    "garment_image": garment_image,
                })
                self._switch_data_bucket()
                return sample
            
            # Load and process video
            if data_info["type"] == "video":
                # Get random start index
                assert 'front_seqs' in data_info and len(data_info['front_seqs']) > 0, "Front seqs not found in video data."
                front_seq = random.choice(data_info['front_seqs'])
                start_idx = random.randint(front_seq[0], front_seq[1] - self.video_sample_n_frames * self.video_sample_stride)
                # Load video frames
                end_idx = start_idx + self.video_sample_n_frames
                frames = read_video(data_info["file_path"], pts_unit='sec', output_format='TCHW')[0][start_idx:end_idx].permute(1, 0, 2, 3)
                # 视频尺寸必须是 self.video_sample_size
                assert frames.size(3) == self.video_sample_size[0] and frames.size(2) == self.video_sample_size[1], f"Video size mismatch: {frames.size()} vs {self.video_sample_size}"
                mask_frames = read_video(data_info["mask_path"], pts_unit='sec', output_format='TCHW')[0][start_idx:end_idx].permute(1, 0, 2, 3)
                frames = (frames / 255.0) * 2 - 1
                mask_frames = (mask_frames / 255.0)
                mask_frames = mask_avg_pooling(mask_frames, kernel_size=15)
                masked_frames = frames * (mask_frames < 0.5) + -1 * torch.ones_like(frames) * (mask_frames >= 0.5)
                if not self.load_pose:
                    pose_frames = 0 
                else:
                    pose_frames = read_video(data_info["pose_path"], pts_unit='sec', output_format='TCHW')[0][start_idx:end_idx].permute(1, 0, 2, 3)
                    pose_frames = (pose_frames / 255.0) * 2 - 1
            else:
                image = Image.open(data_info["file_path"]).convert("RGB")
                image = center_crop_and_resize_image(image, self.image_sample_size)
                image = self.image_transforms(image)
                frames = image.unsqueeze(1) #.repeat(1, self.video_sample_n_frames, 1, 1)
                mask_frames = Image.open(data_info["mask_path"]).convert("L")
                mask_frames = center_crop_and_resize_image(mask_frames, self.image_sample_size)
                mask_frames = transforms.ToTensor()(mask_frames).unsqueeze(1).repeat(3, 1, 1, 1)
                masked_frames = frames * (mask_frames < 0.5) + -1 * torch.ones_like(frames) * (mask_frames >= 0.5)
                if not self.load_pose:
                    pose_frames = 0 
                else:
                    pose_frames = Image.open(data_info["pose_path"])
                    pose_frames = densepose_to_rgb(pose_frames)
                    pose_frames = center_crop_and_resize_image(pose_frames, self.image_sample_size)
                    pose_frames = self.image_transforms(pose_frames).unsqueeze(1) #.repeat(1, self.video_sample_n_frames, 1, 1)
                    
            # Load and process garment image
            garment_image = Image.open(data_info["garment_image_path"]).convert("RGB")
            garment_image = center_crop_and_resize_image(garment_image, self.image_sample_size if data_info["type"] != "video" else self.video_sample_size)
            garment_image = self.image_transforms(garment_image)
            
            sample.update({
                "pixel_values": frames,
                'masked_video': masked_frames,
                "video_mask": mask_frames,
                "video_pose": pose_frames,
                "garment_image": garment_image,
            })
            self._switch_data_bucket()
            return sample
        
        except Exception as e:
            print(f"Error in get_batch for index {idx}: {e}")
            return self.__getitem__(idx + 1)
        
    def __len__(self):
        video_len = len(self.video_dataset)
        image_len = len(self.image_dataset)
        return max(video_len, image_len)


class LatentDataset(Dataset):
    def __init__(
        self, 
        video_data_root,
        image_data_root=None,
        max_frames=12,
        video_repeat=4,
        category_list=['upper_body'],
        is_test=False
    ):
        self.video_data_root = video_data_root
        self.image_data_root = image_data_root
        self.max_frames = max_frames
        self.category_list = category_list
        self.is_test = is_test
        self.video_repeat = video_repeat
        
        
        self.data = []
        self.data_stat = {}
        self._construct_video_sample()
        if self.image_data_root is not None:
            self._construct_image_sample()
        print(f"Dataset size: {len(self.data)}, {self.data_stat}")
        
    def __len__(self):
        return len(self.data)
    
    def _construct_image_sample(self):
        # XXX: Only for VITONHD dataset Now
        files = scan_files_in_dir(os.path.join(self.image_data_root, 'Images'))
        video_files = [f.path for f in files if f.path.endswith("1.pt")]
        for video_file in video_files:
            image_file = video_file.replace("1.pt", "0.pt")
            if os.path.exists(image_file):
                self.data.append({
                    'data_type': 'image',
                    'video_file': video_file,
                    'mask_file': video_file.replace("Images", "AgnosticMask"),
                    'masked_video_file': video_file.replace("Images", "masked_image"),
                    'image_file': image_file,
                    'image_clip_file': image_file.replace("Images", "images_clip"),
                })
                self.data_stat['upper_body'] += 1
                
    def _construct_video_sample(self):
        category_list = os.listdir(self.video_data_root)
        for category in category_list:
            self.data_stat[category] = 0
            if category not in self.category_list:
                continue
            with open(os.path.join(self.video_data_root, category, "train_pairs.txt" if not self.is_test else "test_pairs.txt"), "r") as f:
                train_pairs = f.readlines()                
            for pair in train_pairs:
                # 1057795_detail.mp4 1057795_in_xl.jpg  
                video_file, image_file = pair.strip().split()
                video_file = os.path.join(self.video_data_root, category, 'videos', video_file.replace(".mp4", ".pt"))
                image_file = os.path.join(self.video_data_root, category, 'images', image_file.replace(".jpg", ".pt"))
                if os.path.exists(video_file) and os.path.exists(image_file):
                    self.data.extend([{
                        'data_type': 'video',
                        'video_file': video_file,
                        'mask_file': video_file.replace("videos", "agnostic_mask"),
                        'masked_video_file': video_file.replace("videos", "masked_videos"),
                        'image_file': image_file,
                        'image_clip_file': image_file.replace("images", "images_clip"),
                    }] * self.video_repeat)
                    self.data_stat[category] += self.video_repeat   
    
    def __getitem__(self, idx):
        data = self.data[idx]
        video = torch.load(data['video_file'])
        mask = torch.load(data['mask_file'])
        masked_video = torch.load(data['masked_video_file'])
        image = torch.load(data['image_file'])
        image_clip = torch.load(data['image_clip_file']).squeeze(0)
        
        if data['data_type'] == 'image':
            video = video.repeat(1, self.max_frames, 1, 1)
            mask = mask.repeat(1, self.max_frames, 1, 1)
            masked_video = masked_video.repeat(1, self.max_frames, 1, 1)
        else:
            # 获取随机 self.max_frames 帧
            frame_count = video.size(1)
            start_idx = random.randint(0, frame_count - self.max_frames) if not self.is_test else self.max_frames
            video = video[:, start_idx:start_idx+self.max_frames]
            mask = mask[:, start_idx:start_idx+self.max_frames]
            masked_video = masked_video[:, start_idx:start_idx+self.max_frames]
        
        return {
            'video_path': data['video_file'],
            'image_path': data['image_file'],
            'video': video,
            'mask': mask,
            'masked_video': masked_video,
            'image': image,
            'image_clip': image_clip,
        }


class VVTTestDataset(Dataset):
    def __init__(self, 
        data_root_path,
        video_sample_n_frames=24,
        eval_pair=True
    ):
        self.video_sample_n_frames = video_sample_n_frames
        self.data_root_path = data_root_path
        self.eval_pair = eval_pair
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.data = self.load_data()
    
    def load_data(self):
        txt_path = os.path.join(self.data_root_path, 'test_person_clothes_pose_tuple.txt')   
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            # ve121e11m-q11 to121d05m-c12 am221d05x-c11
            person_id, clothes_id, _ = line.strip().split(' ')
            image_path = os.path.join(self.data_root_path, 'lip_clothes_person', person_id if self.eval_pair else clothes_id)
            image_path = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')][0] # image_path 下 cloth_front.jpg 结尾的文件
            
            densepose_path = os.path.join(self.data_root_path, 'test_densepose-magcianimate', person_id)
            mask_path = os.path.join(self.data_root_path, 'test_agn_mask', person_id)
            video_path = os.path.join(self.data_root_path, 'test_frames', person_id)

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

        first_image = cv2.imread(images[0])
        h, w, c = first_image.shape
        video = np.empty((len(images), h, w, c), dtype=np.uint8)

        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
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

        # Check all files exist
        exist_status = [os.path.exists(video_path), os.path.exists(image_path), os.path.exists(mask_path), os.path.exists(densepose_path)]
        if not all(exist_status):
            return self.__getitem__(idx + 1)
        
        video = self.read_image_folder_as_video(video_path) 
        mask = self.read_image_folder_as_video(mask_path) * 0.5 + 0.5
        densepose = self.read_image_folder_as_video(densepose_path)
        
        # 获取随机 self.video_sample_n_frames 帧
        frame_count = video.size(1)
        start_idx = random.randint(0, frame_count - self.video_sample_n_frames)
        video = video[:, start_idx:start_idx+self.video_sample_n_frames]
        mask = mask[:, start_idx:start_idx+self.video_sample_n_frames]
        densepose = densepose[:, start_idx:start_idx+self.video_sample_n_frames]
        
        image = Image.open(image_path)
        image = self.image_transforms(image)
        
        masked_video = video * (mask < 0.5) + -1 * torch.ones_like(video) * (mask >= 0.5)
        
        # return {
        #     'video': video,
        #     'image': image,
        #     'mask': mask,
        #     'densepose': densepose,
        #     'name': os.path.basename(video_path)[:-4] + '.mp4'
        # }
        return {
            "pixel_values": video,
            'masked_video': masked_video,
            "video_mask": mask,
            "video_pose": densepose,
            "garment_image": image,
        }
 

class ViViDTestDataset(Dataset):
    def __init__(self, 
        data_root_path,
        video_sample_n_frames=24,
        eval_pair=True
    ):
        self.data_root_path = data_root_path
        self.video_sample_n_frames = video_sample_n_frames
        self.eval_pair = eval_pair
        
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.data = self.load_data()
    
    def load_data(self):
        jsonl_path = os.path.join(self.data_root_path, 'test_data_180.jsonl' if self.eval_pair else 'test_data_180_unpaired.jsonl')   
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = json.loads(line)
            video_path = os.path.join(self.data_root_path, line['subdir'], 'videos', line['video'])
            image_path = os.path.join(self.data_root_path, line['subdir'], 'images', line['image'])
            mask_path = os.path.join(self.data_root_path, line['subdir'], 'agnostic_mask_smoothed', line['video'])
            densepose_path = os.path.join(self.data_root_path, line['subdir'], 'densepose', line['video'])
        
            data.append({
                'video': video_path,
                'image': image_path,
                'mask': mask_path,
                'densepose': densepose_path,
                'front_seqs': line['front_seqs'],
                'subdir': line['subdir']
            })
        return data

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
        
        # 读取视频帧
        video = read_video_frames(video_path, front_seqs[0], front_seqs[1]).squeeze(0)  # (C,T,H,W)
        mask = read_video_frames(mask_path, front_seqs[0], front_seqs[1], normalize=False).squeeze(0)  # (C,T,H,W)
        densepose = read_video_frames(densepose_path, front_seqs[0], front_seqs[1]).squeeze(0)  # (C,T,H,W)
        
        # 读取并处理参考图像
        image = Image.open(image_path)
        image = self.image_transforms(image)  # (C,H,W)
        
        # 生成masked视频
        masked_video = video * (mask < 0.5) + -1 * torch.ones_like(video) * (mask >= 0.5)
        
        # 返回与VVTTestDataset相同格式的数据
        return {
            "pixel_values": video,  # 原始视频
            "masked_video": masked_video,  # 遮罩后的视频
            "video_mask": mask,  # 视频遮罩
            "video_pose": densepose,  # 视频姿态
            "garment_image": image,  # 参考服装图像
        }


class SelectedDataset(Dataset):
    def __init__(
        self,
        dataset_roots: List[str],
        subset: str = 'train',
        video_sample_n_frames: int = 24,
        num_cut_transitions: int = 0,
        load_pose: bool = True,
        batch_size: int = 1,
        **kwargs
    ):
        """统一格式的数据集加载器

        Args:
            dataset_roots (List[str]): 数据集根目录列表，例如 ["Datasets/VITONHD-S", "Datasets/ViViD-S"]
            subset (str): 'train' 或 'test'
            video_sample_n_frames (int): 视频采样帧数
            load_pose (bool): 是否加载姿态信息
        """
        
        self.video_sample_n_frames = video_sample_n_frames
        self.load_pose = load_pose
        self.num_cut_transitions = num_cut_transitions
        
        # 加载所有数据集的数据
        self.video_data = []
        self.image_data = []
        self.data_stat = {
            'image': {'upper': 0, 'lower': 0, 'overall': 0}, 
            'video': {'upper': 0, 'lower': 0, 'overall': 0}, 
            'video_frames': {'upper': 0, 'lower': 0, 'overall': 0},
        }
        for dataset_root in dataset_roots:
            jsonl_path = os.path.join(dataset_root, f"{subset}_data.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping...")
                continue
            with open(jsonl_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    for k in ['video', 'image', 'densepose', 'masks']:
                        item[k] = os.path.join(dataset_root, item[k])
                    if item['video'].endswith('.mp4'):
                        if item['total_frames'] >= video_sample_n_frames + 2 * self.num_cut_transitions:
                            self.video_data.append(item)
                            self.data_stat['video'][item['category']] += 1
                            self.data_stat['video_frames'][item['category']] += item['total_frames']
                    else:
                        self.image_data.append(item)
                        self.data_stat['image'][item['category']] += 1
                    
        print(f"Loaded {len(self.video_data)} video samples and {len(self.image_data)} image samples from {len(dataset_roots)} datasets")
        print(f"Data statistics: {dict(self.data_stat)}")

        # 新增批次相关参数
        self.batch_size = batch_size
        self.batch_pos = 0
        self.video_batch_flag = len(self.video_data) > 0
        self.enable_data_bucket = len(self.video_data) > 0 and len(self.image_data) > 0
        
        # 新增图像变换
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _switch_data_bucket(self):
        """在视频和图像数据集之间切换"""
        if self.enable_data_bucket:
            self.batch_pos += 1
            # if self.batch_pos == (self.batch_size if self.video_batch_flag else self.batch_size * self.video_sample_n_frames):
            if self.batch_pos == self.batch_size:
                self.batch_pos = 0
                self.video_batch_flag = not self.video_batch_flag

    def __len__(self):
        return max(len(self.video_data), len(self.image_data))
    
    def __getitem__(self, idx):
        # 根据当前批次标志选择数据
        if self.video_batch_flag:
            data_info = self.video_data[idx % len(self.video_data)]
        else:
            data_info = self.image_data[idx % len(self.image_data)]
            
        try:
            sample = {
                "idx": idx,
                "category": data_info["category"],
                "name": os.path.basename(data_info["video"]),
                "data_type": "video" if data_info["video"].endswith('.mp4') else "image"
            }
            # 加载视频数据
            pose_frames = 0  # 防止没有姿态信息时报错
            if data_info["video"].endswith('.mp4'):
                start_idx = random.randint(self.num_cut_transitions, data_info["total_frames"] - self.video_sample_n_frames - self.num_cut_transitions)
                frames = read_video_frames(data_info["video"], start_frame=start_idx, end_frame=start_idx + self.video_sample_n_frames)[0]
                mask_frames = read_video_frames(data_info["masks"], start_frame=start_idx, end_frame=start_idx + self.video_sample_n_frames, normalize=False)[0]
                if self.load_pose:
                    pose_frames = read_video_frames(data_info["densepose"], start_frame=start_idx, end_frame=start_idx + self.video_sample_n_frames)[0]
            # 加载图片数据
            else:
                frames = self.image_transforms(Image.open(data_info["video"])).unsqueeze(1)
                mask_frames = transforms.ToTensor()(Image.open(data_info["masks"])).unsqueeze(1).repeat(3, 1, 1, 1)
                if self.load_pose:
                    pose_frames = self.image_transforms(Image.open(data_info["densepose"])).unsqueeze(1)
                # # repeat 时序维度
                # frames = frames.repeat(1, self.video_sample_n_frames, 1, 1)
                # mask_frames = mask_frames.repeat(1, self.video_sample_n_frames, 1, 1)
                # pose_frames = pose_frames.repeat(1, self.video_sample_n_frames, 1, 1)
                
            # 加载服装图片 & 理遮罩后的视频/图片
            garment_image = self.image_transforms(Image.open(data_info["image"]))
            masked_frames = frames * (mask_frames < 0.5) + -1 * torch.ones_like(frames) * (mask_frames >= 0.5)
            
            # 更新样本
            sample.update({
                "pixel_values": frames,
                'masked_video': masked_frames,
                "video_mask": mask_frames,
                "video_pose": pose_frames,
                "garment_image": garment_image,
            })
            
            # 在返回样本前切换数据桶
            self._switch_data_bucket()
            return sample
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))


if __name__ == "__main__":
    ...