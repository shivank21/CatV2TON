import os

import torch
from cleanfid import fid as FID
from einops import rearrange
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from data.utils import read_video_frames, scan_files_in_dir
from modules.fid_metrics.vfid import vfid
from huggingface_hub import snapshot_download

class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
    
    def extract_id_from_filename(self, filename):
        # find first number in filename
        id = filename.split('_')[0]
        return id
    
    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={'.mp4'})
        
        pred_files = scan_files_in_dir(self.pred_folder, postfix={'.mp4'})
        pred_ids = [self.extract_id_from_filename(pred_file.name) for pred_file in pred_files]
        gt_files = [file for file in gt_files if self.extract_id_from_filename(file.name) in pred_ids]
        
        gt_paths = [file.path for file in gt_files]
        gt_names = [file.name for file in gt_files]
        pred_paths = [file.path for file in pred_files if file.name in gt_names]
        gt_paths.sort()
        pred_paths.sort()
        return list(zip(gt_paths, pred_paths))
    
    # def resize(self, img):
    #     w, h = img.size
    #     new_w = int(w * self.height / h)
    #     return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]
        gt = read_video_frames(gt_path, normalize=False).squeeze(0)
        pred = read_video_frames(pred_path, normalize=False).squeeze(0)

        # crop to same long side
        min_len = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :min_len]
        pred = pred[:, :min_len]
        
        # 确保数据类型一致
        gt = gt.float()
        pred = pred.float()
        
        return gt, pred


def copy_resize_gt(gt_folder, height):
    new_folder = f"{gt_folder}_{height}"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        w, h = img.size
        new_w = int(w * height / h)
        img = img.resize((new_w, height), Image.LANCZOS)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    num_frames = 0
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        ssim_score += ssim(pred, gt) * batch_size
        num_frames += batch_size
    return ssim_score / num_frames


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to("cuda")
    score = 0
    num_frames = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        if gt.dim() == 5:
            gt = rearrange(gt, 'b c t h w -> (b t) c h w')
            pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
        num_frames += batch_size
    return score / num_frames

def compute_fvd(gt_folder, pred_folder):
    # Compute the FVD on two sets of videos.
    from cdfvd import fvd
    evaluator = fvd.cdfvd('videomae', ckpt_path="Models/vit_g_hybrid_pt_1200e_ssv2_ft.pth")
    evaluator.compute_real_stats(evaluator.load_videos(gt_folder, data_type='video_folder'))
    evaluator.compute_fake_stats(evaluator.load_videos(pred_folder, data_type='video_folder'))
    score = evaluator.compute_fvd_from_stats()
    return score
    
def eval(args):
    assert args.batch_size == 1, "Only `batch_size=1` is supported as videos with different lengths are not batchable"

    # Form dataset
    dataset = EvalDataset(args.gt_folder, args.pred_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
    )
    
    # Calculate Metrics
    header = []
    row = []
    header += ["VFID_I3D", "VFID_RESNEXT", "FVD"]
    repo_path = snapshot_download(repo_id=args.vfid_ckpt_path)
    vfid_resnext = vfid(args.gt_folder, args.pred_folder, ckpt_path=repo_path)
    row += [vfid_resnext["i3d"], vfid_resnext["resnext"]]
    fvd_score = compute_fvd(args.gt_folder, args.pred_folder)
    row += [fvd_score]
    
    if args.paired:
        header += ["SSIM", "LPIPS"]
        ssim_ = ssim(dataloader).item()
        lpips_ = lpips(dataloader).item()
        row += [ssim_, lpips_]
    
    # Print Results
    print("GT Folder  : ", args.gt_folder)
    print("Pred Folder: ", args.pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)
    print(table)
    
         
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--vfid_ckpt_path", type=str, default="zhengchong/VFID")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)  # Only 1 is supported as videos are not batchable
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    eval(args)