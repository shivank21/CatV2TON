
import inspect
import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import DDPMScheduler
from einops import rearrange
from PIL import Image
from torch.nn import functional as F

from data.utils import densepose_to_rgb
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import HunyuanTransformer3DModel
from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder import (
    get_2d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
)
from modules.posenet import PoseNet
from utils import rescale_noise_cfg


def prepare_densepose(
    densepose, 
    device='cuda', 
    dtype=torch.float32
    ):
    assert isinstance(densepose, Image.Image), "Expected densepose to be a PIL.Image.Image"
    densepose = densepose_to_rgb(densepose, colormap=cv2.COLORMAP_VIRIDIS)
    return prepare_image(densepose, device, dtype)

def prepare_image(image, device='cuda', dtype=torch.float32):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    image = image.unsqueeze(2)  # add frame dim
    return image.to(device, dtype=dtype)

def init_transformer3d_model(
    base_model_path,
    finetuned_model_path,
    device: str="cuda",
    weight_dtype: torch.dtype=torch.float32,
    ):
    transformer3d = HunyuanTransformer3DModel.from_pretrained_2d(base_model_path, subfolder="transformer")
    # Set Unuse Modules to None
    for block in transformer3d.blocks:
        block.attn_temporal = None
        block.attn_norm = None
        block.attn_clip = None
        block.norm_clip = None
        block.gate_clip = None
        block.norm_clip_out = None
        block.norm2 = None
        block.attn2 = None
    # Load Attention Modules
    attn_blocks = torch.nn.ModuleList()
    for name, param in transformer3d.named_modules():
        if "attn1" in name:
            attn_blocks.append(param)
    load_checkpoint_in_model(attn_blocks, os.path.join(finetuned_model_path, "attention"))
    
    transformer3d.requires_grad_(False)           
    return transformer3d.to(device, dtype=weight_dtype)

def init_posenet_model(
    base_model_path,
    finetuned_model_path,
    device: str="cuda",
    weight_dtype: torch.dtype=torch.float32,
    ):

    config = json.load(open(os.path.join(base_model_path, "transformer", "config.json"), "r"))
    config["in_channels"] = 4
    posenet = PoseNet(**config)
    load_checkpoint_in_model(posenet, os.path.join(finetuned_model_path, "posenet"))
    posenet.requires_grad_(False)
    return posenet.to(device, dtype=weight_dtype)

def read_video_and_save_as_image(video_path, save_path):
    # Read video using cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Save frames as images
    for i, frame in enumerate(frames):
        # Build filename with sequence number
        filename = f"{save_path}_{i:04d}.jpg"
        # Convert back to BGR for saving
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame)
        
def adacn_params(content_feat, style_feat, eps=1e-5):
    """
    Adaptive Clip Normalization Parameters
    
    Args:
        content_feat (torch.Tensor): feature to be normalized
        style_feat (torch.Tensor): feature to be used for normalization
        eps (float): a small constant for numerical stability
        
    Returns:
        tuple: content_mean, content_std, style_mean, style_std
    """
    assert content_feat.size() == style_feat.size(), "content and style feature must have the same shape"
    
    b, c, t, h, w = content_feat.size()
    
    # Calculate mean and std of content feature
    content_mean = content_feat.mean(dim=(2,3,4), keepdim=True)
    content_std = torch.sqrt(content_feat.var(dim=(2,3,4), keepdim=True) + eps)
    
    # Calculate mean and std of style feature
    style_mean = style_feat.mean(dim=(2,3,4), keepdim=True)
    style_std = torch.sqrt(style_feat.var(dim=(2,3,4), keepdim=True) + eps)
    
    return content_mean, content_std, style_mean, style_std
    

class V2TONPipeline:
    def __init__(
        self,
        base_model_path,
        finetuned_model_path,
        load_pose: bool=True,
        torch_dtype: torch.dtype = torch.float32,
        device: str="cuda",
        allow_tf32: bool=True,
    ):
        self.device = device
        self.weight_dtype = torch_dtype
        self.load_pose = load_pose

        # MagViT VAE
        self.vae = AutoencoderKLMagvit.from_pretrained(base_model_path, subfolder="vae").to(self.device, dtype=self.weight_dtype)
        self.vae.requires_grad_(False)
        # Transformer3D
        self.transformer3d = init_transformer3d_model(base_model_path, finetuned_model_path, self.device, self.weight_dtype)
        # PoseNet
        self.posenet = init_posenet_model(base_model_path, finetuned_model_path, self.device, self.weight_dtype) if self.load_pose else None
        
        # Noise Schedule
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if allow_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_dit_added_args(self, height, width, batch_size):
        patch_size = self.transformer3d.config.patch_size
        inner_dim = self.transformer3d.inner_dim
        num_heads = self.transformer3d.num_heads
        device = self.device
        # create image_rotary_emb, style embedding & time ids
        grid_height = height // 8 // patch_size
        grid_width = width // 8 // patch_size
        base_size = 512 // 8 // patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size
        )
        image_rotary_emb = get_2d_rotary_pos_embed(
            inner_dim // num_heads, grid_crops_coords, (grid_height, grid_width)
        )
        style = torch.tensor([0], device=device)
        target_size = (height, width)
        add_time_ids = list((1024, 1024) + target_size + (0, 0))
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.long, device=device).repeat(
            batch_size, 1
        )
        style = style.to(device=device).repeat(batch_size)

        return {
            "image_meta_size": add_time_ids,
            "style": style,
            "image_rotary_emb": image_rotary_emb,
        }
    
        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.noise_scheduler.timesteps[
            t_start * self.noise_scheduler.order :
        ]
        return timesteps, num_inference_steps - t_start
    
    @torch.no_grad()
    def _slice_vae(self, pixel_values):
        if pixel_values.size(1) == 4:
            return pixel_values.to(self.device, self.weight_dtype)
        bs = pixel_values.shape[0]  # FIXME: Not use mini_batch
        new_pixel_values = []
        for i in range(0, bs, bs):
            pixel_values_bs = pixel_values[i : i + bs]
            pixel_values_bs = self.vae.encode(
                pixel_values_bs.to(self.device, self.weight_dtype)
            )[0]
            pixel_values_bs = pixel_values_bs.sample()
            new_pixel_values.append(pixel_values_bs)
        return torch.cat(new_pixel_values, dim=0) * self.vae.config.scaling_factor

    @torch.no_grad()
    def smooth_output(self, video, mini_batch_encoder, mini_batch_decoder=4):
        if video.size()[2] <= mini_batch_encoder:
            return video
        video = video.to(self.device, self.weight_dtype)
        prefix_index_before = mini_batch_encoder // 2
        prefix_index_after = mini_batch_encoder - prefix_index_before
        pixel_values = video[:, :, prefix_index_before:-prefix_index_after]

        # Encode middle videos
        latents = self.vae.encode(pixel_values)[0]
        latents = latents.mode()
        middle_video = self.vae.decode(latents)[0]

        video[:, :, prefix_index_before:-prefix_index_after] = (
            video[:, :, prefix_index_before:-prefix_index_after] + middle_video
        ) / 2
        return video

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        if self.vae.quant_conv.weight.ndim == 5:
            mini_batch_encoder = self.vae.mini_batch_encoder
            mini_batch_decoder = self.vae.mini_batch_decoder
            video = self.vae.decode(
                latents.to(self.device, self.weight_dtype)
            ).sample
            video = video.clamp(-1, 1)
            if video_length > mini_batch_encoder:
                print(f"Smooth output for {video_length} frames")
                video = (
                    self.smooth_output(video, mini_batch_encoder, mini_batch_decoder).cpu().clamp(-1, 1)
                )
        else:
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)

        # to b f c h w
        # video = video.permute(0, 2, 1, 3, 4).float()
        # video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().clamp(-1, 1)
        return video

    @torch.no_grad()
    def step(
        self,
        latents: torch.Tensor,
        t: int,
        inpaint_latents: torch.Tensor,
        pose_latents: torch.Tensor=None,
        guidance_scale: float=1.0,
        guidance_rescale: float=0.0,
        dit_added_args: dict=None,
        extra_step_kwargs: dict=None,
    ):
        bsz = latents.shape[0]
        timesteps = torch.tensor([t] * bsz, device=latents.device, dtype=self.weight_dtype)
        added_args = deepcopy(dit_added_args)
        
        # Pose Embedding
        if pose_latents is not None:
            pose_embeds = self.posenet(pose_latents, timesteps, return_dict=False, **added_args)
        
        # Prepare for classifier-free guidance
        latent_model_input = latents
        if (do_classifier_free_guidance := guidance_scale > 1.0):
            latent_model_input = torch.cat([latent_model_input] * 2)  
            if pose_latents is not None:
                pose_embeds = torch.cat([pose_embeds] * 2)  
            timesteps = torch.cat([timesteps] * 2)  
            added_args['image_meta_size'] = added_args['image_meta_size'].repeat(2, 1)
            added_args['style'] = added_args['style'].repeat(2)
        latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            
        # predict the noise residual
        noise_pred = self.transformer3d(
            latent_model_input,
            timesteps,
            pose_emb=pose_embeds if pose_latents is not None else None,
            inpaint_latents=inpaint_latents,
            return_dict=False,
            **added_args,
        )[0]
        noise_pred, _ = noise_pred.chunk(2, dim=1)
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        if do_classifier_free_guidance and guidance_rescale > 0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(
                noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
            )
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.noise_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]
        
        return latents
    
    @torch.no_grad()
    def denoising(
        self, 
        masked_latents: torch.Tensor,
        mask_latents: torch.Tensor,
        conditioned_latents: torch.Tensor,
        conditioning_mask_latents: torch.Tensor,
        pose_latents: torch.Tensor=None,
        noise_latents: torch.Tensor=None,
        guidance_scale: float=1.0,
        guidance_rescale: float=0.0,
        num_inference_steps: int=15,
        generator: torch.Generator=None,
        eta: float=0.0,
    ):
        """
        Denoise with latent inputs. All latents are in the same shape as (b f c h w).
        !!! VAE encode & decode will not be applied in this step !!!
        Args:
            masked_latents (torch.Tensor): masked latents (b f c h w)
            mask_latents (torch.Tensor): mask latents (b f c h w)
            conditioned_latents (torch.Tensor): conditioned latents (b f c h w)
            conditioning_mask_latents (torch.Tensor): conditioning mask latents (b f c h w)
            pose_latents (torch.Tensor, optional): pose latents (b f c h w). Defaults to None. 
            guidance_scale (float, optional): guidance scale. Defaults to 1.0.
            guidance_rescale (float, optional): guidance rescale. Defaults to 0.0.
        """
        # Check shape
        assert masked_latents.ndim == 5 and mask_latents.ndim == 5 and conditioned_latents.ndim == 5 and conditioning_mask_latents.ndim == 5, "Expected all latents to have 5 dimensions"
        if pose_latents is not None:
            assert pose_latents.shape == masked_latents.shape, "Expected pose latents to have the same shape as masked latents"
        
        # Concatenate latents
        masked_latents_cat = torch.cat([conditioned_latents, masked_latents], dim=2)
        mask_latents_cat = torch.cat([conditioning_mask_latents, mask_latents], dim=2)
        bsz, c, f, h, w = masked_latents_cat.size()
        if pose_latents is not None:
            pose_latents_cat = torch.cat([conditioning_mask_latents, pose_latents], dim=2)
        if (do_classifier_free_guidance := guidance_scale > 1.0):
            uncond_masked_latents = torch.cat([torch.zeros_like(conditioning_mask_latents), masked_latents], dim=2)
            masked_latents_cat = torch.cat([uncond_masked_latents, masked_latents_cat])
            mask_latents_cat = torch.cat([mask_latents_cat] * 2)
        inpaint_latents = torch.cat([mask_latents_cat, masked_latents_cat], dim=1)
        
        # Sample noise & Timesteps
        latents = torch.randn((bsz, c, f, h, w), device=self.device, dtype=self.weight_dtype) if noise_latents is None else noise_latents
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps,
            strength=1.0,
            device=self.device,
        )
        
        # Denoise
        dit_added_args = self.prepare_dit_added_args(h * 8, w * 8, bsz)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = self.step(
                    latents,
                    t,
                    inpaint_latents,
                    pose_latents_cat if pose_latents is not None else None,
                    guidance_scale,
                    guidance_rescale,
                    dit_added_args,
                    extra_step_kwargs,
                )
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update(1)
        
        return latents[:, :, 1:]

    @torch.no_grad()
    def image_try_on(
        self,
        source_image: Union[str, Image.Image, torch.Tensor],
        source_mask: Union[str, Image.Image, torch.Tensor],
        conditioned_image: Union[str, Image.Image, torch.Tensor],
        pose_image: Union[str, Image.Image, torch.Tensor]=None,
        num_inference_steps: int=50,
        guidance_scale: float=1.0,
        guidance_rescale: float=0.0,
        generator: torch.Generator=None,
        eta: float=0.0,
    ) -> Union[torch.Tensor, List[PIL.Image.Image]]:
        # Check Inputs
        if not isinstance(source_image, torch.Tensor):
            source_image = prepare_image(source_image, self.device, dtype=self.weight_dtype)
        if not isinstance(source_mask, torch.Tensor):
            source_mask = source_mask.convert("RGB")
            source_mask = prepare_image(source_mask, self.device, dtype=self.weight_dtype) * 0.5 + 0.5
        if not isinstance(conditioned_image, torch.Tensor):
            conditioned_image = prepare_image(conditioned_image, self.device, dtype=self.weight_dtype)
        if pose_image is not None and not isinstance(pose_image, torch.Tensor):
            pose_image = prepare_densepose(pose_image, self.device, dtype=self.weight_dtype)
        source_mask = source_mask.clamp(0, 1)
        # max pooling mask
        source_mask = F.max_pool2d(source_mask.squeeze(2), kernel_size=11, stride=1, padding=5)
        source_mask = source_mask.unsqueeze(2)
        # VAE Encoding
        masked_image = source_image * (source_mask < 0.5) + -1 * torch.ones_like(source_image) * (source_mask >= 0.5)
        masked_latents = self._slice_vae(masked_image)
        mask_latents = self._slice_vae(source_mask) 
        conditioning_mask_latents = self._slice_vae(torch.zeros_like(conditioned_image))
        conditioned_latents = self._slice_vae(conditioned_image) 
        pose_latents = None
        if pose_image is not None:
            pose_latents = self._slice_vae(pose_image)
        
        # Denoise
        output_latents = self.denoising(
            masked_latents=masked_latents,
            mask_latents=mask_latents,
            conditioned_latents=conditioned_latents,
            conditioning_mask_latents=conditioning_mask_latents,
            pose_latents=pose_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            generator=generator,
            eta=eta,
        )
        
        # VAE Decoding
        output_images = self.decode_latents(output_latents)
        output_images = output_images.squeeze(2).permute(0, 2, 3, 1).cpu().numpy()
        
        # Convert to PIL images
        output_images = (output_images * 0.5 + 0.5).clip(0, 1)
        output_pil_images = []
        for i in range(output_images.shape[0]):
            image = (output_images[i] * 255).astype(np.uint8)
            output_pil_images.append(Image.fromarray(image))
        
        return output_pil_images
    
    @torch.no_grad()
    def video_try_on(
        self,
        source_video: torch.Tensor,  # [B, C, T, H, W]
        mask_video: torch.Tensor,    # [B, C, T, H, W]
        condition_image: torch.Tensor,   # [B, C, H, W] 
        pose_video: torch.Tensor=None, # [B, C, T, H, W]
        num_inference_steps: int=15,
        guidance_scale: float=2.5,
        guidance_rescale: float=0.0,
        generator: torch.Generator=None,
        slice_frames: int=24,
        pre_frames: int=8,
        eta: float=0.0,
        use_adacn: bool=True,
    ) -> torch.Tensor:
        """
        Video Try-On
        Args:
            source_video (torch.Tensor): source video (B, C, T, H, W)
            mask_video (torch.Tensor): mask video (B, C, T, H, W)
            condition_image (torch.Tensor): condition image (B, C, H, W)
            pose_video (torch.Tensor, optional): pose video (B, C, T, H, W). Defaults to None.
            num_inference_steps (int, optional): number of inference steps. Defaults to 15.
            guidance_scale (float, optional): guidance scale. Defaults to 2.5.
            guidance_rescale (float, optional): guidance rescale. Defaults to 0.0.
            generator (torch.Generator, optional): generator. Defaults to None.
            slice_frames (int, optional): slice frames. Defaults to 24.
            pre_frames (int, optional): pre frames. Defaults to 8.
            eta (float, optional): eta. Defaults to 0.0.
            use_adacn (bool, optional): use adacn. Defaults to True.
        Returns:
            torch.Tensor: output video (B, T, H, W, C)
        """
        # VAE Encoding
        masked_video = source_video * (mask_video < 0.5) + -1 * torch.ones_like(source_video) * (mask_video >= 0.5)
        source_video_latents = self._slice_vae(masked_video)
        mask_video_latents = self._slice_vae(mask_video)
        condition_mask = torch.zeros_like(condition_image)
        condition_mask_latents = self._slice_vae(condition_mask)
        condition_image_latents = self._slice_vae(condition_image)
        if load_pose := (pose_video is not None):
            pose_video_latents = self._slice_vae(pose_video)
        del source_video, mask_video, condition_image, pose_video
            
        # Clip-based Denoising
        if not use_adacn:
            pre_frames = 0
        total_frames, slice_frames, pre_frames = source_video_latents.size(2), slice_frames // 4, pre_frames // 4
        output_latents = None
        start, end = 0, min(slice_frames, total_frames)
        while end <= total_frames:
            source_slice = source_video_latents[:, :, start:end]
            mask_slice = mask_video_latents[:, :, start:end]
            pose_slice = pose_video_latents[:, :, start:end] if load_pose else None
            
            if start > 0 and pre_frames > 0:
                source_slice[:, :, :pre_frames] = output_latents[:, :, start:start+pre_frames]
                mask_slice[:, :, :pre_frames] = condition_mask_latents.repeat(1, 1, pre_frames, 1, 1)

            result = self.denoising(
                masked_latents=source_slice,
                mask_latents=mask_slice,
                conditioned_latents=condition_image_latents,
                conditioning_mask_latents=condition_mask_latents,
                pose_latents=pose_slice,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                generator=generator,
                eta=eta,
            )
            
            if output_latents is not None:
                # Adaptive Clip Normalization
                if use_adacn:
                    overlap_frames = output_latents.size(2) - start
                    overlap_result_latents = result[:, :, :overlap_frames]
                    overlap_output_latents = output_latents[:, :, -overlap_frames:]
                    content_mean, content_std, style_mean, style_std = adacn_params(overlap_result_latents, overlap_output_latents)
                    overlap_result_latents = (overlap_result_latents - content_mean) / content_std * style_std + style_mean
                    result[:, :, overlap_frames:] = (result[:, :, overlap_frames:] - content_mean) / content_std * style_std + style_mean
                    # overlap_latents = (overlap_result_latents + overlap_output_latents) / 2
                    overlap_latents = overlap_result_latents
                    output_latents = torch.cat([output_latents[:, :, :-overlap_frames], overlap_latents, result[:, :, overlap_frames:]], dim=2)
                else:
                    if output_latents.size(2) > start:  # overlap exists
                        output_latents = torch.cat([output_latents, result[:, :, output_latents.size(2)-start:]], dim=2)
                    else:
                        output_latents = torch.cat([output_latents, result], dim=2)
            else:
                output_latents = result
                
            if end == total_frames:
                break
            
            start, end = start+ slice_frames - pre_frames, end + slice_frames - pre_frames
            if end > total_frames and start < total_frames:
                end = total_frames
                start = total_frames - slice_frames
        
        # VAE Decoding
        output_video = self.decode_latents(output_latents)
        
        # Return
        output_video = output_video.permute(0, 2, 3, 4, 1)
        return output_video