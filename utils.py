import math
import os
import random
from typing import List, Optional, Tuple

import accelerate
import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from packaging import version
from PIL import Image
from torch.nn import functional as F


def compute_dream_and_update_latents_for_inpaint(
    unet,
    noise_scheduler,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor=None,
    dream_detail_preservation: float = 1.0,
    attention_in_add_samples: List[torch.Tensor]=None,
    attention_in_concat_samples: List[torch.Tensor]=None,
    **kwargs,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None
    with torch.no_grad():
        pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            attention_in_add_samples=attention_in_add_samples,
            attention_in_concat_samples=attention_in_concat_samples,
            **kwargs,
        ).sample

    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        noisy_latents = noisy_latents[:, :4, :, :] # Split the latents into noised part
        _noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return _noisy_latents, _target


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def compute_vae_encodings(images: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model

    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

def compute_vae_decodings(latents: torch.Tensor, vae: torch.nn.Module) -> PIL.Image.Image:
    
    # VAE Decoding
    latents = (1 / vae.config.scaling_factor * latents).to(vae.device, dtype=vae.dtype)
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    return image

def generate_noise(bs, channel, length, height, width, ratio=0.5, generator=None, device="cuda", dtype=None):
    noise = torch.randn(bs, channel, length, height, width, generator=generator, device=device, dtype=dtype)
    for i in range(1, length):
        noise[:, :, i, :, :] = ratio * noise[:, :, i - 1, :, :] + (1 - ratio) * noise[:, :, i, :, :]
    return noise

def init_accelerator(config):
    # 添加环境变量来避免NCCL错误
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 添加这行
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
    )
    accelerator_ddp_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_ddp_config],
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "train_batch_size": config.train_batch_size,
                "image_size": f"{config.width}x{config.height}",
            },
        )
    return accelerator


def init_weight_dtype(wight_dtype):
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def prepare_masked_video(pixel_values, mask):
    """Prepare masked video by applying mask to pixel values.

    Args:
        pixel_values (torch.Tensor): Pixel values of video, shape [batch, channels, frames, height, width]
        mask (torch.Tensor): Mask tensor, shape [batch, channels, frames, height, width]

    Returns:
        torch.Tensor: Masked pixel values, shape [batch, channels, frames, height, width]
    """
    assert pixel_values.shape == mask.shape, "Pixel values and mask shape mismatch, but got {} and {}".format(pixel_values.shape, mask.shape)
    masked_pixel_values = pixel_values * (mask < 0.5) + torch.ones_like(pixel_values) * (mask > 0.5) * -1
    return masked_pixel_values

# Image -> Tensor
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
    return image.to(device, dtype=dtype)


def prepare_mask_image(mask_image, device='cuda', dtype=torch.float32):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            # Batch and add channel dim for single mask
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            # Single mask, the 0'th dimension is considered to be
            # the existing batch size of 1
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            # Batch of mask, the 0'th dimension is considered to be
            # the batching dimension
            mask_image = mask_image.unsqueeze(1)

        # Binarize mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]

        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

    return mask_image.to(device, dtype=dtype)


def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    image = image * (1 - mask) + mask * 128
    return Image.fromarray(image.astype(np.uint8)).convert("RGB")

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor_to_image(tensor: torch.Tensor):
    """
    Converts a torch tensor to PIL Image.
    """
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    if tensor.min() < 0:
        tensor = tensor + 1
        tensor = tensor / 2
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


def concat_images(images: List[Image.Image], divider: int = 4, cols: int = 4):
    """
    Concatenates images horizontally and with
    """
    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]
    total_width = cols * max(widths)
    total_width += divider * (cols - 1)
    # `col` images each row
    rows = math.ceil(len(images) / cols)
    total_height = max(heights) * rows
    # add divider between rows
    total_height += divider * (len(heights) // cols - 1)

    # all black image
    concat_image = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        concat_image.paste(image, (x_offset, y_offset))
        x_offset += image.size[0] + divider
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += image.size[1] + divider

    return concat_image


def read_prompt_file(prompt_file: str):
    if prompt_file is not None and os.path.isfile(prompt_file):
        with open(prompt_file, "r") as sample_prompt_file:
            sample_prompts = sample_prompt_file.readlines()
            sample_prompts = [sample_prompt.strip() for sample_prompt in sample_prompts]
    else:
        sample_prompts = []
    return sample_prompts


def save_tensors_to_npz(tensors: torch.Tensor, paths: List[str]):
    assert len(tensors) == len(paths), "Length of tensors and paths should be the same!"
    for tensor, path in zip(tensors, paths):
        np.savez_compressed(path, latent=tensor.cpu().numpy())


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def is_xformers_available():
    try:
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                "please update xFormers to at least 0.0.17. "
                "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        return True
    except ImportError:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )

if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageFilter
    from torchvision import transforms

    def vis_sobel_weight(image_path, mask_path) -> PIL.Image.Image:

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        l_w, l_h = w // 8, h // 8
        image = image.resize((l_w, l_h))
        mask = Image.open(mask_path).convert("L").resize((l_w, l_h))
        image_pt = transforms.ToTensor()(image).unsqueeze(0).to("cuda")
        mask_pt = transforms.ToTensor()(mask).unsqueeze(0).to("cuda")
        sobel_pt = sobel(image_pt, mask_pt, scale=1.0)
        sobel_image = sobel_pt.squeeze().cpu().numpy()
        sobel_image = Image.fromarray((sobel_image * 255).astype(np.uint8))
        sobel_image = sobel_image.resize((w, h), resample=Image.NEAREST)
        # 图像平滑
        sobel_image = sobel_image.filter(ImageFilter.SMOOTH)
        from data.utils import grayscale_to_heatmap

        sobel_image = grayscale_to_heatmap(sobel_image)
        image = Image.open(image_path).convert("RGB").resize((w, h))
        sobel_image = Image.blend(image, sobel_image, alpha=0.5)
        return sobel_image

    save_folder = "./sobel_vis-2.0"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    from data.utils import scan_files_in_dir

    for i in scan_files_in_dir(
        "/home/chongzheng/Projects/try-on-project/Datasets/VITONHD-1024/test/Images"
    ):
        image_path = i.path

        if i.path.endswith("-1.jpg"):
            result_path = os.path.join(save_folder, os.path.basename(image_path))

            mask_path = image_path.replace("Images", "AgnosticMask").replace(
                "-1.jpg", "_mask-1.png"
            )
            vis_sobel_weight(image_path, mask_path).save(result_path)
    pass
