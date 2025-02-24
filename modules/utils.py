import os
import json
import torch

from easyanimate.models.transformer3d import HunyuanTransformer3DModel
from modules.attn_processors import AttnProcessor2_0, PlusMixAttnProcessor2_0

# Set Attention Processor (modified from https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py)
def init_adapter(unet, 
                 cross_attn_cls,
                 self_attn_cls=None,
                 cross_attn_dim=None, 
                 **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=hidden_size, **kwargs)
                if isinstance(attn_procs[name], PlusMixAttnProcessor2_0):
                    # Init the PlusMixAttnProcessor with the weights from the original self-attn processor
                    layer_name = name.split(".processor")[0]
                    weights = {
                        # "to_q.weight": unet_sd[layer_name + ".to_q.weight"],
                        "to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name].load_state_dict(weights) 
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
                                                    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

# Diffusion model
def init_diffusion_model(diffusion_model_name_or_path, unet_class):
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(diffusion_model_name_or_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name_or_path, subfolder="tokenizer")
    try:
        unet_folder = os.path.join(diffusion_model_name_or_path, "unet")
        unet_configs = json.load(open(os.path.join(unet_folder, "config.json"), "r"))
        unet = unet_class(**unet_configs)
        unet.load_state_dict(torch.load(os.path.join(unet_folder, "diffusion_pytorch_model.bin"), map_location="cpu"), strict=True)
    except:
        unet = None
    return text_encoder, vae, tokenizer, unet


def init_transformer3d_model(transformer3d_model_name_or_path, use_clip=False, use_text=False):
    transformer3d = HunyuanTransformer3DModel.from_pretrained_2d(transformer3d_model_name_or_path, subfolder="transformer")
    for block in transformer3d.blocks:
        block.attn_temporal = None
        block.attn_norm = None
        if not use_clip:
            block.attn_clip = None
            block.norm_clip = None
            block.gate_clip = None
            block.norm_clip_out = None
        if not use_text:
            block.norm2 = None
            block.attn2 = None
    return transformer3d

def get_trainable_module(model, trainable_module_name, load_caption=False, use_clip=False):
    if trainable_module_name == "transformer":
        return model
    elif trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, param in model.named_modules():
            if "attn1" in name:
                attn_blocks.append(param)
            if "attn2" in name and load_caption:
                attn_blocks.append(param)
            if "attn_clip" in name and use_clip:
                attn_blocks.append(param)
        print(f"Number of attention blocks: {len(attn_blocks)}")
        return attn_blocks
    else:
        raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")
    
                
def offload_unuse_modules_of_dit(dit):
    unuse_modules = torch.nn.ModuleList()
    for name, param in dit.named_modules():
        if "attn2" in name or "norm_clip" in name or "gate_clip" in name or "norm_clip_out" in name or "norm2" in name or "attn_clip" in name:
            unuse_modules.append(param)
    unuse_modules.to("cpu")
    return unuse_modules