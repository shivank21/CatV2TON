from typing import Optional
from einops import rearrange
from torch import nn
from diffusers.models.embeddings import PatchEmbed
import torch
from easyanimate.models.attention import HunyuanDiTBlock
from easyanimate.models.embeddings import HunyuanCombinedTimestepTextSizeStyleEmbedding
                        
class PoseNet(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        n_query=16,
        projection_dim=768,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cross_attention_dim: int = 1024,
        norm_type: str = "layer_norm",
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
        # block type
        basic_block_type: str = "basic",
        # motion module kwargs
        motion_module_type = "VanillaGrid",
        motion_module_kwargs = None,
        motion_module_kwargs_odd = None,
        motion_module_kwargs_even = None,
        time_position_encoding = False,
        after_norm = False,
        **kwargs
    ):
        super(PoseNet, self).__init__()
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.basic_block_type = basic_block_type
        
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )
        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
        )
        self.block = HunyuanDiTBlock(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            activation_fn=activation_fn,
            ff_inner_dim=int(self.inner_dim * mlp_ratio),
            cross_attention_dim=cross_attention_dim,
            qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
            skip=False,
            after_norm=after_norm,
            time_position_encoding=time_position_encoding,
            enable_inpaint=False,
        )

    def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            text_embedding_mask=None,
            encoder_hidden_states_t5=None,
            text_embedding_mask_t5=None,
            image_meta_size=None,
            style=None,
            image_rotary_emb=None,
            inpaint_latents=None,
            clip_encoder_hidden_states: Optional[torch.Tensor]=None,
            clip_attention_mask: Optional[torch.Tensor]=None,
            return_dict=True,
        ):
            # unpatchify: (N, out_channels, H, W)
            patch_size = self.pos_embed.patch_size
            video_length, height, width = hidden_states.shape[-3], hidden_states.shape[-2] // patch_size, hidden_states.shape[-1] // patch_size
            hidden_states = rearrange(hidden_states, "b c f h w ->(b f) c h w")
            hidden_states = self.pos_embed(hidden_states)
            hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
            
            temb = self.time_extra_emb(
                timestep, encoder_hidden_states_t5,
                image_meta_size, style, hidden_dtype=timestep.dtype
            )  # [B, D]
            
            kwargs = {
                "basic": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                "hybrid_attention": {"num_frames":video_length, "height":height, "width":width, "clip_encoder_hidden_states":clip_encoder_hidden_states},
                "motionmodule": {"num_frames":video_length, "height":height, "width":width},
                "global_motionmodule": {"num_frames":video_length, "height":height, "width":width},
            }[self.basic_block_type]
            hidden_states = self.block(
                hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,  
                **kwargs
            )  # (N, L, D)
            
            return hidden_states