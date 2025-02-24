from torch import nn
from torch.nn import functional as F
import torch
# from modules.efficient_modules.adain import adaptive_instance_normalization


class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states


def split_condition(hidden_states):
    _, l, _ = hidden_states.size()
    # slice_size = (l // 8)  # FIXME: 仅限于 3:4 的输入
    # slices = hidden_states.split(slice_size, dim=1)
    # input_hidden_states = torch.cat([slices[i] for i in range(0, len(slices), 2)], dim=1)
    # condition_hidden_states = torch.cat([slices[i] for i in range(1, len(slices), 2)], dim=1)
    # return input_hidden_states, condition_hidden_states
    return hidden_states[:, :l // 2], hidden_states[:, l // 2:]

def interleave_condition(input_hidden_states, condition_hidden_states):
    # _, l, _ = input_hidden_states.size()
    # slice_size = (l // 4)  # FIXME: 仅限于 3:4 的输入
    # slices = [input_hidden_states[:, i * slice_size: (i + 1) * slice_size] for i in range(4)]
    # slices = [s for pair in zip(slices, condition_hidden_states.split(slice_size, dim=1)) for s in pair]
    # return torch.cat(slices, dim=1)
    return torch.cat([input_hidden_states, condition_hidden_states], dim=1)

class EfficientAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        efficient_attention=False,
        cache_kv=False,
        **kwargs
    ):  
        """
        Args:
            efficient_attention (bool, optional): 
                通过只获取 Inpainting 部分 Query, 来减少 Attention 的计算量, Key, Value 仍包含 Condition 部分
                训练时输入为 Concat 的结果，需要对两个部分进行切片, 输出再拼接 Condition 回去
                推理时如果有 KV-Cache, 则输入为 Inpainting 部分，只需要在中间的 Key, Value 部分插入 Cache
            cache_kv (bool, optional): 
                是否缓存 Condition 部分的 Key, Value, 用于在推理时进行加速
        """
        super().__init__()
        self.efficient_attention = efficient_attention  
        self.cache_kv = cache_kv
        self.cached_kv = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    
    def clear_cache(self):
        self.cached_kv = None
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        
        if self.efficient_attention and self.cached_kv is None:
            query_hidden_states, condition_hidden_states = split_condition(hidden_states)
        else:
            query_hidden_states = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(query_hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        if self.cache_kv:
            if self.cached_kv is None:  # 首次须进行 cache k, v
                self.cached_kv = []
                for tokens in [key, value]:
                    _, condition_tokens = split_condition(tokens)
                    self.cached_kv.append(condition_tokens)
            else:
                assert key.size() == self.cached_kv[0].size(), "Input KV Shape Mismatch the Cached KV, expected {} but got {}".format(self.cached_kv[0].size(), key.size())
                key = interleave_condition(key, self.cached_kv[0])
                value = interleave_condition(value, self.cached_kv[1])

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # 如果是 Efficient Attention, 需要将 Condition 部分拼接回去
        if self.efficient_attention:
            hidden_states = interleave_condition(hidden_states, condition_hidden_states)

        return hidden_states


# 适配器模块（Cross-Attention）
class MixCrossAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, dino_dim=None):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        # Feature Key and Value Projection
        self.to_k_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_hi = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states: tuple = None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        batch_size, hidden_len, token_length = hidden_states.shape
        text_tokens = encoder_hidden_states["text"]
        hi_feature = encoder_hidden_states["hi"][0]
        encoder_hidden_states["hi"] = encoder_hidden_states["hi"][1:]

        # 1. Text Embedding Cross Attention (or degrade to Self-Attention)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = (
                hidden_states.view(batch_size, channel, height * width)
                .transpose(1, 2)
                .contiguous()
            )

        batch_size, sequence_length, _ = (
            hidden_states.shape if text_tokens is None else text_tokens.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        if attn.group_norm is not None:
            hidden_states = (
                attn.group_norm(hidden_states.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(text_tokens)
        value = attn.to_v(text_tokens)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = (
            query.view(batch_size, -1, attn.heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        key = (
            key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        )
        value = (
            value.view(batch_size, -1, attn.heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = (
            hidden_states.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, -1, attn.heads * head_dim)
        )
        hidden_states = hidden_states.to(query.dtype)

        # 2. Hi-Encoder Feature Embedding Cross Attention
        if hasattr(self, "to_k_hi") and hasattr(self, "to_v_hi"):
            hi_feature = hi_feature.reshape(
                batch_size, hidden_len, self.hidden_size
            ).contiguous()
            key = self.to_k_hi(hi_feature)
            value = self.to_v_hi(hi_feature)
            key = (
                key.view(batch_size, -1, attn.heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            value = (
                value.view(batch_size, -1, attn.heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            hi_hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hi_hidden_states = (
                hi_hidden_states.transpose(1, 2)
                .contiguous()
                .reshape(batch_size, -1, attn.heads * head_dim)
            )
            hi_hidden_states = hi_hidden_states.to(query.dtype)
            hidden_states = hidden_states + 1.0 * hi_hidden_states

        # 4. Linear Projection and Dropout and Residual Connection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = (
                hidden_states.transpose(-1, -2)
                .contiguous()
                .reshape(batch_size, channel, height, width)
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
   
class PlusAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        image_size=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # w, h = image_size
        # base_size = (w * h) // 64
        # self.avaliable_token_length = [
        #     base_size // (4 ** k) for k in range(0, 4)
        # ]

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        attn_in_concat_samples=None,
        *args,
        **kwargs,
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # Plus: Concat attn_in_concat_samples 
            if has_concat_sample := attn_in_concat_samples is not None:
                attn_in_concat_sample = attn_in_concat_samples.pop(0)
                if attn_in_concat_sample.ndim == 4:
                    batch_size, channel, height, width = attn_in_concat_sample.shape
                    attn_in_concat_sample = attn_in_concat_sample.view(batch_size, channel, height * width).transpose(1, 2)
                encoder_hidden_states = torch.cat([hidden_states, attn_in_concat_sample], dim=1)
            else:
                encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # # split 2X in length
        # if has_concat_sample:
        #     hidden_states = hidden_states.split(hidden_states.size(1) // 2, dim=1)[0]
            
        return hidden_states

# Redesigned Processo
class PlusMixAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # self.to_q = nn.Linear(hidden_size, cross_attention_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, cross_attention_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, cross_attention_dim, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        attn_in_concat_samples=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if (input_ndim := hidden_states.ndim) == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        # Attention Mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        # Encoder Hidden States
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        query = attn.to_q(hidden_states)# if attn_in_concat_samples is not None else self.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states) #if attn_in_concat_samples is not None else self.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) #if attn_in_concat_samples is not None else self.to_v(encoder_hidden_states)

        # MixPlus: Concat attn_in_concat_samples 
        if attn_in_concat_samples is not None:
            attn_in_concat_sample = attn_in_concat_samples.pop(0)
            if attn_in_concat_sample.ndim == 4:
                batch_size, channel, height, width = attn_in_concat_sample.shape
                attn_in_concat_sample = attn_in_concat_sample.view(batch_size, channel, height * width).transpose(1, 2)
            key_concat = self.to_k(attn_in_concat_sample)
            value_concat = self.to_v(attn_in_concat_sample)
            key = torch.cat([key, key_concat], dim=1)
            value = torch.cat([value, value_concat], dim=1)
            
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
            
        return hidden_states

# Plus Self-Attention with AdaIN
class PlusAdaINAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        use_adain=True,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.use_adain = use_adain

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        attn_in_concat_samples=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if (input_ndim := hidden_states.ndim) == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        # Attention Mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        # XXX: AdaIN & Concat
        if attn_in_concat_samples is not None:
            attn_in_concat_sample = attn_in_concat_samples.pop(0)
            batch_size, channel, height, width = attn_in_concat_sample.shape
            if self.use_adain:
                # Project hidden_states to b,c,h,w for AdaIN
                hidden_states_ = hidden_states.transpose(1, 2).reshape(batch_size, channel, height, width)
                # AdaIN for attn_in_concat_sample
                attn_in_concat_sample = adaptive_instance_normalization(attn_in_concat_sample, hidden_states_)
            if attn_in_concat_sample.ndim == 4:
                attn_in_concat_sample = attn_in_concat_sample.view(batch_size, channel, height * width).transpose(1, 2)
            # Concat
            encoder_hidden_states = torch.cat([hidden_states, attn_in_concat_sample], dim=1)
            
        # Encoder Hidden States
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        query = attn.to_q(hidden_states)# if attn_in_concat_samples is not None else self.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states) #if attn_in_concat_samples is not None else self.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) #if attn_in_concat_samples is not None else self.to_v(encoder_hidden_states)
            
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
            
        return hidden_states



class SkipAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            return hidden_states
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states