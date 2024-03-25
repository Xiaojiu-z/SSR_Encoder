# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
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
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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

class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
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


class SSRAttnProcessor(nn.Module):
    r"""
    Attention processor for SSR-Adapater.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, text_context_len=77, scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.text_context_len = text_context_len
        self.scale = scale

        self.to_k_SSR = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_SSR = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
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
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # split hidden states
        encoder_hidden_states, _hidden_states = encoder_hidden_states[:, :self.text_context_len,
                                                  :], encoder_hidden_states[:, self.text_context_len:, :]

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        _key = self.to_k_SSR(_hidden_states)
        _value = self.to_v_SSR(_hidden_states)

        _key = attn.head_to_batch_dim(_key)
        _value = attn.head_to_batch_dim(_value)

        _attention_probs = attn.get_attention_scores(query, _key, None)
        _hidden_states = torch.bmm(_attention_probs, _value)
        _hidden_states = attn.batch_to_head_dim(_hidden_states)

        hidden_states = hidden_states + self.scale * _hidden_states

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


class SSRAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for SSR-Adapater for PyTorch 2.0.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, text_context_len=77, scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num = 0
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.text_context_len = text_context_len
        self.scale = scale

        self.to_k_SSR = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_SSR = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
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

        # split hidden states
        encoder_hidden_states, _hidden_states = encoder_hidden_states[:, :self.text_context_len,
                                                  :], encoder_hidden_states[:, self.text_context_len:, :]

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

        _key = self.to_k_SSR(_hidden_states)
        _value = self.to_v_SSR(_hidden_states)

        _key = _key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        _value = _value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        _hidden_states = F.scaled_dot_product_attention(
            query, _key, _value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        _hidden_states = _hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        _hidden_states = _hidden_states.to(query.dtype)

        self.num += 1
        if self.num <= 0:
            hidden_states = hidden_states
        else:
            hidden_states = hidden_states + self.scale * _hidden_states

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

class AlignerAttnProcessor1_0(nn.Module):

    def __init__(
            self,
            inner_dim=None,
            query_dim=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        self.to_out = nn.ModuleList([])
        self.to_v = nn.ModuleList([])
        for _ in range(6):
            x = nn.Linear(cross_attention_dim, inner_dim, bias=False)
            y = nn.Linear(inner_dim, query_dim, bias=False)
            self.to_v.append(x)
            self.to_out.append(y)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        encoder_hidden_states = torch.chunk(encoder_hidden_states, 6, dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states[0].shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = [attn.norm_encoder_hidden_states(encoder_hidden_state) for encoder_hidden_state in encoder_hidden_states]

        key = attn.to_k(encoder_hidden_states[-1])
        value = []
        for i in range(6):
            value.append(self.to_v[i](encoder_hidden_states[i]))

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = [attn.head_to_batch_dim(v) for v in value]

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = []
        for i in range(6):
            hidden_state = torch.bmm(attention_probs, value[i])
            hidden_state = attn.batch_to_head_dim(hidden_state)
            # linear proj
            hidden_state = self.to_out[i](hidden_state)
            if input_ndim == 4:
                hidden_state = hidden_state.transpose(-1, -2).reshape(batch_size, channel, height, width)
            hidden_state = hidden_state / attn.rescale_output_factor
            hidden_states.append(hidden_state)
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states

class AlignerAttnProcessor(torch.nn.Module):

    def __init__(
            self,
            inner_dim=None,
            query_dim=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        self.to_out = nn.ModuleList([])
        self.to_v = nn.ModuleList([])
        for _ in range(6):
            x = nn.Linear(cross_attention_dim, inner_dim, bias=False)
            y = nn.Linear(inner_dim, query_dim, bias=False)
            self.to_v.append(x)
            self.to_out.append(y)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        encoder_hidden_states = torch.chunk(encoder_hidden_states, 6, dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states[0].shape
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
            encoder_hidden_states = [attn.norm_encoder_hidden_states(encoder_hidden_state) for encoder_hidden_state in encoder_hidden_states]

        key = attn.to_k(encoder_hidden_states[-1])
        value = []
        for i in range(6):
            value.append(self.to_v[i](encoder_hidden_states[i]))

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        values = [v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for v in value]

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = []
        for i in range(6):
            hidden_states.append(F.scaled_dot_product_attention(
                query, key, values[i], attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            ))

        hidden_states = [hidden_state.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) for hidden_state in hidden_states]
        hidden_states = [hidden_state.to(query.dtype) for hidden_state in hidden_states]

        hidden_states_out = []
        for i in range(6):
            # linear proj
            hidden_states_out.append(self.to_out[i](hidden_states[i]))

        if input_ndim == 4:
            hidden_states = [hidden_state.transpose(-1, -2).reshape(batch_size, channel, height, width) for hidden_state in hidden_states_out]

        hidden_states = [hidden_state / attn.rescale_output_factor for hidden_state in hidden_states_out]

        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states


def get_attention_scores_mask(self, query, key, cross_attn_map=None, attention_mask=None,
                              weight=5):
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )  # 8 77 257

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    if cross_attn_map is not None:
        batch = key.shape[0]  # 8
        h, w = int((key.shape[1] - 1) ** 0.5), int((key.shape[1] - 1) ** 0.5)
        t = transforms.Resize([h, w])
        mask = t(cross_attn_map)
        mask = mask[:, 0:1, :, :].reshape(1, -1, h * w).repeat(batch, query.shape[1], 1)  # 8 77 256
        pad = torch.ones(batch, query.shape[1], 1, device="cuda")
        mask = torch.cat([pad, mask.to("cuda")*weight], dim=2)
        mask_1 = mask
        mask_2 = (1-mask) * weight
        attention_scores = attention_scores + mask_1 - mask_2
    else:
        attention_scores = attention_scores

    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(dtype)

    return attention_probs

class AlignerAttnProcessor_mask(nn.Module):

    def __init__(
            self,
            inner_dim=None,
            query_dim=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        self.to_out = nn.ModuleList([])
        self.to_v = nn.ModuleList([])
        for _ in range(6):
            x = nn.Linear(cross_attention_dim, inner_dim, bias=False)
            y = nn.Linear(inner_dim, query_dim, bias=False)
            self.to_v.append(x)
            self.to_out.append(y)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        mask=None,
        mask_weight=5
    ):
        self.mask = mask

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        encoder_hidden_states = torch.chunk(encoder_hidden_states, 6, dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states[0].shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = [attn.norm_encoder_hidden_states(encoder_hidden_state) for encoder_hidden_state in encoder_hidden_states]

        key = attn.to_k(encoder_hidden_states[-1])
        value = []
        for i in range(6):
            value.append(self.to_v[i](encoder_hidden_states[i]))

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = [attn.head_to_batch_dim(v) for v in value]

        attention_probs = attn.get_attention_scores_mask(query=query, key=key, cross_attn_map=self.mask, weight=mask_weight)

        hidden_states = []
        for i in range(6):
            hidden_state = torch.bmm(attention_probs, value[i])
            hidden_state = attn.batch_to_head_dim(hidden_state)
            # linear proj
            hidden_state = self.to_out[i](hidden_state)
            if input_ndim == 4:
                hidden_state = hidden_state.transpose(-1, -2).reshape(batch_size, channel, height, width)
            hidden_state = hidden_state / attn.rescale_output_factor
            hidden_states.append(hidden_state)
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states
