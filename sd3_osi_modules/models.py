from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class JointAttnProcessor2_0_custom:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        target_tokens: Optional[List[List[int]]] = [],
        target_tokens_eng: Optional[List[str]] = [],
        direction: Optional[torch.Tensor] = None,
        alpha: Optional[float] = -10.0,
        mode: str = 'steer',
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        batch_size = hidden_states.shape[0]
        num_image_tokens = hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # T5 text tokens start at num_image_tokens + 77 (after image tokens and CLIP-77 tokens)
        if mode == 'collect':
            target_key = {}
            for idx, target_ in enumerate(target_tokens):
                target_idx = [i_ + num_image_tokens + 77 for i_ in target_]
                target_key[target_tokens_eng[idx]] = key[:, :, target_idx].mean(dim=2, keepdim=True).detach().cpu()
            attn.target_key = target_key

        if mode == 'steer' and direction is not None:
            direction_temp = direction.unsqueeze(0).to(hidden_states.device).type(hidden_states.dtype)
            dir = direction_temp[:, :, :64].unsqueeze(2)  # (1, 24, 1, 64)
            for target_ in target_tokens:
                target_idx = [i_ + num_image_tokens + 77 for i_ in target_]
                key[1, :, target_idx] = key[1, :, target_idx] - dir * alpha

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_attention_kwargs = joint_attention_kwargs or {}
    if self.use_dual_attention:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=temb
        )
    else:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    if self.context_pre_only:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        **joint_attention_kwargs,
        **kwargs,
    )

    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    if self.use_dual_attention:
        attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
        attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
        hidden_states = hidden_states + attn_output2

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    if self._chunk_size is not None:
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output

    if self.context_pre_only:
        encoder_hidden_states = None
    else:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        if self._chunk_size is not None:
            context_ff_output = _chunked_feed_forward(
                self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    return encoder_hidden_states, hidden_states


def model_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    skip_layers: Optional[List[int]] = None,
    direction: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Transformer2DModelOutput]:

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
        joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

    for index_block, block in enumerate(self.transformer_blocks):
        is_skip = True if skip_layers is not None and index_block in skip_layers else False

        if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block.block_forward,
                hidden_states,
                encoder_hidden_states,
                temb,
                joint_attention_kwargs,
                direction=None if direction is None else direction[index_block],
                **kwargs,
            )
        elif not is_skip:
            encoder_hidden_states, hidden_states = block.block_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
                direction=None if direction is None else direction[index_block],
                **kwargs,
            )

        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
