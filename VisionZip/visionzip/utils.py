import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union

from transformers.models.clip.modeling_clip import CLIPEncoderLayer


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    raw_key_states = key_states.clone()
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = (
            attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            + causal_attention_mask
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = (
            attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(
            bsz * self.num_heads, tgt_len, src_len
        )
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, raw_key_states.mean(1)


def CLIP_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)

    hidden_states, attn_weights, metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    r = self._info["r"].pop(0)
    if r > 0:
        self.metric = metric
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    if isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def make_tome_class(transformer_class):
    class VisionZipTransformer(transformer_class):
        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._info["r"] = parse_r(len(self.vision_model.encoder.layers), self.r)
            self._info["size"] = None
            self._info["source"] = None
            return super().forward(*args, **kwargs)

    return VisionZipTransformer


def apply_info(model, dominant_num, contextual_num):
    VisionZipTransformer = make_tome_class(model.__class__)

    model.__class__ = VisionZipTransformer
    model.r = [0] * 22 + [1, 0]

    model._info = {
        "r": [model.r],
        "dominant": dominant_num,
        "contextual": contextual_num,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module._info = model._info
