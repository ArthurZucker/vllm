# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion ops vLLM adds to Transformers' (`transformers.core_model_loading`)."""

import torch
from transformers.core_model_loading import Concatenate, ConversionOps


def _each(input_dict: dict, fn) -> dict:
    """`fn` over every tensor, Transformers' lists of them as well"""
    return {
        k: [fn(t) for t in v] if isinstance(v, list) else fn(v)
        for k, v in input_dict.items()
    }


def _first(input_dict: dict) -> torch.Tensor:
    """The one tensor of a single-source op (Transformers passes lists)"""
    tensor = next(iter(input_dict.values()))
    return tensor[0] if isinstance(tensor, list) else tensor


class OutputConcatenate(Concatenate):
    """`Concatenate` along the target param's `output_dim` (an AWQ `qweight` packs its
    output along dim 1); 0-dim tensors (per-tensor scales) are stacked"""

    def __init__(self):
        super().__init__(dim=0)

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        model, name = kwargs.get("model"), kwargs.get("full_layer_name", "")
        try:
            param = model.get_parameter(name)
        except (AttributeError, TypeError):
            param = None
        self.dim = getattr(param, "output_dim", None) or 0
        return super().convert(input_dict, source_patterns, target_patterns, **kwargs)


class PadToParam(ConversionOps):
    """The tensor padded to its target param's shape with the param's `pad_value`
    (default 0): a vocab padded to a multiple of the ranks, sink heads (-inf)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        param = kwargs["model"].get_parameter(kwargs["full_layer_name"])
        tensor = _first(input_dict)
        if tensor.shape != param.shape and tensor.dim() == param.dim():
            padded = tensor.new_full(param.shape, getattr(param, "pad_value", 0))
            padded[tuple(slice(0, n) for n in tensor.shape)] = tensor
            tensor = padded
        return {target_patterns[0]: tensor}


class RepeatLastSource(ConversionOps):
    """The last source taken twice: a checkpoint shipping `k_proj` only, loaded into
    both the K and V shards of a fused `qkv_proj` (Gemma 4's `attention_k_eq_v`)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        last = input_dict[source_patterns[-1]]
        input_dict[source_patterns[-1]] = (
            [*last, *last] if isinstance(last, list) else [last, last]
        )
        return input_dict


class ZeroMissingSource(ConversionOps):
    """A source the checkpoint does not ship, as zeros shaped like a present one:
    Whisper's `k_proj` has no bias, the fused `qkv_proj.bias` has a K slot"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        present = _first(input_dict)
        for pattern in source_patterns:
            input_dict.setdefault(pattern, [torch.zeros_like(present)])
        return input_dict


class NegExp(ConversionOps):
    """`-exp(x)`, in float32: Mamba's `A` from the checkpoint's `A_log`"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return _each(input_dict, lambda t: -torch.exp(t.float()))


class Affine(ConversionOps):
    """`x * scale + offset`, for norms stored shifted or scaled (PLaMo 3's `1 + w`,
    LongCat's MLA LoRA scaling)"""

    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale, self.offset = scale, offset

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return _each(input_dict, lambda t: t * self.scale + self.offset)


class Gather(ConversionOps):
    """This rank's shard reordered by `index` along `dim` (DeepSeek V4.1's fused
    layouts); 1-byte dtypes (fp8, ue8m0) are gathered as bytes"""

    def __init__(self, index: torch.Tensor, dim: int):
        self.index, self.dim = index, dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        def gather(t):
            raw = t.view(torch.uint8) if t.element_size() == 1 else t
            return raw.index_select(self.dim, self.index.to(t.device)).view(t.dtype)

        return _each(input_dict, gather)


class SwapHalves(ConversionOps):
    """The two halves of `dim` swapped: a fused projection stored `[up, gate]` where
    the param is `[gate, up]` (GTE's `up_gate_proj`)"""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        first, second = tensor.chunk(2, dim=self.dim)
        return {target_patterns[0]: torch.cat([second, first], dim=self.dim)}


class Deinterleave(ConversionOps):
    """A fused projection stored per head (`[heads, parts, head_dim]`, Bloom's
    `query_key_value`) as its parts one after the other (`[parts, heads, head_dim]`)"""

    def __init__(self, parts: int, heads_attr: str = "num_attention_heads"):
        self.parts, self.heads_attr = parts, heads_attr

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        heads = getattr(kwargs["config"].get_text_config(), self.heads_attr)
        shape = tensor.shape
        tensor = tensor.view((heads, self.parts, -1) + shape[1:]).transpose(0, 1)
        return {target_patterns[0]: tensor.reshape(shape)}


class ExpertRows(ConversionOps):
    """Experts stacked along the rows (`[experts * rows, cols]`, Nomic MoE) as
    `[experts, rows, cols]`, `experts` read from the config"""

    def __init__(self, experts_attr: str = "num_experts"):
        self.experts_attr = experts_attr

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        experts = getattr(kwargs["config"].get_text_config(), self.experts_attr)
        return {target_patterns[0]: tensor.view(experts, -1, *tensor.shape[1:])}


class HeadPermute(ConversionOps):
    """Each head's channels gathered by `index` (`head_dim` channels a head), on the
    sources at `positions` (all by default): K2-Horizon's partial-RoPE channels
    reordered for NeoX rope. Per head: sharded heads permute alone"""

    def __init__(self, index: torch.Tensor, head_dim: int, positions=None):
        self.index, self.head_dim, self.positions = index, head_dim, positions

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        chosen = [
            p
            for i, p in enumerate(source_patterns)
            if self.positions is None or i in self.positions
        ]
        for pattern in chosen:
            if (value := input_dict.get(pattern)) is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            input_dict[pattern] = [
                t.view(-1, self.head_dim, *t.shape[1:])[:, self.index.to(t.device)]
                .reshape(t.shape)
                .contiguous()
                for t in tensors
            ]
        return input_dict
