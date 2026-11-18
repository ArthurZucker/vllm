# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion ops vLLM adds to Transformers' (`transformers.core_model_loading`)."""

import torch
from transformers.core_model_loading import Concatenate, ConversionOps


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
        tensor = next(iter(input_dict.values()))
        tensor = tensor[0] if isinstance(tensor, list) else tensor
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
        present = next(iter(input_dict.values()))
        present = present[0] if isinstance(present, list) else present
        for pattern in source_patterns:
            input_dict.setdefault(pattern, [torch.zeros_like(present)])
        return input_dict
