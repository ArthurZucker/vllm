# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaders that quantize each arriving tensor (direct reload of online quant)."""

import torch

from vllm.model_executor.models.transformers.conversion import VllmQuantizeQuantizer


def test_quantizer_gets_contiguous_tensor():
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(torch.empty(4, 6, dtype=torch.int8), False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(()), False)

    def quantize(w):
        assert w.is_contiguous()
        return w.to(torch.int8), w.float().abs().amax()

    root = torch.nn.Module()
    root.layer = layer
    op = VllmQuantizeQuantizer({"weight": (quantize, ["weight_scale"], False)})
    transposed = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4).t()
    out = op.get_quantize_ops().convert(
        {"layer.weight": [transposed]},
        [],
        [],
        full_layer_name="layer.weight",
        model=root,
    )
    assert torch.equal(out["layer.weight"], transposed.to(torch.int8))
    assert out["layer.weight_scale"].item() == 23
