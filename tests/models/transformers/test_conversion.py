# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers' checkpoint converters, run on a stream of weights."""

import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from vllm.model_executor.models.transformers.conversion import (
    convert_weights,
    streamable_converters,
)


def _convert(model_type, weights):
    converters = streamable_converters(get_checkpoint_conversion_mapping(model_type))
    return dict(convert_weights(weights, converters))


def test_split_fused_qkv():
    # HRM's checkpoint fuses gate, q, k and v
    gqkv = torch.randn(16, 4)
    out = _convert(
        "hrm_text",
        [
            ("model.layers.0.attn.gqkv_proj.weight", gqkv),
            ("model.layers.0.input_layernorm.weight", torch.ones(4)),
        ],
    )
    prefix = "model.layers.0.self_attn."
    for i, name in enumerate(["gate_proj", "q_proj", "k_proj", "v_proj"]):
        torch.testing.assert_close(
            out[f"{prefix}{name}.weight"], gqkv[4 * i : 4 * i + 4]
        )
    assert "model.layers.0.input_layernorm.weight" in out  # untouched
    assert len(out) == 5


def test_concatenate_waits_for_every_source():
    # OLMo-Hybrid's checkpoint has one conv per q, k, v, interleaved with other keys
    prefix = "model.layers.0.linear_attn."
    q, k, v = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)
    stream = [
        (f"{prefix}q_conv1d.weight", q),
        ("model.norm.weight", torch.ones(3)),
        (f"{prefix}k_conv1d.weight", k),
        (f"{prefix}v_conv1d.weight", v),
    ]
    out = list(
        convert_weights(
            stream,
            streamable_converters(get_checkpoint_conversion_mapping("olmo_hybrid")),
        )
    )
    # The unrelated key is not held back behind the pending group
    assert [name for name, _ in out] == ["model.norm.weight", f"{prefix}conv1d.weight"]
    torch.testing.assert_close(out[1][1], torch.cat([q, k, v]))


def test_expert_merges_left_to_fused_moe():
    converters = streamable_converters(get_checkpoint_conversion_mapping("mixtral"))
    assert converters == []
    name = "model.layers.0.mlp.experts.3.gate_proj.weight"
    assert _convert("mixtral", [(name, torch.ones(1))]).keys() == {name}
