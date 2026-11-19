# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's checkpoint Q/K row layout, restored as the weights stream in"""

from collections.abc import Iterable

import torch


def interleave_qk(config, weights: Iterable[tuple[str, torch.Tensor]]):
    """HF stores Q/K rows as [even | odd] per head (NoPE too): interleaved pairs
    restored before TP slicing. Per-row scales move with their rows; 0-d
    scales stay."""
    qk_rows = (
        config.num_attention_heads * config.head_dim,
        config.num_key_value_heads * config.head_dim,
    )
    for name, weight in weights:
        if (
            name.startswith(
                ("model.language_model.layers.", "model.layers.", "layers.")
            )
            and (".self_attn.q_proj." in name or ".self_attn.k_proj." in name)
            and name.rsplit(".", 1)[-1].startswith("weight")
        ):
            weight = weight if isinstance(weight, torch.Tensor) else weight[...]
            if weight.dim() >= 1 and weight.shape[0] in qk_rows:
                shape = weight.shape
                weight = (
                    weight.reshape(-1, 2, config.head_dim // 2, *shape[1:])
                    .transpose(1, 2)
                    .reshape(shape)
                )
        yield name, weight
