# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling's weights: its MoE experts by `InklingMoE.load_expert_weight` (rows
interleaved gate/up, `input_amax` scales, sink experts), everything else by the ops
loader with Inkling's `conversion_mapping` entry."""

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers.core_model_loading import WeightConverter, apply_renamings

from vllm.model_executor.model_loader.conversion_mapping import get_conversion_mapping
from vllm.model_executor.model_loader.ops_loader import load_with_ops

# `<...>.mlp.experts.<param>` (routed) / `<...>.mlp.shared_experts.<param>` (sinks)
_EXPERTS = re.compile(r"^(?P<mlp>.*\.mlp)\.(?P<rest>(?:shared_)?experts\..+)$")


def load_inkling_weights(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    conversions = get_conversion_mapping(model)
    renamings = [c for c in conversions if not isinstance(c, WeightConverter)]
    moes = {n: m for n, m in model.named_modules() if hasattr(m, "load_expert_weight")}
    loaded: set[str] = set()

    def the_rest() -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            match = _EXPERTS.match(apply_renamings(name, renamings))
            if match is None or match["mlp"] not in moes:
                yield name, weight
                continue
            # ponytail: a stacked expert tensor is read whole, one at a time
            weight = weight if isinstance(weight, torch.Tensor) else weight[...]
            rest = match["rest"].replace("routed_experts.", "")
            for param in moes[match["mlp"]].load_expert_weight(rest, weight):
                loaded.add(f"{match['mlp']}.{param}")

    loaded |= load_with_ops(model, the_rest(), conversions)
    for name, moe in moes.items():
        loaded.update(f"{name}.{param}" for param in moe.finalize_load())
    return loaded
