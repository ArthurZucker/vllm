# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling's checkpoint stream: its MoE experts loaded by
`InklingMoE.load_expert_weight` (rows interleaved gate/up, `input_amax` scales,
sink experts), everything else passed on to the ops loader (Inkling's
`conversion_mapping` entry)."""

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers.core_model_loading import WeightConverter, apply_renamings

from vllm.model_executor.model_loader.conversion_mapping import get_conversion_mapping

# `<...>.mlp.experts.<param>` (routed) / `<...>.mlp.shared_experts.<param>` (sinks)
_EXPERTS = re.compile(r"^(?P<mlp>.*\.mlp)\.(?P<rest>(?:shared_)?experts\..+)$")


def inkling_expert_weights(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """A `checkpoint_transforms` entry (`partial(inkling_expert_weights, model)`):
    the expert tensors loaded by their MoE (reported loaded), the rest yielded;
    once all are read, the MoEs' padding experts zeroed (`finalize_load`)"""
    renamings = [
        c for c in get_conversion_mapping(model) if not isinstance(c, WeightConverter)
    ]
    moes = {n: m for n, m in model.named_modules() if hasattr(m, "load_expert_weight")}
    params = dict(model.named_parameters())
    for name, weight in weights:
        match = _EXPERTS.match(apply_renamings(name, renamings))
        if match is None or match["mlp"] not in moes:
            yield name, weight
            continue
        # ponytail: a stacked expert tensor is read whole, one at a time
        weight = weight if isinstance(weight, torch.Tensor) else weight[...]
        rest = match["rest"].replace("routed_experts.", "")
        for param in moes[match["mlp"]].load_expert_weight(rest, weight):
            if (loaded := params.get(f"{match['mlp']}.{param}")) is not None:
                loaded.loaded_at_init = True  # written here, not by the ops
    for moe in moes.values():
        moe.finalize_load()
