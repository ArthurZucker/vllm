# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run Transformers' checkpoint `WeightConverter`s on vLLM's stream of weights."""

from collections.abc import Iterable, Iterator
from copy import deepcopy

import torch
from transformers.core_model_loading import (
    WeightConverter,
    WeightTransform,
    rename_source_key,
)


def streamable_converters(conversions: list[WeightTransform]) -> list[WeightConverter]:
    """The converters to run on the stream: those taking one key per source pattern.

    Wildcard sources (`experts.*.gate_proj`) are expert merges, left out: `FusedMoE`'s
    `weight_loader` shards each checkpoint expert straight into place, without
    stacking all of them first.
    """
    return [
        conversion
        for conversion in conversions
        if isinstance(conversion, WeightConverter)
        and not any("*" in pattern for pattern in conversion.source_patterns)
    ]


def convert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    converters: list[WeightConverter],
    config=None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """`weights`, with the keys a converter matches replaced by its outputs.

    A converter's tensors are held until it has one per source pattern (e.g. `gate`
    and `up` for a `gate_up` concatenation), then converted. Keys no converter
    matches pass through. Quantization scales follow their weight: a `.weight`
    pattern also matches `.weight_scale`, into a target of its own.
    """
    # ponytail: matched on checkpoint names, renamings stay in `WeightsMapper`. A
    # converter whose pattern only exists after a renaming is missed
    pending: dict[str, WeightConverter] = {}
    for name, tensor in weights:
        target, pattern = rename_source_key(name, [], converters)
        if pattern is None:
            yield name, tensor
            continue
        if target not in pending:
            converter = next(c for c in converters if pattern in c.source_patterns)
            pending[target] = deepcopy(converter)
        converter = pending[target]
        converter.add_tensor(target, name, pattern, tensor)
        if len(converter.collected_tensors) == len(converter.source_patterns):
            yield from pending.pop(target).convert(target, config=config).items()
    # A group missing a source, e.g. a scale only some of its weights have
    for target, converter in pending.items():
        yield from converter.convert(target, config=config).items()
