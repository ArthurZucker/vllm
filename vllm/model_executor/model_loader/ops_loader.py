# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one weight loader: Transformers' `convert_and_load_state_dict_in_model`.

```
checkpoint key ─ rename_source_key(conversion_mapping) ─▶ target param
              ─ VllmShard(target).shard_tensor(source) ─▶ this rank's bytes, read
              ─ WeightConverter ops (Concatenate, MergeModulelist, PadToParam, ...)
              ─ copied into vLLM's param (`load_in_place`)
```

What a checkpoint becomes is written in `conversion_mapping.py`; nothing is inferred.
How a param is sharded is the param's, like Transformers' DTensor placements: vLLM
records it on the param (`output_dim` / `input_dim` in a column / row parallel
linear, `shard_dim`, `expert_dim` for the experts, `partial`). Each source is
sharded on read, before the ops: a fused `qkv_proj` concatenates this rank's q, k
and v slices; a lazy safetensors slice reads only them.
"""

import re
from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from transformers.core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightTransform,
    convert_and_load_state_dict_in_model,
)
from transformers.modeling_utils import LoadStateDictConfig

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.conversion_mapping import get_conversion_mapping
from vllm.model_executor.model_loader.weight_ops import PadToParam

logger = init_logger(__name__)


def _shape(source) -> list[int]:
    return (
        list(source.shape) if isinstance(source, torch.Tensor) else source.get_shape()
    )


def _read(source, dim: int | None = None, start: int = 0, length: int | None = None):
    """`source` (a tensor or a lazy safetensors slice), only `[start, start+length)`
    of `dim` read"""
    if dim is None:
        return source[...]
    index = [slice(None)] * len(_shape(source))
    index[dim] = slice(start, start + length)
    return source[tuple(index)]


def _placement(layer: nn.Module, param: nn.Parameter) -> tuple[int | None, int, int]:
    """(sharded dim, rank, ranks) of a param, as vLLM recorded it"""
    if isinstance(layer, ReplicatedLinear):  # every rank holds all of it
        return None, 0, 1
    if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
        row = isinstance(layer, RowParallelLinear)
        dim = getattr(param, "input_dim" if row else "output_dim", None)
        size = getattr(layer, "tp_size", 1)
        if isinstance(layer, VocabParallelEmbedding) and dim is not None:  # its range
            return (
                dim,
                layer.shard_indices.org_vocab_start_index // param.shape[dim],
                size,
            )
        return dim, getattr(layer, "tp_rank", 0), size
    if hasattr(layer, "moe_config"):  # the experts
        config = layer.moe_config.moe_parallel_config
        return getattr(param, "shard_dim", None), config.tp_rank, config.tp_size
    rank = get_tensor_model_parallel_rank()
    return (
        getattr(param, "shard_dim", None),
        rank,
        get_tensor_model_parallel_world_size(),
    )


class VllmShard:
    """Shard-on-read for one source of one target param, Transformers'
    `DtensorShardOperation` interface: `shard_tensor(source, tensor_idx)` is this
    rank's slice, `None` if the rank holds none of it (another rank's expert)."""

    def __init__(self, model, name: str, param, sources: int, index: int, expert: bool):
        self.param, self.sources, self.index, self.expert = (
            param,
            sources,
            index,
            expert,
        )
        self.layer = model.get_submodule(name.rpartition(".")[0])

    def shard_tensor(self, source, tensor_idx=None, device=None, dtype=None):
        layer, param, shape = self.layer, self.param, _shape(source)
        dim, rank, size = _placement(layer, param)
        target = list(param.shape)
        if len(shape) != param.dim() and dim is not None and dim < param.dim():
            # stored at another rank (an old 4D `A_log`, a 2D conv1d): viewed at the
            # param's, the full length on the sharded dim
            full = list(param.shape)
            full[dim] = 1
            full[dim] = torch.Size(shape).numel() // torch.Size(full).numel()
            source, shape = _read(source).reshape(full), full
        if getattr(param, "expert_dim", None) is not None:
            if self.expert:  # one checkpoint expert, a slice of the experts
                if layer._map_global_expert_id_to_local_expert_id(tensor_idx) == -1:
                    return None
                target, dim = target[1:], None if dim is None else dim - 1
            elif shape[0] > target[0]:  # all of them, on disk: this rank's
                ep = layer.moe_config.moe_parallel_config.ep_rank
                source = _read(source, 0, ep * target[0], target[0])
                shape = _shape(source)
        if getattr(param, "partial", False) and rank != 0:  # added once, after reduce
            return torch.zeros(shape, device=device, dtype=dtype or param.dtype)
        if dim is None or len(shape) <= dim or size == 1:
            # replicated: a param as large as one source (GPTQ's `g_idx` in a fused
            # layer) takes the first only
            one = self.sources > 1 and torch.Size(shape).numel() == param.numel()
            tensor = None if one and self.index > 0 else _read(source)
        else:
            tensor = self._slice(source, shape, dim, rank, size, target)
        if tensor is None:
            return None
        # on the device first: no CPU kernel in the loader's threads (OpenMP's pool is
        # invalid there, in a forked worker)
        tensor = tensor.to(device=device).to(dtype=dtype)
        for op in getattr(param, "load_ops", ()):  # e.g. A_log's `-exp`
            out = next(iter(op.convert({"t": [tensor]}, ["t"], ["t"]).values()))
            tensor = out[0] if isinstance(out, list) else out
        return tensor

    def _slice(self, source, shape, dim, rank, size, target):
        if segments := getattr(self.param, "shard_segments", None):
            return self._segments(source, segments, dim, rank, size)
        layer = self.layer
        shards = getattr(layer, "shard_ids", None) or []
        if self.sources == 1 and len(shards) > 1:  # fused on disk, e.g. `qkv_proj`
            return self._fused_on_disk(source, shards, dim, rank, size)
        replicas = 1
        if getattr(self.param, "shard_even", False):  # the source evenly, e.g. sinks
            local = shape[dim] // size
        elif self.sources == 1:  # the param's own range, e.g. a padded vocab
            local = target[dim]
        else:
            if isinstance(layer, QKVParallelLinear) and shards[self.index] in (
                "k",
                "v",
            ):
                replicas = layer.num_kv_head_replicas  # ranks sharing a kv head
            local = shape[dim] * replicas // size
        start = min(rank // replicas * local, shape[dim])
        return _read(source, dim, start, min(local, shape[dim] - start))

    def _segments(self, source, segments, dim, rank, size):
        """A param concatenating segments sharded on their own (Mamba2's [gate, x,
        B, C, dt]): `(full size, padded extra, replicated)` each. A replicated segment
        (one group, more ranks) is rank 0's slice on every rank; padding stays zeros."""
        pieces, offset = [], 0
        for full, extra, replicated in segments:
            local = full // size
            skip = 0 if replicated else rank * local
            take = max(min(local, full - extra - skip), 0)
            piece = _read(source, dim, offset + skip, take)
            if take < local:
                pad = list(piece.shape)
                pad[dim] = local - take
                piece = torch.cat([piece, piece.new_zeros(pad)], dim=dim)
            pieces.append(piece)
            offset += full - extra
        return torch.cat(pieces, dim=dim)

    def _fused_on_disk(self, source, shards, dim, rank, size):
        """Split (by the layer's shard sizes) -> Shard each -> Concatenate: this rank's
        slice of every segment of a tensor fused on disk, read alone"""
        layer, length = self.layer, _shape(source)[dim]
        qkv = isinstance(layer, QKVParallelLinear)
        replicas = [
            layer.num_kv_head_replicas if qkv and s in ("k", "v") else 1 for s in shards
        ]
        rows = [
            layer._get_shard_size_mapping(s) * size // r
            for s, r in zip(shards, replicas)
        ]
        pieces, offset = [], 0
        for full, r in zip(rows, replicas):
            segment = full * length // sum(rows)  # scales, packed: same proportions
            local = segment * r // size
            start = offset + min(rank // r * local, segment)
            pieces.append(
                _read(source, dim, start, min(local, offset + segment - start))
            )
            offset += segment
        return torch.cat(pieces, dim=dim)


class ShardOutputs(ConversionOps):
    """After a split (e.g. a `qkv` on disk into vLLM's q, k and v), each output
    sharded as its own param is"""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        layer_name = kwargs["full_layer_name"]  # the first target's full name
        output = {}
        for pattern, tensor in input_dict.items():
            tensor = tensor[0] if isinstance(tensor, list) else tensor
            name = layer_name.replace(target_patterns[0], pattern)
            try:
                param = self.model.get_parameter(name)
            except AttributeError:
                output[pattern] = tensor  # reported as unexpected
                continue
            sharder = VllmShard(self.model, name, param, 1, 0, False)
            output[pattern] = sharder.shard_tensor(tensor, dtype=tensor.dtype)
        return output


def _padded_converters(model: nn.Module) -> list[WeightConverter]:
    """A `PadToParam` converter for each param larger than its checkpoint tensor:
    vocabularies padded to a multiple of the ranks, `pad_value` params (sinks)"""
    return [
        WeightConverter(f"^{re.escape(name)}$", name, operations=[PadToParam()])
        for module_name, module in model.named_modules()
        for param_name, param in module.named_parameters(recurse=False)
        if isinstance(module, VocabParallelEmbedding) or hasattr(param, "pad_value")
        for name in [f"{module_name}.{param_name}" if module_name else param_name]
    ]


def load_weights(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Load `weights` into `model`. Returns the params loaded."""
    if (custom := getattr(model, "load_weights", None)) is not None:
        return custom(weights)  # the Transformers backend: its own conversions first
    return load_with_ops(model, weights)


def load_with_ops(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    conversions: list[WeightTransform] | None = None,
    hf_quantizer=None,
) -> set[str]:
    """Load `weights` (tensors, or lazy safetensors slices: then each rank reads its
    bytes only) into `model` with `conversions` (default: the model's entries in
    `conversion_mapping`). Returns the params loaded."""
    if conversions is None:
        conversions = get_conversion_mapping(model)
    conversions = [*deepcopy(conversions), *_padded_converters(model)]
    # the live params: `state_dict()` (what Transformers passes) drops their attributes
    params = dict(model.named_parameters()) | dict(model.named_buffers())

    def get_sharding_op(target, _detached, converter, source_pattern):
        param = params.get(target)
        param = model.get_parameter(target) if param is None else param  # tied
        if isinstance(converter, WeightConverter):
            full = any(getattr(op, "needs_full", False) for op in converter.operations)
            if len(converter.target_patterns) > 1 or full:  # each output sharded after
                if not any(isinstance(op, ShardOutputs) for op in converter.operations):
                    converter.operations = [*converter.operations, ShardOutputs(model)]
                return None
            sources = converter.source_patterns
            index, expert = sources.index(source_pattern), "*" in source_pattern
        else:
            sources, index, expert = [source_pattern], 0, False
        return VllmShard(model, target, param, len(sources), index, expert)

    if not hasattr(model, "base_model_prefix"):
        # The module holding the model's body (e.g. `transformer`): Transformers adds or
        # strips it from checkpoint keys saved with or without it (Bloom, GPT-2, ...)
        bodies = [
            name
            for name, child in model.named_children()
            if "head" not in name and next(child.parameters(), None) is not None
        ]
        model.base_model_prefix = bodies[0] if len(bodies) == 1 else None
    device = next(model.parameters()).device
    config = LoadStateDictConfig(
        weight_mapping=conversions,
        device_map={"": device},
        hf_quantizer=hf_quantizer,
        get_sharding_op=get_sharding_op,
        load_in_place=True,
    )
    info, _ = convert_and_load_state_dict_in_model(model, dict(weights), config)
    unexpected = sorted(k for k in info.unexpected_keys if "_dropped_" not in k)
    if unexpected or info.mismatched_keys or info.conversion_errors:
        logger.warning(
            "Checkpoint tensors matching no param (add them to conversion_mapping.py): "
            "%s; mismatched: %s; conversion errors: %s",
            unexpected[:8],
            sorted(info.mismatched_keys)[:4],
            list(info.conversion_errors.items())[:4],
        )
    # a param the checkpoint lacks, tied to another (an MLM decoder to the embeddings)
    for name, param in params.items():
        if name in info.missing_keys and (source := getattr(param, "tied_to", None)):
            param.data.copy_(params[source].data[: param.shape[0]])
            info.missing_keys.discard(name)
    # params loaded as they were built (e.g. a pooler's Sentence-Transformers projector)
    at_init = {
        n for n, p in model.named_parameters() if getattr(p, "loaded_at_init", 0)
    }
    return {name for name, _ in model.named_parameters()} - info.missing_keys | at_init
