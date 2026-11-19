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

import inspect
import re
from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from transformers.core_model_loading import (
    ConversionOps,
    Transpose,
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
from vllm.model_executor.model_loader.conversion_mapping import (
    from_mapper,
    get_conversion_mapping,
)
from vllm.model_executor.model_loader.weight_ops import (
    Deinterleave,
    PadToParam,
    SwapHalves,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

# ops reordering a whole tensor: the loader shards their output, not their input
NEEDS_FULL = (Transpose, SwapHalves, Deinterleave)

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


def _replicas(layer: nn.Module, shard) -> int:
    """How many ranks share a shard's rows: a kv head over more ranks than kv heads
    (`QKVParallelLinear`, or a layer's `shard_replicas`, e.g. Inkling's `qkvr`)"""
    if (replicas := getattr(layer, "shard_replicas", None)) is not None:
        return replicas.get(shard, 1)
    if isinstance(layer, QKVParallelLinear) and shard in ("k", "v"):
        return layer.num_kv_head_replicas
    return 1


class VllmShard:
    """Shard-on-read for one source of one target param, Transformers'
    `DtensorShardOperation` interface: `shard_tensor(source, tensor_idx)` is this
    rank's slice, `None` if the rank holds none of it (another rank's expert)."""

    def __init__(
        self, model, name, param, sources: int, index: int, expert: bool, shards=None
    ):
        self.param, self.sources, self.index, self.expert = (
            param,
            sources,
            index,
            expert,
        )
        self.covers = shards  # shards each source fills (`conversion_mapping.Fused`)
        self.layer = model.get_submodule(name.rpartition(".")[0])

    def shard_tensor(self, source, tensor_idx=None, device=None, dtype=None):
        layer, param, shape = self.layer, self.param, _shape(source)
        dim, rank, size = _placement(layer, param)
        target = list(param.shape)
        if getattr(param, "expert_dim", None) is not None:
            if self.expert:  # one checkpoint expert, a slice of the experts
                if layer._map_global_expert_id_to_local_expert_id(tensor_idx) == -1:
                    return None
                target, dim = target[1:], None if dim is None else dim - 1
            elif shape[0] > target[0] and len(shape) == len(target):
                # all of them, on disk: this rank's
                ep = layer.moe_config.moe_parallel_config.ep_rank
                source = _read(source, 0, ep * target[0], target[0])
                shape = _shape(source)
        if len(shape) != len(target) and dim is not None and dim < len(target):
            # stored at another rank (an old 4D `A_log`, a 2D conv1d): viewed at the
            # param's, the full length on the sharded dim
            full = list(target)
            full[dim] = 1
            full[dim] = torch.Size(shape).numel() // torch.Size(full).numel()
            source, shape = _read(source).reshape(full), full
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
        if self.covers and self.covers[self.index] > 1:  # a source filling several
            first = sum(self.covers[: self.index])
            covered = shards[first : first + self.covers[self.index]]
            return self._fused_on_disk(source, covered, dim, rank, size)
        if self.covers:  # each source's first shard
            shards = [shards[sum(self.covers[:i])] for i in range(len(self.covers))]
        replicas = 1
        if getattr(self.param, "shard_even", False):  # the source evenly, e.g. sinks
            local = shape[dim] // size
        elif self.sources == 1:  # the param's own range, e.g. a padded vocab
            local = target[dim]
            replicas = getattr(self.param, "shard_replicas", 1)  # e.g. a kv norm
        else:
            shard = shards[self.index] if self.index < len(shards) else None
            replicas = _replicas(layer, shard)
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
        replicas = [_replicas(layer, shard) for shard in shards]
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


def _load_into(param: torch.Tensor, value: torch.Tensor) -> None:
    """`value` (this rank's, converted) into `param`. A same-size value is viewed at
    the param's shape (a scalar scale into `[1]`); a layer's output padding (rows
    after `weight_loader_numel`) stays zero. Then the param's loader when it takes
    `(param, value)` (a layer sharding for itself, online quantization, layerwise
    reload), else a copy: a shape mismatch raises."""
    if value.shape != param.shape and value.numel() == param.numel():
        value = value.reshape(param.shape)
    elif getattr(param, "weight_loader_numel", None) == value.numel():
        param = param[: value.shape[0]]  # ponytail: padding is trailing rows only
    loader = getattr(param, "weight_loader", None)
    try:
        inspect.signature(loader).bind(param, value)
    except (TypeError, ValueError):
        loader = default_weight_loader
    loader(param, value)


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
    # functions of the whole stream, run first: a model's (Jina v5's task LoRA merged,
    # MiniMax M3's ROCm redirect), then the quant config's (pair scales, repack)
    stream = list(getattr(model, "checkpoint_transforms", []))
    if conversions is None:
        conversions = get_conversion_mapping(model)
    # the quant config's: KV-cache scales onto `attn`, ModelOpt's CSF pairing, ...
    modules = (model, *model.children())
    quant = next(
        (m.quant_config for m in modules if getattr(m, "quant_config", None)), None
    )
    if quant is not None:
        mapper = quant.get_cache_scale_mapper() | quant.get_checkpoint_weight_mapper()
        conversions = [*conversions, *from_mapper(mapper)]
        stream += mapper.weights_transforms
    for transform in stream:
        weights = transform(weights)
    conversions = [*deepcopy(conversions), *_padded_converters(model)]
    # the live params: `state_dict()` (what Transformers passes) drops their attributes
    params = dict(model.named_parameters()) | dict(model.named_buffers())

    def get_sharding_op(target, _detached, converter, source_pattern):
        param = params.get(target)
        param = model.get_parameter(target) if param is None else param  # tied
        if isinstance(converter, WeightConverter):
            full = any(isinstance(op, NEEDS_FULL) for op in converter.operations)
            if len(converter.target_patterns) > 1 or full:  # each output sharded after
                if not any(isinstance(op, ShardOutputs) for op in converter.operations):
                    converter.operations = [*converter.operations, ShardOutputs(model)]
                return None
            sources = converter.source_patterns
            index, expert = sources.index(source_pattern), "*" in source_pattern
        else:
            sources, index, expert = [source_pattern], 0, False
        shards = getattr(converter, "shards", None)
        return VllmShard(model, target, param, len(sources), index, expert, shards)

    if not hasattr(model, "base_model_prefix"):
        # The module holding the model's body (e.g. `transformer`, else `model`):
        # Transformers adds or strips it from checkpoint keys saved with or without it
        bodies = [
            name
            for name, child in model.named_children()
            if "head" not in name and next(child.parameters(), None) is not None
        ]
        model.base_model_prefix = bodies[0] if len(bodies) == 1 else "model"
    # online quantization builds its layers on `meta`, filled when complete
    device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device(current_platform.device_type),  # this rank's current device
    )
    config = LoadStateDictConfig(
        weight_mapping=conversions,
        device_map={"": device},
        hf_quantizer=hf_quantizer,
        get_sharding_op=get_sharding_op,
        load_in_place=_load_into,
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
