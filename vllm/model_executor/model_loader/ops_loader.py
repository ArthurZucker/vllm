# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one weight loader: Transformers' `convert_and_load_state_dict_in_model`.

```
checkpoint key ─ rename_source_key(conversion_mapping) ─▶ target param
              ─ VllmShard(target).shard_tensor(source) ─▶ this rank's bytes, read
              ─ WeightConverter ops (Concatenate, MergeModulelist, ...)
              ─ copied into vLLM's param (`load_in_place`)
```

What a model's checkpoint becomes is written in `conversion_mapping.py`; nothing is
inferred. How a param is sharded is the param's, like Transformers' DTensor
placements: read from what vLLM records on it (`output_dim` / `input_dim` in a
column / row parallel linear, the experts' layer, `shard_dim`). Each source is
sharded on read, before the ops: a fused `qkv_proj` concatenates this rank's q, k
and v slices, a lazy safetensors slice reads only them.
"""

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from transformers.core_model_loading import (
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
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.conversion_mapping import get_conversion_mapping

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


class VllmShard:
    """Shard-on-read for one source of one target param, the interface of
    Transformers' `DtensorShardOperation`: `shard_tensor(source, tensor_idx)` is this
    rank's slice, `None` if the rank holds none of it (another rank's expert)."""

    def __init__(self, model, name: str, param, sources: int, index: int, expert: bool):
        self.param, self.sources, self.index, self.expert = (
            param,
            sources,
            index,
            expert,
        )
        self.layer = model.get_submodule(name.rpartition(".")[0])
        self.w2 = name.rpartition(".")[2].startswith("w2_")

    def shard_tensor(self, source, tensor_idx=None, device=None, dtype=None):
        if hasattr(self.layer, "_map_global_expert_id_to_local_expert_id"):
            tensor = self._experts(source, tensor_idx if self.expert else None)
        else:
            tensor = self._dense(source)
        if tensor is None:
            return None
        for op in getattr(self.param, "load_ops", ()):  # e.g. A_log's `-exp`
            out = next(iter(op.convert({"t": [tensor]}, ["t"], ["t"]).values()))
            tensor = out[0] if isinstance(out, list) else out
        return tensor.to(device=device, dtype=dtype)

    def _dense(self, source):
        layer, param, shape = self.layer, self.param, _shape(source)
        if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            row = isinstance(layer, RowParallelLinear)
            dim = getattr(param, "input_dim" if row else "output_dim", None)
            size = getattr(layer, "tp_size", 1)
        else:
            dim = getattr(param, "shard_dim", None)
            size = get_tensor_model_parallel_world_size()
        if dim is None or len(shape) <= dim or size == 1:
            # replicated: a param as large as one source (GPTQ's `g_idx` in a fused
            # layer) takes the first only
            one = self.sources > 1 and torch.Size(shape).numel() == param.numel()
            return None if one and self.index > 0 else self._pad(_read(source), dim)
        if isinstance(layer, VocabParallelEmbedding):  # keeps its range, not its rank
            rank = layer.shard_indices.org_vocab_start_index // param.shape[dim]
        else:
            rank = getattr(layer, "tp_rank", None)
            rank = get_tensor_model_parallel_rank() if rank is None else rank
        replicas = 1
        if getattr(param, "shard_even", False):  # the source split evenly, e.g. sinks
            local = shape[dim] // size
        elif self.sources == 1:  # the param's own size, e.g. a padded vocab
            local = param.shape[dim]
        else:
            shard = layer.shard_ids[self.index] if hasattr(layer, "shard_ids") else None
            if isinstance(layer, QKVParallelLinear) and shard in ("k", "v"):
                replicas = layer.num_kv_head_replicas  # ranks sharing a kv head
            local = shape[dim] * replicas // size
        start = min(rank // replicas * local, shape[dim])
        tensor = _read(source, dim, start, min(local, shape[dim] - start))
        return self._pad(tensor, dim) if self.sources == 1 else tensor

    def _pad(self, tensor: torch.Tensor, dim: int | None) -> torch.Tensor:
        """A padded param (vocab, sink heads) keeps its `pad_value` past the source"""
        param = self.param
        if tensor.dim() == 0 and param.dim() == 1 and self.sources == 1:
            return tensor.reshape(1)  # a scalar scale into a `[1]` param
        if (
            self.sources > 1
            or tensor.dim() != param.dim()
            or tensor.shape == param.shape
        ):
            return tensor
        padded = torch.full(
            param.shape, getattr(param, "pad_value", 0), dtype=tensor.dtype
        )
        padded[tuple(slice(0, n) for n in tensor.shape)] = tensor
        return padded

    def _experts(self, source, expert: int | None):
        """One checkpoint expert (`expert`), or all of them fused on disk"""
        layer, param, shape = self.layer, self.param, _shape(source)
        config = layer.moe_config.moe_parallel_config
        target = list(param.shape)
        tensor = source
        if expert is not None:
            if layer._map_global_expert_id_to_local_expert_id(expert) == -1:
                return None
            target = target[1:]
        elif shape[0] > target[0]:  # fused on disk: this rank's experts
            tensor = _read(source, 0, config.ep_rank * target[0], target[0])
            shape = list(tensor.shape)
        # the intermediate dim: gate and up (`w13`) each take half the param's
        dim = (1 if self.w2 else 0) + (expert is None)
        if getattr(param, "is_transposed", False):
            dim = (1 - dim) if expert is not None else (3 - dim)
        if dim < len(shape) and dim < len(target):
            local = target[dim] // 2 if self.sources == 2 else target[dim]
            if shape[dim] > local:
                tensor = _read(tensor, dim, config.tp_rank * local, local)
        tensor = tensor if isinstance(tensor, torch.Tensor) else _read(tensor)
        if self.w2 and param.dim() <= 2 and config.tp_rank != 0:  # w2 bias
            tensor = torch.zeros_like(tensor)  # added once, after the reduce
        return tensor


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
    bytes only) into `model` with `conversions` (default: the model's entry in
    `conversion_mapping`). Returns the params loaded."""
    if conversions is None:
        conversions = get_conversion_mapping(model)

    def get_sharding_op(target, param, converter, source_pattern):
        if isinstance(converter, WeightConverter):
            if len(converter.target_patterns) > 1:
                return None  # split on disk (e.g. qkv): TODO shard after the split
            sources = converter.source_patterns
            index, expert = sources.index(source_pattern), "*" in source_pattern
        else:
            sources, index, expert = [source_pattern], 0, False
        return VllmShard(model, target, param, len(sources), index, expert)

    device = next(model.parameters()).device
    config = LoadStateDictConfig(
        weight_mapping=deepcopy(conversions),
        device_map={"": device},
        hf_quantizer=hf_quantizer,
        get_sharding_op=get_sharding_op,
        load_in_place=True,
    )
    info, _ = convert_and_load_state_dict_in_model(model, dict(weights), config)
    if info.unexpected_keys or info.mismatched_keys or info.conversion_errors:
        logger.warning(
            "Checkpoint tensors matching no param (add them to conversion_mapping.py): "
            "%s; mismatched: %s; conversion errors: %s",
            sorted(info.unexpected_keys)[:8],
            sorted(info.mismatched_keys)[:4],
            list(info.conversion_errors.items())[:4],
        )
    return {name for name, _ in model.named_parameters()} - info.missing_keys
