# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load weights with Transformers' ops instead of each layer's `weight_loader`.

One rule covers every linear, fused or not, quantized or not:

```
q_proj.weight ─ Shard(output_dim, rank)    ─┐
k_proj.weight ─ Shard(output_dim, rank//r) ─┼─ Concatenate(output_dim) ─▶ qkv_proj
v_proj.weight ─ Shard(output_dim, rank//r) ─┘   (0-dim scales: stacked)
```

The dim is the one vLLM records on the param (`output_dim` on column parallel
layers, `input_dim` on row parallel ones, none: replicated). Each source's local size
is its full size over the ranks it is split across, `r` being the ranks sharing a
GQA kv head. Packed (int4 in int32) and block-scale params need nothing more: their
checkpoint tensors are packed or blocked the same way.
"""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from transformers.core_model_loading import Concatenate, ConversionOps

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# Exact types: a subclass may lay its params out differently (e.g. MiniMax-M3's
# indexer in its qkv), it keeps its own `weight_loader`
LAYERS = (
    ReplicatedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    ParallelLMHead,
)


class Shard(ConversionOps):
    """This rank's slice of each source along `dim`, `replicas[i]` ranks sharing
    the slice of source `i` (GQA's kv heads, when there are fewer than ranks).

    `local_size`: the size of a single source's slice, the param's (e.g. a vocab
    padded to a multiple of the ranks); its last rank's slice may be shorter."""

    def __init__(self, dim, rank, world_size, replicas, local_size=None):
        self.dim, self.rank, self.world_size = dim, rank, world_size
        self.replicas, self.local_size = replicas, local_size

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        output = {}
        for pattern, replicas in zip(source_patterns, self.replicas):
            tensor = input_dict[pattern][0]
            if self.dim is None or tensor.dim() <= self.dim:
                output[pattern] = [tensor]
                continue
            size = tensor.shape[self.dim]
            local = self.local_size or size * replicas // self.world_size
            start = min(self.rank // replicas * local, size)
            output[pattern] = [tensor.narrow(self.dim, start, min(local, size - start))]
        return output


def _shards(layer: nn.Module) -> list | None:
    """The shard ids of a fused layer, in their order in its params"""
    if type(layer) is QKVParallelLinear:
        return ["q", "k", "v"]
    if type(layer) is MergedColumnParallelLinear:
        return list(range(len(layer.output_sizes)))
    return None


def _rank(layer: nn.Module, param: nn.Parameter) -> int:
    if isinstance(layer, VocabParallelEmbedding):  # keeps its range, not its rank
        return layer.shard_indices.org_vocab_start_index // param.shape[0]
    return layer.tp_rank


def _write(layer: nn.Module, param: nn.Parameter, tensors: dict) -> None:
    """Transformers' ops: `tensors` (shard id -> checkpoint tensor) into `param`"""
    row = type(layer) is RowParallelLinear
    dim = getattr(param, "input_dim" if row else "output_dim", None)
    dim = dim if type(layer) in LAYERS else None  # e.g. a norm: replicated
    shards = _shards(layer) or [None]
    sources = [str(shard) for shard in shards]
    if dim is None and tensors[shards[0]].numel() == param.numel():
        sources = sources[:1]  # replicated in every shard, e.g. GPTQ's `g_idx`
    kv = getattr(layer, "num_kv_head_replicas", 1)
    replicas = [kv if shard in ("k", "v") else 1 for shard in shards]
    tp_size = getattr(layer, "tp_size", 1)
    ops = [
        Shard(
            dim,
            _rank(layer, param) if tp_size > 1 else 0,
            tp_size,
            replicas,
            local_size=param.shape[dim]
            if len(shards) == 1 and dim is not None
            else None,
        ),
        Concatenate(0 if dim is None else dim),
    ]
    collected = {str(shard): [tensor] for shard, tensor in tensors.items()}
    for op in ops:
        collected = op.convert(collected, sources, ["param"])
    value = collected["param"]
    if dim is not None and value.shape[dim] < param.shape[dim]:  # padded: vocab
        param.data.narrow(
            dim, value.shape[dim], param.shape[dim] - value.shape[dim]
        ).zero_()
        param.data.narrow(dim, 0, value.shape[dim]).copy_(value)
    else:
        param.data.copy_(value.reshape(param.shape))


def load_with_ops(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]], loaded: set[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load `weights` (vLLM names, `shard_id` on fused ones) with Transformers' ops.

    Adds what it loads to `loaded`, yields what it doesn't (a layer it does not know
    keeps its `weight_loader`).
    """
    params = dict(model.named_parameters())
    pending: dict[str, dict] = {}
    for name, tensor in weights:
        param = params.get(name)
        layer = (
            model.get_submodule(name.rpartition(".")[0]) if param is not None else None
        )
        shard = getattr(tensor, "shard_id", None)
        known = (
            type(layer) in LAYERS
            or getattr(param, "weight_loader", default_weight_loader)
            is default_weight_loader
        )
        if (
            param is None
            or not known
            or getattr(param, "marlin_tile_size", None)
            or (shard is None) != (_shards(layer) is None)  # e.g. qkv fused on disk
        ):
            yield name, tensor
            continue
        tensors = pending.setdefault(name, {})
        tensors[shard] = tensor
        if len(tensors) == len(_shards(layer) or [None]):
            _write(layer, param, pending.pop(name))
            loaded.add(name)
    # A fused param missing a shard: its `weight_loader` takes what came
    for name, tensors in pending.items():
        yield from ((name, tensor) for tensor in tensors.values())
