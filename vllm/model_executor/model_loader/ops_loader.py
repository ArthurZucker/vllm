# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one weight loader: Transformers' conversion ops, for every vLLM model.

No layer or model loads its own weights. A checkpoint key finds its param by name
(`hf_to_vllm_mapper`, `packed_modules_mapping`, the experts' checkpoint names), then
the param's ops fill it:

```
linear    q_proj ─ Shard(dim, rank)    ─┐
          k_proj ─ Shard(dim, rank//r) ─┼─ Concatenate(dim) ─▶ qkv_proj  (0-dim: stack)
          v_proj ─ Shard(dim, rank//r) ─┘
          qkv on disk ─ Split(sizes) ─▶ the same, per shard
experts   experts.{e}.gate_proj ─ Shard ─┐
          experts.{e}.up_proj   ─ Shard ─┴─ Concatenate ─▶ w13[e]  (MergeModulelist
          experts.{e}.down_proj ─ Shard ───────────────▶ w2[e]    in place)
fused     experts.gate_up_proj ─ Shard(each dim larger than the param) ─▶ w13
other     the param's own `weight_loader` if it keeps one, else copied as is
```

`dim` is the one vLLM records on the param: `output_dim` in a column parallel layer,
`input_dim` in a row parallel one, none: replicated. A source's local size is its
full size over the ranks splitting it (`r` ranks share a GQA kv head), so packed
(int4 in int32), block-scale and padded params need nothing more.
"""

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers.core_model_loading import Concatenate, ConversionOps, WeightConverter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
    StackedColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)

EXPERT_KEY = re.compile(r"(.*\.experts)\.(\d+)\.(\w+)\.(.+)")
# Experts fused on disk, e.g. gpt-oss: `experts.gate_up_proj_blocks`
FUSED_EXPERTS_KEY = re.compile(
    r"(.*\.experts)\.(gate_up_proj|down_proj)(?:_(blocks|scales|bias)|\.(weight|bias))?"
)
FUSED_SUFFIX = {None: "weight", "blocks": "weight", "scales": "weight_scale"}
W1, W3, W2 = ("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")


class Shard(ConversionOps):
    """This rank's slice of each source along `dim`, `replicas[i]` ranks sharing
    source `i`'s slice. `local_size` (a single source): the param's own, e.g. a
    vocab padded to a multiple of the ranks; the last rank's slice may be shorter."""

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


class Split(ConversionOps):
    """A tensor fused on disk (e.g. `qkv`) into its shards, `fractions` of `dim`"""

    def __init__(self, dim, fractions):
        self.dim, self.fractions = dim, fractions

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = next(iter(input_dict.values()))[0]
        if self.dim is None or tensor.dim() <= self.dim:  # replicated, e.g. `g_idx`
            return {target: [tensor] for target in target_patterns}
        sizes = [round(f * tensor.shape[self.dim]) for f in self.fractions]
        chunks = tensor.split(sizes, dim=self.dim)
        return {target: [chunk] for target, chunk in zip(target_patterns, chunks)}


def _run(ops, tensors: dict, sources: list[str], targets: list[str]):
    for op in ops:
        tensors = op.convert(tensors, sources, targets)
    return tensors


def _param_ops(param: nn.Parameter, tensor: torch.Tensor) -> torch.Tensor:
    """The param's own `load_ops` (Transformers `ConversionOps`, e.g. A_log's
    `-exp`), on the checkpoint tensor before it is sharded and placed"""
    for op in getattr(param, "load_ops", ()):
        out = next(iter(op.convert({"t": [tensor]}, ["t"], ["t"]).values()))
        tensor = out[0] if isinstance(out, list) else out
    return tensor


def _load_other(param: nn.Parameter, tensor: torch.Tensor) -> None:
    """A param outside linears and experts: replicated, or sharded along its
    `shard_dim`, in `shard_segments` (full sizes of the segments concatenated along
    it, e.g. mamba's conv1d `[x, B, C]`) each sharded on its own"""
    dim = getattr(param, "shard_dim", None)
    if dim is None or tensor.dim() <= dim:
        return _copy(param, tensor, None)
    rank = getattr(param, "tp_rank", None)
    rank = get_tensor_model_parallel_rank() if rank is None else rank
    size = getattr(param, "tp_size", None) or get_tensor_model_parallel_world_size()
    segments = getattr(param, "shard_segments", None) or [tensor.shape[dim]]
    names = [str(i) for i in range(len(segments))]
    fractions = [segment / sum(segments) for segment in segments]
    ops = [Split(dim, fractions), Shard(dim, rank, size, [1] * len(names))]
    if len(segments) == 1:  # the param's own size, e.g. padded
        ops = [Shard(dim, rank, size, [1], param.shape[dim])]
        names = ["0"]
    out = _run(ops, {"0": [tensor]}, names, names)
    value = _run([Concatenate(dim)], out, names, ["param"])["param"]
    _copy(param, value, dim)


def _copy(param: nn.Parameter, value: torch.Tensor, dim: int | None) -> None:
    """`value` into `param`, zero-padded along `dim` (e.g. a padded vocab)"""
    data = param.data
    if (
        dim is not None
        and value.dim() == data.dim()
        and value.shape[dim] < data.shape[dim]
    ):
        data.narrow(dim, value.shape[dim], data.shape[dim] - value.shape[dim]).zero_()
        data = data.narrow(dim, 0, value.shape[dim])
    data.copy_(value.reshape(data.shape))


# Linears and embeddings


def _shard_ids(layer: nn.Module) -> list:
    return list(getattr(layer, "shard_ids", None) or [None])


def _replicas(layer: nn.Module, shard) -> int:
    kv = isinstance(layer, QKVParallelLinear) and shard in ("k", "v")
    return layer.num_kv_head_replicas if kv else 1


def _tp(layer: nn.Module, param: nn.Parameter, dim: int | None) -> tuple[int, int]:
    size = getattr(layer, "tp_size", 1)
    if size == 1 or dim is None:
        return 0, 1
    if isinstance(layer, VocabParallelEmbedding):  # keeps its range, not its rank
        return layer.shard_indices.org_vocab_start_index // param.shape[dim], size
    return layer.tp_rank, size


def _load_linear(layer: nn.Module, param: nn.Parameter, tensors: dict) -> None:
    """`tensors` (shard id -> checkpoint tensor, `None`: all shards) into `param`"""
    row = isinstance(layer, RowParallelLinear)
    dim = getattr(param, "input_dim" if row else "output_dim", None)
    shards = _shard_ids(layer)
    names = [str(shard) for shard in shards]
    collected = {str(shard): [tensor] for shard, tensor in tensors.items()}
    if None in tensors and shards != [None]:  # fused on disk
        rank, tp = _tp(layer, param, dim)
        sizes = [
            layer._get_shard_size_mapping(s) * tp / _replicas(layer, s) for s in shards
        ]
        fractions = [size / sum(sizes) for size in sizes]
        collected = _run([Split(dim, fractions)], collected, ["None"], names)
    first = collected[names[0]][0]
    if dim is None and first.numel() == param.numel():
        names = names[:1]  # replicated in every shard, e.g. GPTQ's `g_idx`
    rank, tp = _tp(layer, param, dim)
    single = len(names) == 1 and dim is not None and first.dim() > dim
    shard = Shard(
        dim,
        rank,
        tp,
        [_replicas(layer, s) for s in shards[: len(names)]],
        local_size=param.shape[dim] if single else None,
    )
    value = _run([shard, Concatenate(dim or 0)], collected, names, ["param"])
    _copy(param, value["param"], dim)


# Experts


def _expert_shard(layer: nn.Module, proj: str) -> str | None:
    names = {
        getattr(layer, "ckpt_gate_proj_name", None): "w1",
        getattr(layer, "ckpt_up_proj_name", None): "w3",
        getattr(layer, "ckpt_down_proj_name", None): "w2",
    }
    return names.get(proj) or next(
        (
            shard
            for shard, alias in zip(("w1", "w3", "w2"), (W1, W3, W2))
            if proj in alias
        ),
        None,
    )


def _load_expert(layer, param, tensor, expert: int, shard: str, name: str) -> None:
    """One checkpoint expert's tensor into its slice of the fused experts' param"""
    local = layer._map_global_expert_id_to_local_expert_id(expert)
    if local == -1:
        return
    data = param.data[local]
    config = layer.moe_config
    rank, tp = config.tp_rank, config.moe_parallel_config.tp_size
    if data.dim() == 0 or tensor.numel() == 1:  # a per-tensor scale
        if data.numel() == 2:  # one per gate and up: [w1, w3]
            data[0 if shard == "w1" else 1] = tensor.reshape(())
        else:
            data.copy_(tensor.reshape(data.shape))
        return
    if shard == "w2" and name.endswith("bias"):  # added once, after the reduce
        data.copy_(tensor if rank == 0 else torch.zeros_like(tensor))
        return
    dim = 0 if shard != "w2" else 1
    if getattr(param, "is_transposed", False):
        dim = 1 - dim
    dim = min(dim, data.dim() - 1)
    gated = shard != "w2" and config.is_act_and_mul
    local_size = data.shape[dim] // 2 if gated else data.shape[dim]
    if tensor.shape[dim] > local_size:
        tensor = _run(
            [Shard(dim, rank, tp, [1], local_size)], {"t": [tensor]}, ["t"], []
        )["t"][0]
    if gated:  # Concatenate([gate, up]), each into its half
        data = data.narrow(dim, 0 if shard == "w1" else local_size, local_size)
    data = data.narrow(dim, 0, tensor.shape[dim])
    data.copy_(tensor)


def _load_fused_experts(layer, param, tensor, name: str) -> None:
    """All experts of a checkpoint tensor: every dim it is larger than the param in
    is sharded, the experts (dim 0) over the EP ranks, the others over the TP ranks
    (gpt-oss' interleaved gate/up: a contiguous slice keeps both)"""
    config = layer.moe_config.moe_parallel_config
    if name.endswith("w2_bias") and config.tp_rank != 0:  # added once, after reduce
        tensor = torch.zeros_like(tensor)
    for dim in range(tensor.dim()):
        local = param.shape[dim]
        if tensor.shape[dim] > local:
            rank, size = (
                (config.ep_rank, config.ep_size)
                if dim == 0
                else (config.tp_rank, config.tp_size)
            )
            shard = Shard(dim, rank, size, [1], local)
            tensor = _run([shard], {"t": [tensor]}, ["t"], [])["t"][0]
    data = param.data
    for dim in range(tensor.dim()):  # padded params, e.g. a rounded up hidden size
        data = data.narrow(dim, 0, tensor.shape[dim])
    data.copy_(tensor)


def _resolve_expert(name: str, params: dict, model: nn.Module):
    if (fused := FUSED_EXPERTS_KEY.fullmatch(name)) is not None:
        prefix, proj, suffix, plain = fused.groups()
        suffix = plain or ("bias" if suffix == "bias" else FUSED_SUFFIX[suffix])
        fused_name = "w13_" if proj == "gate_up_proj" else "w2_"
        for owner in ("", "routed_experts.", "base_layer."):
            if (param_name := f"{prefix}.{owner}{fused_name}{suffix}") in params:
                return param_name, "fused", -1
        return None
    match = EXPERT_KEY.fullmatch(name)
    if match is None:
        return None
    prefix, expert, proj, suffix = match.groups()
    for owner in ("", "routed_experts.", "base_layer.", "routed_experts.base_layer."):
        for fused in ("w13_", "w2_"):
            param_name = f"{prefix}.{owner}{fused}{suffix}"
            if param_name not in params:
                continue
            layer = model.get_submodule(param_name.rpartition(".")[0])
            shard = _expert_shard(layer, proj)
            if shard is not None and (fused == "w2_") == (shard == "w2"):
                return param_name, shard, int(expert)
    return None


# The loader


def _packed(model: nn.Module) -> dict[str, tuple[str, int]]:
    """`q_proj` -> (`qkv_proj`, 0) from every `packed_modules_mapping`"""
    reverse = {}
    for module in model.modules():
        for fused, sources in (
            getattr(module, "packed_modules_mapping", None) or {}
        ).items():
            for i, source in enumerate(sources):
                reverse.setdefault(source, (fused, i))
    return reverse


def _resolve(name: str, params: dict, packed: dict, model: nn.Module):
    """(param name, shard id, expert id) of a checkpoint key, `None`: no param"""
    if name in params:
        return name, None, None
    if (expert := _resolve_expert(name, params, model)) is not None:
        return expert
    module, _, leaf = name.rpartition(".")
    parent, _, child = module.rpartition(".")
    if child in packed:
        fused, index = packed[child]
        param_name = f"{parent}.{fused}.{leaf}" if parent else f"{fused}.{leaf}"
        if param_name in params:
            layer = model.get_submodule(param_name.rpartition(".")[0])
            return param_name, _shard_ids(layer)[index], None
    return None


def load_weights(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Load `weights` into `model`. Returns the params loaded."""
    if (custom := getattr(model, "load_weights", None)) is not None:
        return custom(weights)  # e.g. the Transformers backend, converting first
    return load_with_ops(model, weights)


def load_with_ops(
    model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Load `weights` into `model` with Transformers' ops. Returns the params loaded."""
    if conversions := getattr(model, "weight_conversions", None):
        # The model's own Transformers conversions, on checkpoint names
        from vllm.model_executor.models.transformers.conversion import (
            convert_weights,
        )

        converters = [c for c in conversions if isinstance(c, WeightConverter)]
        renamings = [c for c in conversions if not isinstance(c, WeightConverter)]
        weights = convert_weights(weights, converters, renamings=renamings)
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        weights = mapper.apply(weights)
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    packed = _packed(model)
    loaded: set[str] = set()
    unknown: list[str] = []
    pending: dict[str, dict] = {}
    for name, tensor in weights:
        resolved = _resolve(name, params, packed, model)
        if resolved is None:
            unknown.append(name)
            continue
        param_name, shard, expert = resolved
        shard = getattr(tensor, "shard_id", shard)  # from `orig_to_new_stacked`
        param = params[param_name]
        tensor = _param_ops(param, tensor)
        layer = model.get_submodule(param_name.rpartition(".")[0])
        if callable(custom := getattr(param, "weight_loader", None)):
            # a param with its own layout, e.g. mamba's conv1d or sharded sinks
            custom(param, tensor) if shard is None else custom(param, tensor, shard)
        elif shard == "fused":
            _load_fused_experts(layer, param, tensor, param_name)
        elif expert is not None:
            _load_expert(layer, param, tensor, expert, shard, param_name)
        elif isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            tensors = pending.setdefault(param_name, {})
            tensors[shard] = tensor
            stacked = isinstance(layer, StackedColumnParallelLinear)
            if shard is not None and stacked and len(tensors) < len(_shard_ids(layer)):
                continue
            _load_linear(layer, param, pending.pop(param_name))
        else:
            _load_other(param, tensor)
        loaded.add(param_name)
    if pending:
        raise ValueError(f"Missing shards of fused params: {sorted(pending)[:8]}")
    if unknown:
        logger.warning(
            "%d checkpoint tensors match no param (first: %s)",
            len(unknown),
            unknown[:5],
        )
    return loaded
