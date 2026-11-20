# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion ops vLLM adds to Transformers' (`transformers.core_model_loading`)."""

import operator

import torch
from transformers.core_model_loading import Concatenate, ConversionOps


def _each(input_dict: dict, fn) -> dict:
    """`fn` over every tensor, Transformers' lists of them as well"""
    return {
        k: [fn(t) for t in v] if isinstance(v, list) else fn(v)
        for k, v in input_dict.items()
    }


def _one(value) -> torch.Tensor:
    """A source's tensor (Transformers passes lists)"""
    return value[0] if isinstance(value, list) else value


def _first(input_dict: dict) -> torch.Tensor:
    """The one tensor of a single-source op"""
    return _one(next(iter(input_dict.values())))


class OutputConcatenate(Concatenate):
    """`Concatenate` along the target param's `output_dim` (an AWQ `qweight` packs its
    output along dim 1); 0-dim tensors (per-tensor scales) are stacked"""

    def __init__(self):
        super().__init__(dim=0)

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        model, name = kwargs.get("model"), kwargs.get("full_layer_name", "")
        try:
            param = model.get_parameter(name)
        except (AttributeError, TypeError):
            param = None
        self.dim = getattr(param, "output_dim", None) or 0
        return super().convert(input_dict, source_patterns, target_patterns, **kwargs)


class PadToParam(ConversionOps):
    """The tensor padded to its target param's shape with the param's `pad_value`
    (default 0): a vocab padded to a multiple of the ranks, sink heads (-inf)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        param = kwargs["model"].get_parameter(kwargs["full_layer_name"])
        tensor = _first(input_dict)
        if tensor.shape != param.shape and tensor.dim() == param.dim():
            padded = tensor.new_full(param.shape, getattr(param, "pad_value", 0))
            padded[tuple(slice(0, n) for n in tensor.shape)] = tensor
            tensor = padded
        return {target_patterns[0]: tensor}


class RepeatLastSource(ConversionOps):
    """The last source taken twice: a checkpoint shipping `k_proj` only, loaded into
    both the K and V shards of a fused `qkv_proj` (Gemma 4's `attention_k_eq_v`)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        last = input_dict[source_patterns[-1]]
        input_dict[source_patterns[-1]] = (
            [*last, *last] if isinstance(last, list) else [last, last]
        )
        return input_dict


class ZeroMissingSource(ConversionOps):
    """A source the checkpoint does not ship, as zeros shaped like a present one:
    Whisper's `k_proj` has no bias, the fused `qkv_proj.bias` has a K slot"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        present = _first(input_dict)
        for pattern in source_patterns:
            input_dict.setdefault(pattern, [torch.zeros_like(present)])
        return input_dict


class NegExp(ConversionOps):
    """`-exp(x)`, in float32: Mamba's `A` from the checkpoint's `A_log`"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        # in float64: float32 `exp` rounds differently on GPU and CPU (upstream's)
        return _each(input_dict, lambda t: -torch.exp(t.double()).float())


class Affine(ConversionOps):
    """`x * scale + offset`, for norms stored shifted or scaled (PLaMo 3's `1 + w`,
    LongCat's MLA LoRA scaling)"""

    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale, self.offset = scale, offset

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return _each(input_dict, lambda t: t * self.scale + self.offset)


class Gather(ConversionOps):
    """This rank's shard reordered by `index` along `dim` (DeepSeek V4.1's fused
    layouts); 1-byte dtypes (fp8, ue8m0) are gathered as bytes"""

    def __init__(self, index: torch.Tensor, dim: int):
        self.index, self.dim = index, dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        def gather(t):
            raw = t.view(torch.uint8) if t.element_size() == 1 else t
            return raw.index_select(self.dim, self.index.to(t.device)).view(t.dtype)

        return _each(input_dict, gather)


class SwapHalves(ConversionOps):
    """The two halves of `dim` swapped: a fused projection stored `[up, gate]` where
    the param is `[gate, up]` (GTE's `up_gate_proj`)"""

    needs_full = True  # the halves are the whole tensor's

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        first, second = tensor.chunk(2, dim=self.dim)
        return {target_patterns[0]: torch.cat([second, first], dim=self.dim)}


class Deinterleave(ConversionOps):
    """A fused projection stored per head (`[heads, parts, head_dim]`, Bloom's
    `query_key_value`) as its parts one after the other (`[parts, heads, head_dim]`)"""

    needs_full = True  # the heads are the whole tensor's

    def __init__(self, parts: int, heads_attr: str = "num_attention_heads"):
        self.parts, self.heads_attr = parts, heads_attr

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        heads = getattr(kwargs["config"].get_text_config(), self.heads_attr)
        shape = tensor.shape
        tensor = tensor.view((heads, self.parts, -1) + shape[1:]).transpose(0, 1)
        return {target_patterns[0]: tensor.reshape(shape)}


class GroupedQKV(ConversionOps):
    """A fused `qkv` stored per kv head (`[kv_heads, heads / kv_heads + 2, head_dim]`:
    each group's queries, its key, its value; Falcon's `query_key_value`) as all the
    queries, then the keys, then the values"""

    needs_full = True  # the groups are kv heads before they are a rank's

    def __init__(self, heads: int, kv_heads: int):
        self.heads, self.kv_heads = heads, kv_heads

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = _first(input_dict)
        shape, group = tensor.shape, self.heads // self.kv_heads + 2
        grouped = tensor.view(self.kv_heads, group, -1, *shape[1:])
        q, k, v = grouped.split([group - 2, 1, 1], dim=1)
        parts = [part.reshape(-1, *shape[1:]) for part in (q, k, v)]
        return {target_patterns[0]: torch.cat(parts)}


class ExpertRows(ConversionOps):
    """Experts stacked along the rows (`[experts * rows, cols]`, Nomic MoE, DBRX) as
    `[experts, rows, cols]`, `experts` read from the config (a dotted path)"""

    needs_full = True  # the rows are experts' before they are a rank's

    def __init__(self, experts_attr: str = "num_experts"):
        self.experts_attr = experts_attr

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        config = kwargs["config"].get_text_config()
        experts = operator.attrgetter(self.experts_attr)(config)
        return _each(input_dict, lambda t: t.view(experts, -1, *t.shape[1:]))


class HeadPermute(ConversionOps):
    """Each head's channels gathered by `index` (`head_dim` channels a head), on the
    sources at `positions` (all by default): K2-Horizon's partial-RoPE channels
    reordered for NeoX rope. Per head: sharded heads permute alone; a per-tensor
    scale stays as is"""

    def __init__(self, index: torch.Tensor, head_dim: int, positions=None):
        self.index, self.head_dim, self.positions = index, head_dim, positions

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        chosen = [
            p
            for i, p in enumerate(source_patterns)
            if self.positions is None or i in self.positions
        ]
        for pattern in chosen:
            if (value := input_dict.get(pattern)) is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            input_dict[pattern] = [
                t.view(-1, self.head_dim, *t.shape[1:])[:, self.index.to(t.device)]
                .reshape(t.shape)
                .contiguous()
                if t.numel() > 1
                else t
                for t in tensors
            ]
        return input_dict


class RankChunk(ConversionOps):
    """Rank `rank`'s chunk of `dim` out of `ranks`, for a tensor whose row count is
    known only on read: with fewer rows than ranks (kv heads replicated), consecutive
    ranks take the same row (FoPE's per-kv-head coefficients)"""

    def __init__(self, rank: int, ranks: int, dim: int = 0):
        self.rank, self.ranks, self.dim = rank, ranks, dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        def chunk(t):
            ranks, rank = self.ranks, self.rank
            if t.shape[self.dim] < ranks:
                rank //= ranks // t.shape[self.dim]
                ranks = t.shape[self.dim]
            return t.chunk(ranks, self.dim)[rank]

        return _each(input_dict, chunk)


def _target(kwargs: dict) -> torch.Tensor:
    return kwargs["model"].get_parameter(kwargs["full_layer_name"])


class ViewAsParam(ConversionOps):
    """The tensors' bytes as their target param's dtype, of the same width: ue8m0
    scales (`float8_e8m0fnu`) or packed FP4 into a `uint8` param, which a cast would
    convert by value. The param keeps the sent dtype (`keeps_sent_dtype`)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        dtype = _target(kwargs).dtype

        def view(t):
            same_width = t.element_size() == dtype.itemsize
            return t.view(dtype) if t.dtype != dtype and same_width else t

        return _each(input_dict, view)


class ParamRows(ConversionOps):
    """The target param's rows of the tensor, from its `row_start` (default 0):
    DeepSeek V4.1's engram heads, unevenly sized, each rank a contiguous range"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        param = _target(kwargs)
        start = getattr(param, "row_start", 0)
        return _each(input_dict, lambda t: t.narrow(0, start, param.shape[0]))


class Narrow(ConversionOps):
    """`[start, start + length)` of `dim`: a layer's own experts of the experts
    stacked on disk (mega-MoE layers). With `full`, only a tensor that long on `dim`;
    another is taken as narrowed already"""

    def __init__(self, dim: int, start: int, length: int, full: int | None = None):
        self.dim, self.start, self.length, self.full = dim, start, length, full

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        def narrow(t):
            if self.full is not None and t.shape[self.dim] != self.full:
                return t
            return t.narrow(self.dim, self.start, self.length)

        value = next(iter(input_dict.values()))
        tensors = value if isinstance(value, list) else [value]
        return {target_patterns[0]: [narrow(t) for t in tensors]}


class Refuse(ConversionOps):
    """Raises instead of loading: a layer whose weights were transformed in place
    for its kernel (Step5's mega-MoE), which a reload would corrupt"""

    def __init__(self, message: str):
        self.message = message

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        raise RuntimeError(self.message)


class Reciprocal(ConversionOps):
    """`1 / x`, in float32: compressed-tensors' global scales (quantization scales)
    as the dequantization scales a param holds"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return _each(input_dict, lambda t: 1.0 / t.float())


class EqualSources(ConversionOps):
    """The first source, which the others must equal (to `rtol`): NVFP4 experts'
    `w1` and `w3` `weight_scale_2`, one L1 scale per expert"""

    def __init__(self, rtol: float = 1e-5):
        self.rtol = rtol

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        first, *others = map(_one, input_dict.values())
        for other in others:
            if not torch.allclose(other.float(), first.float(), rtol=self.rtol):
                raise ValueError(
                    f"{kwargs.get('full_layer_name')}: {source_patterns} differ "
                    f"({first.flatten().tolist()[:4]} vs "
                    f"{other.flatten().tolist()[:4]}); one value is loaded"
                )
        return {target_patterns[0]: first}


class RepeatToParam(ConversionOps):
    """Each dim repeated (`repeat_interleave`) to its target param's size: block
    scales coarser than the param's (an FP8 shared expert's 128x128 UE8M0 scales
    into MegaMoE's 1x32); a scalar into every element (one scale for all experts)"""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        shape = _target(kwargs).shape

        def repeat(t):
            if t.numel() == 1:
                t = t.reshape((1,) * len(shape))
            for dim, (have, want) in enumerate(zip(t.shape, shape)):
                if have != want:
                    if want % have:
                        raise ValueError(f"{tuple(t.shape)} does not tile {shape}")
                    t = t.repeat_interleave(want // have, dim)
            return t

        return _each(input_dict, repeat)


class MergeLora(ConversionOps):
    """A LoRA-parametrized weight merged, `W + (B @ A) * scaling` with one adapter
    (`index`) of the stacked ones: jina-embeddings-v3's `parametrizations`
    (`original`, `0.lora_A`, `0.lora_B`); an embedding's A and B are swapped"""

    needs_full = True  # sharded after the merge

    def __init__(self, scaling: float, index: int = -1):
        self.scaling, self.index = scaling, index

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        def get(part: str) -> torch.Tensor:
            return _one(next(v for k, v in input_dict.items() if part in k))

        weight, lora_a, lora_b = get("original"), get("lora_A"), get("lora_B")
        a, b = lora_a[self.index].float(), lora_b[self.index].float()
        if "embeddings" in kwargs.get("full_layer_name", ""):
            a, b = b, a
        delta = (b @ a).view(weight.shape) * self.scaling
        return {target_patterns[0]: (weight.float() + delta).to(weight.dtype)}


class ExpertScalars(ConversionOps):
    """Per-expert scalars stacked on disk (`[experts]`, or one for all: `[1]`, `[]`;
    Step 3.5 / Step 5's `moe.gate_proj.input_scale`), one source per shard (gate,
    up), as the target param: `[local experts, shards]`, or `[local experts]` where
    the shards share one value (which they must equal). This rank's experts (EP)"""

    needs_full = True  # a scalar for all experts is not the experts' rows yet

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        param = _target(kwargs)
        rows = param.shape[0]
        name = kwargs["full_layer_name"]
        layer = kwargs["model"].get_submodule(name.rpartition(".")[0])
        ep_rank = layer.moe_config.moe_parallel_config.ep_rank

        def local(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(-1).float()
            if t.numel() == 1:
                return t.expand(rows)
            return t if t.numel() == rows else t.narrow(0, ep_rank * rows, rows)

        columns = {
            p: local(_one(input_dict[p])) for p in source_patterns if p in input_dict
        }
        if param.dim() == 2:
            return {target_patterns[0]: torch.stack(list(columns.values()), dim=1)}
        return EqualSources().convert(
            columns, source_patterns, target_patterns, **kwargs
        )
