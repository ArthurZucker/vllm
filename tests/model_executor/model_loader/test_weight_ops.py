# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each op of `weight_ops`, on CPU tensors: what it makes of its sources."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader import weight_ops as ops

T = "t"  # one source, one target


def run(op, inputs, sources=(T,), targets=(T,), **kwargs):
    return op.convert(dict(inputs), list(sources), list(targets), **kwargs)


def model_with(param: torch.Tensor, **attrs) -> tuple[nn.Module, dict]:
    """A model holding `param` as `layer.p` (with `attrs` on it), and the kwargs an
    op gets for that param"""
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.p = nn.Parameter(param, requires_grad=False)
    for k, v in attrs.items():
        setattr(model.layer.p, k, v)
    return model, {"model": model, "full_layer_name": "layer.p"}


def config(**text):
    ns = SimpleNamespace(**text)
    return SimpleNamespace(get_text_config=lambda: ns)


def test_output_concatenate_along_output_dim():
    _, kw = model_with(torch.zeros(2, 4), output_dim=1)
    out = run(
        ops.OutputConcatenate(),
        {"a": [torch.ones(2, 2)], "b": [torch.zeros(2, 2)]},
        ("a", "b"),
        **kw,
    )
    assert next(iter(out.values())).shape == (2, 4)


def test_pad_to_param():
    _, kw = model_with(torch.zeros(4, 2), pad_value=-1.0)
    out = run(ops.PadToParam(), {T: [torch.ones(3, 2)]}, **kw)[T]
    assert out[:3].eq(1).all() and out[3].eq(-1).all()


def test_repeat_last_source():
    out = run(ops.RepeatLastSource(), {"q": [1], "k": [2]}, ("q", "k"))
    assert out == {"q": [1], "k": [2, 2]}


def test_zero_missing_source():
    out = run(ops.ZeroMissingSource(), {"q": [torch.ones(3)]}, ("q", "k", "v"))
    assert set(out) == {"q", "k", "v"} and out["k"][0].eq(0).all()


def test_neg_exp_float64_rounding():
    x = torch.randn(100)
    out = run(ops.NegExp(), {T: [x]})[T][0]
    assert out.dtype == torch.float32 and torch.equal(
        out, -torch.exp(x.double()).float()
    )


def test_affine():
    assert run(ops.Affine(2.0, 1.0), {T: [torch.ones(2)]})[T][0].eq(3).all()


def test_gather_bytes():
    x = torch.arange(4, dtype=torch.uint8).view(torch.float8_e4m3fn)
    out = run(ops.Gather(torch.tensor([3, 0]), 0), {T: [x]})[T][0]
    assert out.view(torch.uint8).tolist() == [3, 0]


def test_swap_halves():
    out = run(ops.SwapHalves(), {T: [torch.arange(4)]})[T]
    assert out.tolist() == [2, 3, 0, 1]


def test_deinterleave():
    # 2 heads, parts (q, k), head_dim 1: [q0, k0, q1, k1] -> [q0, q1, k0, k1]
    out = run(
        ops.Deinterleave(2),
        {T: [torch.tensor([0, 10, 1, 11])]},
        config=config(num_attention_heads=2),
    )[T]
    assert out.tolist() == [0, 1, 10, 11]


def test_grouped_qkv():
    # 4 heads, 2 kv heads, head_dim 1: per group [q, q, k, v]
    x = torch.tensor([0, 1, 10, 20, 2, 3, 11, 21])
    out = run(ops.GroupedQKV(4, 2), {T: [x]})[T]
    assert out.tolist() == [0, 1, 2, 3, 10, 11, 20, 21]


def test_expert_rows():
    out = run(
        ops.ExpertRows("moe.n"),
        {T: [torch.zeros(6, 2)]},
        config=config(moe=SimpleNamespace(n=3)),
    )[T][0]
    assert out.shape == (3, 2, 2)


def test_head_permute():
    x = torch.tensor([[0, 1], [2, 3]]).reshape(4)  # 2 heads of 2 channels
    out = run(ops.HeadPermute(torch.tensor([1, 0]), head_dim=2), {T: [x]})[T][0]
    assert out.tolist() == [1, 0, 3, 2]


def test_rank_chunk_replicates_fewer_rows():
    x = torch.arange(2)  # 2 kv heads, 4 ranks: ranks 2 and 3 take row 1
    assert run(ops.RankChunk(3, 4), {T: [x]})[T][0].tolist() == [1]
    assert run(ops.RankChunk(1, 2), {T: [torch.arange(4)]})[T][0].tolist() == [2, 3]


def test_view_as_param():
    _, kw = model_with(torch.zeros(2, dtype=torch.uint8))
    x = torch.tensor([1, 2], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    out = run(ops.ViewAsParam(), {T: [x]}, **kw)[T][0]
    assert out.dtype == torch.uint8 and out.tolist() == [1, 2]


def test_param_rows():
    _, kw = model_with(torch.zeros(2), row_start=1)
    assert run(ops.ParamRows(), {T: [torch.arange(4)]}, **kw)[T][0].tolist() == [1, 2]


def test_narrow_only_full_length():
    op = ops.Narrow(0, 1, 2, full=4)
    assert run(op, {T: [torch.arange(4)]})[T][0].tolist() == [1, 2]
    assert run(op, {T: [torch.arange(2)]})[T][0].tolist() == [0, 1]


def test_refuse():
    with pytest.raises(RuntimeError, match="no reload"):
        run(ops.Refuse("no reload"), {T: [torch.ones(1)]})


def test_reciprocal():
    assert run(ops.Reciprocal(), {T: [torch.tensor([4.0])]})[T][0].item() == 0.25


def test_equal_sources():
    one = torch.tensor([1.0])
    assert run(ops.EqualSources(), {"a": [one], "b": [one]}, ("a", "b"))[T] is one
    with pytest.raises(ValueError):
        run(ops.EqualSources(), {"a": [one], "b": [one * 2]}, ("a", "b"))


def test_repeat_to_param():
    _, kw = model_with(torch.zeros(2, 4))
    out = run(ops.RepeatToParam(), {T: [torch.tensor([[1, 2]])]}, **kw)[T][0]
    assert out.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2]]


def test_merge_lora():
    w = torch.zeros(2, 2)
    a, b = torch.ones(1, 1, 2), torch.ones(1, 2, 1)  # one adapter, rank 1
    out = run(
        ops.MergeLora(0.5, 0),
        {"x.original": [w], "x.0.lora_A": [a], "x.0.lora_B": [b]},
        ("x.original", "x.0.lora_A", "x.0.lora_B"),
        full_layer_name="layer.weight",
    )[T]
    assert out.eq(0.5).all()


def test_expert_scalars():
    model, kw = model_with(torch.zeros(2, 2))
    model.layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(ep_rank=1)
    )
    gate, up = torch.arange(4.0), torch.tensor([7.0])  # 4 experts on disk; one for all
    out = run(ops.ExpertScalars(), {"g": [gate], "u": [up]}, ("g", "u"), **kw)[T]
    assert out.tolist() == [[2, 7], [3, 7]]  # rank 1's experts 2, 3
