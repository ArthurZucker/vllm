# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 MegaMoE: block-FP8 checkpoint experts requantized to the MXFP8
MegaMoE kernel layout after loading, and BF16 experts laid out as they are."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import MegaMoEActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    requant_block_fp8_to_ue8m0,
)
from vllm.model_executor.model_loader.ops_loader import load_with_ops
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.glm5next.nvidia.mega_moe import Glm5NextMegaMoEExperts

BLOCK = (128, 128)
HIDDEN, INTER = 256, 128
FP8 = torch.float8_e4m3fn
NUM_EXPERTS = 4
# Global ids of the experts _experts() places on this rank.
LOCAL_EXPERTS = slice(2, 4)


@pytest.fixture(autouse=True)
def sm100_mega_moe_layout(monkeypatch):
    """Build the SM100 kernel layout on any device."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _block_fp8(real: torch.Tensor, power_of_two: bool = False):
    """Quantize `(..., mn, k)` to E4M3 with one float32 scale per 128x128."""
    *lead, mn, k = real.shape
    blocks = real.reshape(*lead, mn // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(-3, -1)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    full = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (real / full).to(FP8), scale


def _dequant_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


def _dequant_ue8m0(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return weight.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)


def test_requant_is_exact_for_power_of_two_block_scales():
    torch.manual_seed(1)
    real = torch.randn(1, 256, 256) * 0.05
    weight, scale = _block_fp8(real, power_of_two=True)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale)

    # A finer power-of-two scale only shifts exponents, so nothing rounds.
    assert torch.equal(_dequant_ue8m0(requant, sf), _dequant_block(weight, scale))


def _experts(weight_block_size=BLOCK) -> Glm5NextMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return Glm5NextMegaMoEExperts(
        vllm_config,  # type: ignore[arg-type]
        num_experts=NUM_EXPERTS,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        weight_block_size=weight_block_size,
        activation=MegaMoEActivation("swiglu", clamp=10.0),
        num_shared_experts=1,
        prefix="model.layers.3.mlp.experts",
    )


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Block-FP8 experts as stored: gate rows, then up rows, for w13."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    w13, s13 = _block_fp8(rand(NUM_EXPERTS, 2 * INTER, HIDDEN))
    w2, s2 = _block_fp8(rand(NUM_EXPERTS, HIDDEN, INTER))
    return {
        "w13_weight": w13,
        "w13_weight_scale_inv": s13,
        "w2_weight": w2,
        "w2_weight_scale_inv": s2,
    }


def _projections(name: str, tensor: torch.Tensor):
    """`w13_<suffix>` as its gate and up rows, `w2_<suffix>` as the down one."""
    param, _, suffix = name.partition("_")
    if param == "w2":
        return [(f"down_proj.{suffix}", tensor)]
    gate, up = tensor.chunk(2)
    return [(f"gate_proj.{suffix}", gate), (f"up_proj.{suffix}", up)]


def _names(ckpt, shared_ckpt=None, prefix=""):
    """The checkpoint's tensors: per expert and projection, then the shared
    expert's."""
    names = [
        (f"{prefix}{e}.{proj}", t)
        for name, tensor in ckpt.items()
        for e, expert in enumerate(tensor)
        for proj, t in _projections(name, expert)
    ]
    return names + [
        (f"{prefix}shared_experts.{proj}", t)
        for name, tensor in (shared_ckpt or {}).items()
        for proj, t in _projections(name, tensor)
    ]


def _interleave(t):
    """Gate and up rows interleaved by 8, as DeepGEMM's MegaMoE L1 takes them."""
    g, n, k = t.shape
    halves = t.view(g, 2, n // 16, 8, k)
    return torch.stack([halves[:, 0], halves[:, 1]], dim=2).reshape(g, n, k)


def _reference_mega_moe_layout(w13, s13, w2, s2):
    """CPU copy of DeepGEMM's transform_weights_for_mega_moe on packed scales."""

    def utccp(sf):
        g, mn, k = sf.shape
        return sf.reshape(g, -1, 4, 32, k).transpose(2, 3).reshape(g, mn, k)

    return (
        _interleave(w13.view(torch.uint8)),
        utccp(_interleave(s13.view(torch.int32))),
        w2.view(torch.uint8),
        utccp(s2.view(torch.int32)),
    )


def _expected_routed(ckpt):
    w13, s13 = requant_block_fp8_to_ue8m0(
        ckpt["w13_weight"][LOCAL_EXPERTS], ckpt["w13_weight_scale_inv"][LOCAL_EXPERTS]
    )
    w2, s2 = requant_block_fp8_to_ue8m0(
        ckpt["w2_weight"][LOCAL_EXPERTS], ckpt["w2_weight_scale_inv"][LOCAL_EXPERTS]
    )
    return _reference_mega_moe_layout(w13, s13, w2, s2)


def _assert_layout(actual, expected):
    for tensor, reference in zip(actual, expected):
        assert torch.equal(tensor.view(reference.dtype), reference)


def _routed(experts):
    return (*experts.l1_weights, *experts.l2_weights)


def test_block_fp8_checkpoint_requantized_into_kernel_layout():
    """This rank's experts load as stored, then are rounded to UE8M0 1x32 scales
    and laid out for the kernel."""
    experts = _experts()
    ckpt, shared_ckpt = _checkpoint(seed=0), _shared_checkpoint(seed=0)
    load_with_ops(experts, _names(ckpt, shared_ckpt))
    experts.quant_method.process_weights_after_loading(experts)
    _assert_layout(_routed(experts), _expected_routed(ckpt))
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt))


def _shared_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w13, s13 = _block_fp8(torch.randn(1, 2 * INTER, HIDDEN, generator=generator) * 0.02)
    w2, s2 = _block_fp8(torch.randn(1, HIDDEN, INTER, generator=generator) * 0.02)
    return {
        "w13_weight": w13[0],
        "w13_weight_scale_inv": s13[0],
        "w2_weight": w2[0],
        "w2_weight_scale_inv": s2[0],
    }


def _expected_shared(ckpt):
    q13, sf13 = requant_block_fp8_to_ue8m0(
        ckpt["w13_weight"][None], ckpt["w13_weight_scale_inv"][None]
    )
    q2, sf2 = requant_block_fp8_to_ue8m0(
        ckpt["w2_weight"][None], ckpt["w2_weight_scale_inv"][None]
    )
    return [t[0] for t in _reference_mega_moe_layout(q13, sf13, q2, sf2)]


def _shared(experts):
    return (
        experts.shared_l1_weight,
        experts.shared_l1_weight_scale,
        experts.shared_l2_weight,
        experts.shared_l2_weight_scale,
    )


class _Model(torch.nn.Module):
    """Finishes a load the way the GLM model, wrapper and MTP drafter do."""

    def __init__(self):
        super().__init__()
        self.experts = _experts()


@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_reload_requantizes_into_the_kernel_tensors(monkeypatch, mode):
    """A reload writes the block-FP8 experts in the checkpoint layout and finishes
    with the model's process_weights_after_loading; the kernel tensors keep their
    storage."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model_config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False, quantization=None
    )
    device = torch.get_default_device()
    model = _Model()
    experts = model.experts
    record_metadata_for_reloading(model)
    ckpt, shared_ckpt = _checkpoint(seed=0), _shared_checkpoint(seed=0)
    load_with_ops(model, _names(ckpt, shared_ckpt, prefix="experts."))
    process_weights_after_loading(model, model_config, device)  # type: ignore[arg-type]
    _assert_layout(_routed(experts), _expected_routed(ckpt))
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt))
    storage = {n: p.data_ptr() for n, p in experts.named_parameters()}

    ckpt, shared_ckpt = _checkpoint(seed=1), _shared_checkpoint(seed=1)
    start_reload(model)
    load_with_ops(model, _names(ckpt, shared_ckpt, prefix="experts."))
    # Every loaded tensor is in, so the layer is not meta any more: the derived
    # kernel scales do not count as missing loads.
    assert not experts.w13_weight.is_meta
    finish_reload(model, model_config, device)  # type: ignore[arg-type]

    _assert_layout(_routed(experts), _expected_routed(ckpt))
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt))
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == storage


def test_bf16_checkpoint_loads_as_is():
    """Without a weight block size the experts load BF16 tensors and lay them
    out for the BF16 kernel, the shared expert included, staging no scales."""
    experts = _experts(weight_block_size=None)
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    w13, w2 = rand(NUM_EXPERTS, 2 * INTER, HIDDEN), rand(NUM_EXPERTS, HIDDEN, INTER)
    shared13, shared2 = rand(1, 2 * INTER, HIDDEN), rand(HIDDEN, INTER)
    shared_ckpt = {"w13_weight": shared13[0], "w2_weight": shared2}
    load_with_ops(experts, _names({"w13_weight": w13, "w2_weight": w2}, shared_ckpt))
    experts.prepare_weights()

    assert not [n for n, _ in experts.named_parameters() if "scale" in n]
    assert torch.equal(experts.w13_weight, _interleave(w13[LOCAL_EXPERTS]))
    assert torch.equal(experts.w2_weight, w2[LOCAL_EXPERTS])
    assert torch.equal(experts.shared_l1_weight, _interleave(shared13)[0])
    assert torch.equal(experts.shared_l2_weight, shared2)
