# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 routed experts re-quantized to MXFP4 at load (ROCm).

`VLLM_FP8_MOE_REQUANT=mxfp4` dequantizes each block-FP8 checkpoint expert to
bf16 as it loads, and quantizes the merged experts to MXFP4 (e2m1 values, e8m0
block-32 scales) for the aiter MXFP4 MoE kernels: half the expert bytes of FP8.
`VLLM_FP8_MOE_REQUANT_ACT` selects the activation dtype: a8w4 (fp8 per-token;
aiter's interleaved FlyDSL kernels, the `AITER_MXFP4_BF16` backend; default),
a4w4 (MXFP4 activations, `AITER_MXFP4_MXFP4`) or a16w4 (bf16,
`AITER_TRITON_MXFP4_BF16`).
"""

from functools import partial

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

logger = init_logger(__name__)


def dequant_block_fp8_to_bf16(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: list[int]
) -> torch.Tensor:
    """`[N, K]` fp8 x `[ceil(N/bn), ceil(K/bk)]` fp32 block scales -> bf16."""
    n, k = weight.shape
    bn, bk = block_size
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


class Fp8BlockExpertsAsMxfp4FusedMoE(Mxfp4OnlineMoEMethod):
    """Block-FP8 routed experts stored as MXFP4 (VLLM_FP8_MOE_REQUANT=mxfp4).

    The plan requantizes them as they load (`checkpoint_quantizer`): each
    checkpoint expert (fp8 values + block `weight_scale_inv`) is dequantized to
    bf16 before the experts merge, the merged experts quantized to MXFP4 after.
    Rounding the values to the e2m1 grid with power-of-two block-32 scales is
    lossy: the weight error is of the order of an MXFP4 quantization of the bf16
    weights.
    """

    def __init__(self, quant_config, layer: RoutedExperts) -> None:
        act = envs.VLLM_FP8_MOE_REQUANT_ACT.lower()
        moe_config = layer.moe_config
        # a4w4 asks for MXFP4 activations (AITER_MXFP4_MXFP4). With bf16
        # activations the oracle picks AITER_MXFP4_BF16, whose interleaved
        # weight layout makes aiter quantize activations to fp8 (a8w4); true
        # bf16 activations need the aiter Triton backend.
        self.activation_quant_key = kMxfp4Dynamic if act == "a4w4" else None  # type: ignore[assignment]
        assert quant_config.weight_block_size is not None, (
            "FP8 re-quant needs block-scaled experts"
        )
        self.fp8_block_size = list(quant_config.weight_block_size)
        if act == "a16w4" and moe_config.moe_backend == "auto":
            moe_config.moe_backend = "aiter_triton_mxfp4_bf16"
        super().__init__(moe=moe_config)
        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            raise ValueError(
                "VLLM_FP8_MOE_REQUANT=mxfp4 needs an MXFP4 MoE backend, and none "
                "is available on this platform."
            )
        self.quant_config = quant_config
        logger.info_once(
            "FP8 block experts re-quantized to MXFP4 at load (%s, %s)",
            act,
            self.mxfp4_backend.value,
        )

    def get_fused_moe_quant_config(self, layer):
        # Triton-format backends keep the scales as PrecisionConfigs (set by
        # _setup_kernel), not as layer parameters.
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=self.w13_precision_config,
                w2_scale=self.w2_precision_config,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                layer=layer,
            )
        return super().get_fused_moe_quant_config(layer)

    def checkpoint_quantizer(self, layer: RoutedExperts):
        """Each FP8 expert dequantized before the experts merge, the merged
        experts quantized to MXFP4 after (the online MXFP4 quantize op)"""
        from vllm.model_executor.models.transformers.conversion import (
            VllmRequantizeQuantizer,
        )

        dequantize = partial(dequant_block_fp8_to_bf16, block_size=self.fp8_block_size)
        targets = super().checkpoint_quantizer(layer).targets
        return VllmRequantizeQuantizer(
            targets, dequantize, ["weight", "weight_scale_inv"]
        )
