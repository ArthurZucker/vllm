# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online LUT-B quantization of routed MoE experts (MXFP8 activations)."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBEmulationTritonExperts,
    LutBSm107Experts,
    make_lut_b_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_SM107_TILE_BYTES,
    LUT_B_SM107_TILE_K,
    LUT_B_SM107_TILE_N,
    dequantize_lut_b,
    lut_b_to_sm107_tiles,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kLutBStatic,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

# Weights per SM107 tile
_TILE = LUT_B_SM107_TILE_N * LUT_B_SM107_TILE_K


class LutBOnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16/FP16 routed experts to LUT-B while loading.

    Weights use 3-bit indices into one E4M3 codebook (with a UE8M0 scale) per
    8x64 tile and activations are MXFP8. Rubin runs the native LUT-B tensor
    core kernel; other GPUs emulate the same numerics.
    """

    def __init__(self, *, moe: FusedMoEConfig):
        super().__init__(moe)
        if moe.has_bias:
            raise ValueError("LUT-B online MoE quantization does not support bias.")

        if moe.moe_backend not in ("auto", "emulation"):
            raise ValueError(
                f"moe_backend={moe.moe_backend!r} is not supported for LUT-B "
                "MoE; expected 'auto' or 'emulation'."
            )
        native_shape = (
            moe.hidden_dim % LUT_B_SM107_TILE_K == 0
            and moe.intermediate_size_per_partition % LUT_B_SM107_TILE_N == 0
        )
        supported, reason = LutBSm107Experts.is_supported_config(
            LutBSm107Experts,
            moe,
            kLutBStatic,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
        self.experts_cls: type[LutBSm107Experts | LutBEmulationTritonExperts]
        if moe.moe_backend == "emulation":
            self.experts_cls = LutBEmulationTritonExperts
        elif supported and native_shape:
            self.experts_cls = LutBSm107Experts
        else:
            self.experts_cls = LutBEmulationTritonExperts
            if reason is None:
                reason = "hidden and intermediate sizes must be multiples of 128"
            logger.info_once("LUT-B MoE falls back to emulation: %s", reason)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # Both experts quantize activations themselves, and the native weights
        # carry their codebooks and scales inline.
        return FUSED_MOE_UNQUANTIZED_CONFIG

    @property
    def _native(self) -> bool:
        return self.experts_cls is LutBSm107Experts

    def quantized_params(
        self, num_experts: int, w13_rows: int, intermediate: int, hidden: int
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        params = []
        for name, rows, cols in (
            ("w13_weight", w13_rows, hidden),
            ("w2_weight", hidden, intermediate),
        ):
            if self._native:  # SM107 tiles, viewed as rows of their bytes
                shape = (num_experts, rows, cols * LUT_B_SM107_TILE_BYTES // _TILE)
                params.append((name, shape, torch.uint8))
            else:  # emulation: the dequantized weights
                params.append((name, (num_experts, rows, cols), self.moe.in_dtype))
        return params

    def expert_scale_names(self, prefix: str) -> list[str]:
        return []  # the native tiles carry their codebooks and scales inline

    def quantize_experts(
        self, layer: RoutedExperts, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        packed, codebooks, scales = quantize_lut_b(weight)
        if self._native:
            return (lut_b_to_sm107_tiles(packed, codebooks, scales),)
        return (dequantize_lut_b(packed, codebooks, scales, weight.dtype),)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        for name, weight in zip(
            ("w13_weight", "w2_weight"), self.quantized_weights(layer)
        ):
            if self._native and weight.dim() == 3:  # loaded: back to its tiles
                num_experts, rows, width = weight.shape
                cols = width * _TILE // LUT_B_SM107_TILE_BYTES
                weight = weight.view(
                    num_experts,
                    rows // LUT_B_SM107_TILE_N,
                    cols // LUT_B_SM107_TILE_K,
                    LUT_B_SM107_TILE_BYTES,
                )
            replace_parameter(layer, name, weight)
        self.release_requantization_source_weights(layer)

        quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant_config
        self.moe_kernel = make_lut_b_moe_kernel(
            self.experts_cls,
            self.moe,
            quant_config,
            layer._expert_routing_tables(),
        )
        layer._already_called_process_weights_after_loading = True
