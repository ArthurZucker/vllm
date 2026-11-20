# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_BF16,
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    requant_block_fp8_to_ue8m0,
)
from vllm.model_executor.utils import is_weights_pre_processed

logger = init_logger(__name__)


class Glm5NextMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from a GLM-5.3 checkpoint.

    A BF16 checkpoint (`weight_block_size` None) runs the BF16 kernel. GLM-5.3-Flash's
    block-FP8 checkpoint stores E4M3 weights with one float32 scale per
    `weight_block_size` block, loaded into `*_weight_scale_inv`; the kernel takes
    power-of-two UE8M0 scales per 1x32, which `prepare_weights` rounds the weights
    to in place before the kernel layout. The shared expert is fused the same way.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        weight_block_size: tuple[int, int] | None,
        num_shared_experts: int = 0,
        **kwargs,
    ):
        self.weight_block_size = weight_block_size
        self._num_fused_shared_experts = num_shared_experts
        weight_format = MEGA_MOE_BF16 if weight_block_size is None else MEGA_MOE_MXFP8
        super().__init__(vllm_config, weight_format=weight_format, **kwargs)

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        n = self._num_fused_shared_experts
        return n, self.weight_format.routed if n else None

    def _create_weights(self) -> None:
        super()._create_weights()
        if self.weight_block_size is None:
            return
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        self._register_block_scale("w13", (e, 2 * i, h))
        self._register_block_scale("w2", (e, h, i))
        if self.has_fused_shared_experts:
            s = i * self.num_shared_experts
            self._register_block_scale("shared_l1", (2 * s, h))
            self._register_block_scale("shared_l2", (h, s))

    def _register_block_scale(self, name: str, shape: tuple[int, ...]) -> None:
        """Stage the checkpoint's float32 block scales of `name`. The kernel's
        `{name}_weight_scale` is derived from them and never loaded."""
        *outer, n, k = shape
        assert self.weight_block_size is not None
        block_m, block_k = self.weight_block_size
        scale = torch.zeros(*outer, n // block_m, k // block_k, dtype=torch.float32)
        self.register_parameter(
            f"{name}_weight_scale_inv", nn.Parameter(scale, requires_grad=False)
        )
        kernel_scale = getattr(self, f"{name}_weight_scale")
        kernel_scale.weight_loader_numel = 0
        kernel_scale.loaded_at_init = True

    @torch.no_grad()
    def prepare_weights(self, chunk: int = 8) -> None:
        """Round loaded block-FP8 weights to UE8M0 1x32 scales in place, then lay
        them out for the kernel."""
        for name in ("w13", "w2", "shared_l1", "shared_l2"):
            if is_weights_pre_processed() or not hasattr(
                self, f"{name}_weight_scale_inv"
            ):
                continue
            weight, scale, block_scale = (
                getattr(self, f"{name}_{suffix}").data
                for suffix in ("weight", "weight_scale", "weight_scale_inv")
            )
            # Power-of-two block scales requantize exactly; any other is rounded.
            if not torch.equal(block_scale, block_scale.log2().round().exp2()):
                assert self.weight_block_size is not None
                logger.warning_once(
                    "GLM-5.3-Flash on the deep_gemm_mega_moe backend: the "
                    "checkpoint's %dx%d block-FP8 MoE scales are not all powers of "
                    "two, so requantizing them to MXFP8 (UE8M0 scales per 1x32) "
                    "changes the expert weights. Use another moe_backend to run the "
                    "block-FP8 weights as they are.",
                    *self.weight_block_size,
                )
            if weight.dim() == 2:  # the shared expert
                weight, scale = weight[None], scale[None]
                block_scale = block_scale[None]
            for start in range(0, len(weight), chunk):
                end = start + chunk
                w, sf = requant_block_fp8_to_ue8m0(
                    weight[start:end], block_scale[start:end]
                )
                weight[start:end].copy_(w)
                scale[start:end].copy_(sf)
        super().prepare_weights()
