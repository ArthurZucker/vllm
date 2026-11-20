# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp routed experts on DeepGEMM MegaMoE.

Routed experts stored as MXFP8, MXFP4 or BF16 run as stored.
`--quantization` / `--quantization-config` can force another MegaMoE format, and
the experts are then requantized after loading. Block-FP8 experts,
which MegaMoE cannot run, are requantized to MXFP8 unless a format is forced.
The MTP drafter's config does not carry the online config, so its experts
take the default format. The sigmoid-gated shared expert stays unfused and runs
on an aux stream next to the router.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_MXFP8,
    MEGA_MOE_WEIGHT_FORMATS,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_checkpoint_quant_keys,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp4Static,
    kMxfp8Static,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

# The dtype and values per element of each checkpoint expert format. Formats MegaMoE
# runs share the name of their MEGA_MOE_* format; "fp8_block" is the 128x128 block-FP8
# layout of the official FP8 checkpoint.
_CHECKPOINT_WEIGHTS = {
    "bf16": (torch.bfloat16, 1),
    "fp8_block": (torch.float8_e4m3fn, 1),
    "mxfp8": (torch.float8_e4m3fn, 1),
    "mxfp4": (torch.int8, 2),
}
_ONLINE_FORMATS = {kMxfp8Static: MEGA_MOE_MXFP8, kMxfp4Static: MEGA_MOE_MXFP4}


def _requantize(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    checkpoint_format: str,
    weight_format: MegaMoEWeightFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize `[N, K]` checkpoint expert weights to MXFP8 or MXFP4.

    Dequantizes to FP32, rounds as DeepGEMM's own casts and returns the scales
    as `[N, K // 32]` UE8M0 exponents.
    """
    if checkpoint_format == "mxfp4":
        x = break_fp4_bytes(weight.view(torch.uint8), torch.float32)
    else:
        x = weight.float()
    if scale is not None:
        if checkpoint_format != "fp8_block":
            # UE8M0 exponents.
            scale = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
        rows, cols = x.shape
        scale = scale.float().repeat_interleave(rows // scale.shape[0], 0)
        x = x * scale.repeat_interleave(cols // scale.shape[1], 1)
    deep_gemm = _import_deep_gemm()
    cast = (
        deep_gemm.per_token_cast_to_fp4
        if weight_format.name == MEGA_MOE_MXFP4.name
        else deep_gemm.per_token_cast_to_fp8
    )
    values, sf = cast(x, use_ue8m0=True, gran_k=32)
    # Power-of-two FP32 scales: the exponent bits are the UE8M0 value.
    return values, (sf.view(torch.int32) >> 23).to(torch.uint8)


def _checkpoint_expert_format(
    quant_config: QuantizationConfig | None, prefix: str
) -> str:
    """The format the checkpoint stores the routed experts at `prefix` in."""
    if quant_config is None or isinstance(quant_config, OnlineQuantizationConfig):
        return "bf16"
    if isinstance(quant_config, Fp8Config):
        if is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return "bf16"
        if (
            quant_config.weight_block_size == [128, 128]
            and quant_config.store_dtype is None
        ):
            return "fp8_block"
        raise NotImplementedError(
            "Qwen4Exp MegaMoE reads FP8 experts with 128x128 block scales only."
        )
    formats = {None: "bf16", kMxfp8Static: "mxfp8", kMxfp4Static: "mxfp4"}
    try:
        weight_key, _ = get_checkpoint_quant_keys(quant_config, prefix, RoutedExperts)
        return formats[weight_key]
    except (NotImplementedError, KeyError):
        raise NotImplementedError(
            "Qwen4Exp MegaMoE reads BF16, 128x128 block-FP8, MXFP8 or MXFP4 "
            f"experts; {quant_config.get_name()} stores {prefix} otherwise."
        ) from None


def mega_moe_expert_formats(
    vllm_config: VllmConfig, prefix: str
) -> tuple[str, MegaMoEWeightFormat]:
    """The checkpoint format of the routed experts and the format they run in.

    A format forced through the online quantization config wins. Otherwise the
    experts run as stored, and block-FP8 experts are requantized to MXFP8.
    """
    quant_config = vllm_config.quant_config
    checkpoint_format = _checkpoint_expert_format(quant_config, prefix)
    online_config = (
        quant_config
        if isinstance(quant_config, OnlineQuantizationConfig)
        else getattr(quant_config, "online_quantization_config", None)
    )
    if (
        online_config is not None
        and (resolved := online_config.resolve_quant_method_cls(RoutedExperts, prefix))
        is not None
    ):
        source, quant_key_str, target_pattern, spec, _ = resolved
        weight_format = _ONLINE_FORMATS.get(spec.weight) if spec.weight else None
        if weight_format is None:
            raise NotImplementedError(
                f"Qwen4Exp MegaMoE runs routed experts as mxfp8 or mxfp4; "
                f"quantization_config asks for {quant_key_str} at {prefix}."
            )
        if weight_format.name != checkpoint_format:
            # Counted in the online quantization summary like a requantized layer.
            online_config.quantized_layers[prefix] = (
                source.value,
                quant_key_str,
                target_pattern,
            )
        return checkpoint_format, weight_format
    if checkpoint_format == "fp8_block":
        logger.warning_once(
            "DeepGEMM MegaMoE cannot run the 128x128 block-FP8 routed experts of "
            "this checkpoint, so they are requantized to MXFP8 at load time. Pass "
            '--quantization-config \'{"moe": "mxfp8"}\' (or "mxfp4") to choose '
            "the format explicitly."
        )
        return checkpoint_format, MEGA_MOE_MXFP8
    return checkpoint_format, MEGA_MOE_WEIGHT_FORMATS[checkpoint_format]


class Qwen4ExpMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from per-expert or fused checkpoint tensors.

    Experts stored in `weight_format` load as they are. Others load in their
    checkpoint format and are requantized after loading, before the kernel layout.
    """

    def __init__(
        self, vllm_config: VllmConfig, checkpoint_format: str, **kwargs
    ) -> None:
        self.checkpoint_format = checkpoint_format
        super().__init__(vllm_config, activation=MegaMoEActivation(), **kwargs)

    @property
    def _requantizes(self) -> bool:
        return self.checkpoint_format != self.weight_format.name

    def _create_weights(self) -> None:
        super()._create_weights()
        if not self._requantizes:
            return
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        dtype, per_element = _CHECKPOINT_WEIGHTS[self.checkpoint_format]
        for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
            weight = torch.zeros(e, n, k // per_element, dtype=dtype)
            self.register_parameter(
                f"{name}_weight", nn.Parameter(weight, requires_grad=False)
            )
            if self.checkpoint_format == "fp8_block":
                scale = torch.zeros(e, n // 128, k // 128, dtype=torch.float32)
                self.register_parameter(
                    f"{name}_weight_scale_inv", nn.Parameter(scale, requires_grad=False)
                )
            if self.checkpoint_format in ("bf16", "fp8_block"):
                # Derived from the weights after loading.
                getattr(self, f"{name}_weight_scale").loaded_at_init = True

    @torch.no_grad()
    def prepare_weights(self) -> None:
        if self._requantizes and not is_weights_pre_processed():
            scale_suffix = {"fp8_block": "weight_scale_inv", "bf16": None}.get(
                self.checkpoint_format, "weight_scale"
            )
            for name in ("w13", "w2"):
                weight = getattr(self, f"{name}_weight")
                scales = (
                    [None] * len(weight)
                    if scale_suffix is None
                    else getattr(self, f"{name}_{scale_suffix}").data
                )
                values, sf = zip(
                    *(
                        _requantize(w, s, self.checkpoint_format, self.weight_format)
                        for w, s in zip(weight.data, scales)
                    )
                )
                # DeepGEMM's FP4 cast may pack into uint8.
                weight.data = torch.stack(values).view(self.weight_format.routed.dtype)
                getattr(self, f"{name}_weight_scale").data.copy_(torch.stack(sf))
        super().prepare_weights()


class Qwen4ExpMegaMoeBlock(nn.Module):
    """Qwen3Next-style MoE whose routed experts run as one MegaMoE launch.

    Takes this rank's tokens and returns their routed plus shared-expert
    output; no collective follows.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE requires expert parallel. Enable it with "
                "--enable-expert-parallel."
            )
        placement = parallel_config.expert_placement_strategy
        if placement != "linear":
            # The EP weight filter would hand this rank other experts.
            raise NotImplementedError(
                "Qwen4Exp MegaMoE places experts linearly; got "
                f"--expert-placement-strategy {placement}."
            )
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > 1:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE supports tensor parallel size 1; scale out "
                "with --data-parallel-size instead."
            )
        # Read by the DeepGEMM MegaMoE warmup.
        self.use_sequence_parallel = False
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob

        ep_group = get_ep_group()
        self.n_routed_experts = config.num_experts
        self.n_shared_experts = 1
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        if self.n_physical_experts % ep_group.world_size:
            raise ValueError(
                f"{self.n_physical_experts} experts do not divide evenly over "
                f"{ep_group.world_size} EP ranks."
            )
        self.n_local_physical_experts = self.n_physical_experts // ep_group.world_size

        self.gate = GateLinear(
            config.hidden_size, config.num_experts, prefix=f"{prefix}.gate"
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            expert_gate=self.shared_expert_gate,
            prefix=f"{prefix}.shared_expert",
        )
        # The shared MLP overlaps the router on the aux stream and joins before
        # the MegaMoE launch, which needs every SM.
        self.shared_stream = (
            None if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM else aux_stream()
        )
        self.shared_events = (torch.cuda.Event(), torch.cuda.Event())
        checkpoint_format, weight_format = mega_moe_expert_formats(
            vllm_config, f"{prefix}.experts"
        )
        logger.info_once(
            "Qwen4Exp MegaMoE runs %s routed experts (checkpoint: %s).",
            weight_format.name,
            checkpoint_format,
        )
        self.experts = Qwen4ExpMegaMoEExperts(
            vllm_config,
            checkpoint_format=checkpoint_format,
            weight_format=weight_format,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=ep_group.rank_in_group * self.n_local_physical_experts,
            num_logical_experts=self.n_logical_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def route():
            router_logits, _ = self.gate(hidden_states)
            return fused_topk(
                hidden_states, router_logits, self.top_k, self.renormalize
            )

        (topk_weights, topk_ids, _), shared_out = maybe_execute_in_parallel(
            route,
            lambda: self.shared_expert(hidden_states),
            *self.shared_events,
            self.shared_stream,
        )
        y = self.experts(hidden_states, topk_weights, topk_ids)
        y += shared_out
        return y
