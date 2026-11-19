# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import triton_weight
from vllm.model_executor.layers.quantization.online.requantization import (
    OnlineQuantMixin,
)
from vllm.model_executor.utils import set_weight_attrs


def _zero_padded(
    weight: torch.Tensor, shape: tuple[int, ...], segments: int
) -> torch.Tensor:
    """`weight` zero-padded to `shape`, each of its `segments` along dim 1 (w13's
    gate and up) padded on its own."""
    if tuple(weight.shape) == tuple(shape):
        return weight
    padded = weight.new_zeros(shape)
    for source, target in zip(weight.chunk(segments, 1), padded.chunk(segments, 1)):
        target[tuple(slice(0, n) for n in source.shape)] = source
    return padded


class OnlineMoEMethodBase(OnlineQuantMixin, FusedMoEMethodBase):
    """Base for MoE methods quantizing full-precision experts as they load.

    `create_quantized_weights` registers the quantized params, which the plan's
    quantize op (`checkpoint_quantizer`) fills from the merged `[E, N, K]` experts
    with `quantize_experts`, after the model's converter ops; a requantization
    source's dequantize op runs on each checkpoint expert before them.
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition
        for name, shape, dtype in self.quantized_params(
            num_experts, w13_rows, intermediate_size_per_partition, hidden_size
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
        if layer.w2_weight.shape[-1] != intermediate_size_per_partition:
            # packed along its TP-sharded dim: the loader shards the checkpoint's
            # full-precision w2 evenly, not by the param's (packed) width
            layer.w2_weight.shard_even = True

        # BIASES (for models like GPT-OSS that have biased MoE), loaded as they are
        if self.moe.has_bias:
            for name, shape in (
                ("w13_bias", (num_experts, w13_rows)),
                ("w2_bias", (num_experts, hidden_size)),
            ):
                bias = torch.nn.Parameter(
                    torch.zeros(shape, dtype=params_dtype), requires_grad=False
                )
                layer.register_parameter(name, bias)
                set_weight_attrs(bias, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.quantizes_on_load = True

    def quantized_params(
        self, num_experts: int, w13_rows: int, intermediate: int, hidden: int
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        """`(name, shape, dtype)` of each quantized param: `w13_weight` holds
        `[num_experts, w13_rows, hidden]` experts, `w2_weight`
        `[num_experts, hidden, intermediate]`, then their scales."""
        raise NotImplementedError

    def expert_scale_names(self, prefix: str) -> list[str]:
        """The scales `quantize_experts` returns for `{prefix}weight`, in order."""
        return [f"{prefix}weight_scale"]

    def quantize_experts(
        self, layer: torch.nn.Module, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """`(values, *scales)` of the full-precision `[E, N, K]` experts of `name`
        (`w13_weight` or `w2_weight`), zero-padded to the layer's sizes."""
        raise NotImplementedError

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        """The plan's quantize op of each expert weight, after the experts merge
        (and a requantization source's dequantize op on each expert before it)"""
        from vllm.model_executor.models.transformers.conversion import (
            VllmQuantizeQuantizer,
        )

        targets = {
            f"{prefix}weight": (
                partial(self._quantize_on_load, layer, f"{prefix}weight"),
                self.expert_scale_names(prefix),
                True,
            )
            for prefix in ("w13_", "w2_")
        }
        if self.requantization_source is not None:
            return self.requantizing(layer, targets)
        return VllmQuantizeQuantizer(targets)

    def _quantize_on_load(
        self, layer: torch.nn.Module, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        config = layer.moe_config
        num_experts, hidden = weight.shape[0], config.hidden_dim
        rows = config.intermediate_size_per_partition
        if name == "w13_weight":
            segments = self.moe.w13_num_shards
            shape = (num_experts, segments * rows, hidden)
        else:
            shape, segments = (num_experts, hidden, rows), 1
        return self.quantize_experts(layer, name, _zero_padded(weight, shape, segments))

    def quantized_weights(self, layer: torch.nn.Module) -> tuple[torch.Tensor, ...]:
        """`(w13, *w13 scales, w2, *w2 scales)`, as loaded."""
        return tuple(
            getattr(layer, name)
            for prefix in ("w13_", "w2_")
            for name in (f"{prefix}weight", *self.expert_scale_names(prefix))
        )

    def quantize_shared_experts(
        self, layer: torch.nn.Module, w13: torch.Tensor, w2: torch.Tensor
    ) -> None:
        """Quantize the fused shared experts (staged, dequantized after loading by
        `OnlineSharedExpertLoader`) into their slots, after the routed experts."""
        num_experts = layer.local_num_experts
        shared = slice(num_experts - w13.shape[0], num_experts)
        for prefix, weight in (("w13_", w13), ("w2_", w2)):
            name = f"{prefix}weight"
            quantized = self._quantize_on_load(layer, name, weight.to(layer.orig_dtype))
            for param_name, tensor in zip(
                (name, *self.expert_scale_names(prefix)), quantized
            ):
                slot = getattr(layer, param_name).data[shared]
                slot.copy_(tensor.reshape(slot.shape))

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @property
    def supports_eplb(self) -> bool:
        return True

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            triton_weight(layer, "w13_weight"),
            triton_weight(layer, "w2_weight"),
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            routing_sink=layer.routing_sink,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            triton_weight(layer, "w13_weight"),
            triton_weight(layer, "w2_weight"),
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
