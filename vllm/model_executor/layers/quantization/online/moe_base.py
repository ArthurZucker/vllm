# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable, Iterator
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
    OnlineRequantizationMixin,
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


class OnlineMoEMethodBase(OnlineRequantizationMixin, FusedMoEMethodBase):
    """Base for MoE methods quantizing full-precision experts as they load.

    `create_quantized_weights` registers the quantized params, which the plan's
    quantize op (`checkpoint_quantizer`) fills from the merged `[E, N, K]` experts
    with `quantize_experts`, after the model's converter ops. A requantization
    source keeps its own params instead, quantized after loading.
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
        # Target conversion needs this metadata even when the source creates weights.
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.requantization_source is not None:
            # In case FSE is used, shared experts slots are initialized on RoutedExperts
            # online quant method, the source only holds the routed experts.
            self.create_requantization_source_weights(
                layer,
                num_experts - layer.expert_map_manager.num_fused_shared_experts,
                layer.moe_config.hidden_dim_unpadded,
                layer.moe_config.intermediate_size_per_partition_unpadded,
                params_dtype,
                **extra_weight_attrs,
            )
            layer.num_experts = num_experts
            return

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
        """The plan's quantize op of each expert weight, after the experts merge"""
        if self.requantization_source is not None:
            return self.source_checkpoint_quantizer(layer)
        from vllm.model_executor.models.transformers.conversion import (
            VllmQuantizeQuantizer,
        )

        return VllmQuantizeQuantizer(
            {
                f"{prefix}weight": (
                    partial(self._quantize_on_load, layer, f"{prefix}weight"),
                    self.expert_scale_names(prefix),
                    True,
                )
                for prefix in ("w13_", "w2_")
            }
        )

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
        """`(w13, *w13 scales, w2, *w2 scales)`: as loaded, or the requantization
        source's, quantized now."""
        if self.requantization_source is None:
            return tuple(
                getattr(layer, name)
                for prefix in ("w13_", "w2_")
                for name in (f"{prefix}weight", *self.expert_scale_names(prefix))
            )
        return self.quantize_weights(
            layer,
            lambda w13, w2: (
                *self.quantize_experts(layer, "w13_weight", w13),
                *self.quantize_experts(layer, "w2_weight", w2),
            ),
        )

    def iter_weights_for_quantization(
        self, layer: torch.nn.Module
    ) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor]]:
        """Yield `(experts, w13, w2)` for groups of experts to quantize.

        The source dequantizes in chunks to bound staging memory. Without a source,
        all routed experts are yielded at once. Fused shared experts come last.
        """
        num_shared = layer.expert_map_manager.num_fused_shared_experts
        num_routed = layer.local_num_experts - num_shared

        source = self.requantization_source
        if source is None:
            # These include the fused shared experts slots, never staged here.
            yield slice(0, num_routed), layer.w13_weight, layer.w2_weight
        else:
            chunks = source.dequantize_weight(layer)
            if isinstance(chunks, torch.Tensor):
                raise TypeError("MoE requantization requires w13 and w2 weights.")
            for experts, w13, w2 in chunks:
                yield (
                    experts,
                    *self._pad_requantized_weights(
                        layer, w13, w2, experts.stop - experts.start
                    ),
                )

        if num_shared > 0:
            loader = layer.shared_expert_online_loader
            shared_w13, shared_w2 = loader.dequantize_staging(layer)
            loader.release_staging(layer)
            yield (
                slice(num_routed, num_routed + num_shared),
                *self._pad_requantized_weights(
                    layer,
                    shared_w13.to(layer.orig_dtype),
                    shared_w2.to(layer.orig_dtype),
                    num_shared,
                ),
            )

    def quantize_weights(
        self,
        layer: torch.nn.Module,
        quantize: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """Quantize the layer's weights one group of experts at a time.

        Args:
            layer: Layer to quantize.
            quantize: Maps `(w13, w2)` of a group of experts to tensors, e.g.
                `(w13, w13_scale, w2, w2_scale)`.

        Returns:
            The `quantize` outputs for all local experts.

        """
        num_experts = layer.local_num_experts
        outputs: list[torch.Tensor] | None = None
        for experts, w13, w2 in self.iter_weights_for_quantization(layer):
            quantized = quantize(w13, w2)
            if outputs is None:
                if w13.shape[0] == num_experts:
                    return quantized
                outputs = [t.new_empty((num_experts, *t.shape[1:])) for t in quantized]
            for output, tensor in zip(outputs, quantized):
                output[experts] = tensor
        assert outputs is not None
        return tuple(outputs)

    def _pad_requantized_weights(
        self,
        layer: torch.nn.Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = layer.moe_config
        hidden_size = config.hidden_dim_unpadded
        intermediate_size = config.intermediate_size_per_partition_unpadded
        target_hidden_size = config.hidden_dim
        target_intermediate_size = config.intermediate_size_per_partition

        num_shards = self.moe.w13_num_shards

        if w13.shape != (num_experts, num_shards * intermediate_size, hidden_size):
            raise ValueError("Requantization requires unpadded source w13 weights.")
        if w2.shape != (num_experts, hidden_size, intermediate_size):
            raise ValueError("Requantization requires unpadded source w2 weights.")

        if (
            target_hidden_size < hidden_size
            or target_intermediate_size < intermediate_size
        ):
            raise ValueError("Requantization target dimensions cannot shrink weights.")

        # No padding required.
        if (hidden_size, intermediate_size) == (
            target_hidden_size,
            target_intermediate_size,
        ):
            return w13, w2

        if any(
            getattr(layer, name, None) is not None for name in ("w13_bias", "w2_bias")
        ):
            raise NotImplementedError(
                "Padded requantization with biases is unsupported."
            )

        def pad_moe_weights(
            w13: torch.Tensor, w2: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            padded_w13 = w13.new_zeros(
                (num_experts, num_shards * target_intermediate_size, target_hidden_size)
            )
            padded_w2 = w2.new_zeros(
                (num_experts, target_hidden_size, target_intermediate_size)
            )
            for shard in range(num_shards):
                source_start = shard * intermediate_size
                target_start = shard * target_intermediate_size
                padded_w13[
                    :, target_start : target_start + intermediate_size, :hidden_size
                ] = w13[:, source_start : source_start + intermediate_size, :]
            padded_w2[:, :hidden_size, :intermediate_size] = w2
            return padded_w13, padded_w2

        return pad_moe_weights(w13, w2)

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
