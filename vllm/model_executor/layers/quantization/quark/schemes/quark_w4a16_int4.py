# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

import torch

from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.auto_awq import (
    _convert_awq_to_standard_format,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    canonicalize_quark_packed_int4,
    parse_w4a16_int4_weight_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
)
from vllm.model_executor.model_loader.weight_ops import PadToParam
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types


class QuarkW4A16Int4(QuarkScheme):
    """Quark packed INT4 weight-only linear scheme via MPLinearKernel."""

    supported_activation_quant_keys: list[QuantKey | None] = [None]
    supported_weight_quant_keys: list[QuantKey] = [
        kInt4Static,
        kInt4Static32,
        kInt4StaticAsym,
        kInt4Static32Asym,
    ]

    def __init__(
        self,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
        pack_method: str,
        weight_config: dict[str, Any],
    ):
        super().__init__(weight_quant_key, activation_quant_key)

        # group_size is read from the Quark config rather than weight_quant_key:
        # the int4 QuantKeys only encode group_size 32 or -1, but Quark int4
        # checkpoints commonly use group_size 128, which the QuantKey cannot
        # represent. Symmetry is unambiguous, so it comes from the key.
        self.group_size = parse_w4a16_int4_weight_config(weight_config)[0]
        self.pack_factor = 8
        self.pack_reorder = pack_method == "reorder"
        self.is_symmetric = weight_quant_key.symmetric
        self.quant_type = (
            scalar_types.uint4b8 if self.is_symmetric else scalar_types.uint4
        )
        self.kernel: MPLinearKernel | None = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        input_size = kwargs["input_size"]
        output_size = kwargs["output_size"]
        group_size = (
            self.group_size if self.group_size != -1 else input_size_per_partition
        )
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size. "
                f"input_size_per_partition={input_size_per_partition}, "
                f"group_size={group_size}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        packed_output_size_per_partition = math.ceil(
            output_size_per_partition / self.pack_factor
        )
        layer.output_size_per_partition = output_size_per_partition
        layer.packed_output_size_per_partition = packed_output_size_per_partition

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=group_size,
            zero_points=not self.is_symmetric,
        )
        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
        )
        num_groups = input_size_per_partition // group_size
        weight_zero_point = PackedvLLMParameter(
            data=torch.zeros(
                num_groups,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                packed_output_size_per_partition * self.pack_factor,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point" if not self.is_symmetric else None,
        )

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        """`weight_scale` has a column per packed output slot: this rank's
        scales load zero-padded to them (`PadScaleColumns`)"""
        from vllm.model_executor.models.transformers.conversion import (
            VllmLayoutQuantizer,
        )

        if layer.output_size_per_partition == layer.weight_scale.shape[1]:
            return None
        quantizer = VllmLayoutQuantizer(PadScaleColumns, ["weight_scale"])
        quantizer.load_before_model_ops = False  # pads the param's own columns
        return quantizer

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = canonicalize_quark_packed_int4(
            layer.weight.data,
            pack_reorder=self.pack_reorder,
            is_symmetric=self.is_symmetric,
            pack_factor=self.pack_factor,
        )
        if not self.is_symmetric:
            layer.weight_zero_point.data = canonicalize_quark_packed_int4(
                layer.weight_zero_point.data,
                pack_reorder=self.pack_reorder,
                is_symmetric=self.is_symmetric,
                pack_factor=self.pack_factor,
            )
        output_size = layer.output_size_per_partition
        packed_output_size = layer.packed_output_size_per_partition * self.pack_factor
        if output_size < packed_output_size:
            layer.weight_scale.data[:, output_size:].zero_()

        _convert_awq_to_standard_format(
            layer,
            "weight",
            "weight_zero_point" if not self.is_symmetric else None,
            self.quant_type.size_bits,
        )
        assert self.kernel is not None
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        assert self.kernel is not None
        return self.kernel.apply_weights(layer, x, bias)


class PadScaleColumns(PadToParam):
    """This rank's scales (after sharding and any fused-shard concat), zero
    padded to the param's columns: one per packed output slot"""

    def __init__(self, quantizer=None):
        self.quantizer = quantizer
