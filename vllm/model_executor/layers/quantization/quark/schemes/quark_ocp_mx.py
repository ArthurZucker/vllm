# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction

import torch
from transformers.core_model_loading import ConversionOps

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MxFp4LinearKernel,
    MxFp6LinearKernel,
    Mxfp8LinearKernel,
    init_mxfp4_linear_kernel,
    init_mxfp6_linear_kernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    expose_input_quant_key,
)
from vllm.model_executor.layers.quantization.input_hadamard import (
    InputHadamardFusable,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_KEY_MAP,
    _WEIGHT_QUANT_KEY_MAP,
    OCP_MX_BLOCK_SIZE,
    OCP_MX_UNPACKED_DTYPES,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp6E2M3Static,
    kMxfp6E3M2Static,
    kMxfp8Static,
)
from vllm.model_executor.model_loader.weight_ops import _each
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


class QuarkOCP_MX(QuarkScheme, InputHadamardFusable):
    ocp_mx_linear: MxFp6LinearKernel | MxFp4LinearKernel | Mxfp8LinearKernel
    supported_activation_quant_keys = [*_ACTIVATION_QUANT_KEY_MAP.values(), None]
    supported_weight_quant_keys = [*_WEIGHT_QUANT_KEY_MAP.values()]

    def __init__(
        self,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
        scale_block_rows: int = 1,
    ):
        super().__init__(weight_quant_key, activation_quant_key)
        self.scale_block_rows = scale_block_rows
        self.weight_dtype = next(
            dtype
            for dtype, quant_key in _WEIGHT_QUANT_KEY_MAP.items()
            if quant_key == weight_quant_key
        )
        self.input_dtype = (
            next(
                dtype
                for dtype, quant_key in _ACTIVATION_QUANT_KEY_MAP.items()
                if quant_key == activation_quant_key
            )
            if activation_quant_key is not None
            else None
        )

        self.is_unpacked = self.weight_dtype in OCP_MX_UNPACKED_DTYPES

        if self.is_unpacked:
            self.packed_factor: int | Fraction = 1
        elif self.weight_dtype == "mxfp4":
            self.packed_factor = 2
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)

        if self.is_unpacked:
            # MXFP8 runs on the shared kernel abstraction, which picks a native
            # backend where one exists and emulates otherwise. The MXFP4/MXFP6
            # warnings below do not apply.
            return

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if current_platform.supports_mx() and (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype in OCP_MX_UNPACKED_DTYPES:
            return dim
        elif quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    def checkpoint_quantizer(self, layer: torch.nn.Module):
        """A `block_size=[R, 32]` export's scales (one e8m0 value per R weight
        rows) load expanded to one per row (`ExpandScaleRows`); at `R == 1` the
        layouts coincide"""
        if self.scale_block_rows == 1:
            return None
        from vllm.model_executor.models.transformers.conversion import (
            VllmLayoutQuantizer,
        )

        return VllmLayoutQuantizer(ExpandScaleRows, ["weight_scale"])

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.ocp_mx_linear.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        if input_size_per_partition % OCP_MX_BLOCK_SIZE != 0:
            layer_name = getattr(layer, "prefix", "") or type(layer).__name__
            raise ValueError(
                f"OCP MX linear layer {layer_name!r} has an input size per "
                f"partition of {input_size_per_partition}, which must be "
                f"divisible by the OCP MX group size {OCP_MX_BLOCK_SIZE}. "
                "Choose a compatible tensor-parallel size or avoid "
                "tensor-parallel sharding for this layer."
            )

        if self.is_unpacked:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = output_size_per_partition
            layer.params_dtype = params_dtype

            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=MXFP8_VALUE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
            )
            layer.register_parameter("weight", weight)

            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=MXFP8_SCALE_DTYPE,
                ),
                input_dim=1,
                output_dim=0,
            )
            layer.register_parameter("weight_scale", weight_scale)
            layer.weight_block_size = [1, OCP_MX_BLOCK_SIZE]
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes

            # WEIGHT
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.packed_factor,
            )
            layer.register_parameter("weight", weight)

            # WEIGHT SCALE
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
            )
            layer.register_parameter("weight_scale", weight_scale)

        if self.weight_quant_key == kMxfp4Static:
            self.ocp_mx_linear = init_mxfp4_linear_kernel(
                activation_quant_key=self.activation_quant_key,
            )
        elif self.weight_quant_key in [kMxfp6E2M3Static, kMxfp6E3M2Static]:
            self.ocp_mx_linear = init_mxfp6_linear_kernel(
                weight_quant_key=self.weight_quant_key,
                activation_quant_key=self.activation_quant_key,
            )
        elif self.weight_quant_key == kMxfp8Static:
            self.ocp_mx_linear = init_mxfp8_linear_kernel(
                weight_shape=layer.weight.shape,
                activation_quant_key=self.activation_quant_key,
            )
            if current_platform.is_rocm():
                expose_input_quant_key(layer, self.ocp_mx_linear)
        else:
            raise NotImplementedError(
                "No OCP MX linear kernel for weight_quant_key="
                f"{self.weight_quant_key}. Something is wrong, please open an "
                "issue."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.ocp_mx_linear.apply_weights(layer, x, bias)

    def fuse_input_hadamard(self, layer: torch.nn.Module, size: int) -> bool:
        if isinstance(self.ocp_mx_linear, MxFp4LinearKernel):
            return self.ocp_mx_linear.fuse_input_hadamard(size)
        return False


class ExpandScaleRows(ConversionOps):
    """A 2-D block scale (one e8m0 value per `scale_block_rows` weight rows)
    repeated to one row per weight row, as the kernels read it. Whole: a rank's
    rows are sharded after (its slice of the compressed scale is not the
    param's). `hf_quantizer` keeps the tensor's own dtype: the e8m0 bytes are
    viewed, not cast"""

    needs_full = True

    def __init__(self, quantizer=None):
        self.hf_quantizer = quantizer

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        name = kwargs["full_layer_name"]
        layer = kwargs["model"].get_submodule(name.rpartition(".")[0])
        rows = layer.scheme.scale_block_rows
        return _each(
            input_dict, lambda t: t.view(torch.uint8).repeat_interleave(rows, dim=0)
        )
