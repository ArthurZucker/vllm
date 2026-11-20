# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.determinism.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization import resolve_quant_method
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.utils import (
    cuda_unquantized_gemm,
    dispatch_unquantized_gemm,
    maybe_pad_weight_avoid_cache_aliasing,
    maybe_use_skinny_gemm,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.

        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    supports_pre_processed_weights = True

    def __init__(self) -> None:
        config = get_current_vllm_config_or_none()
        linear_backend = (
            config.kernel_config.linear_backend if config is not None else "auto"
        )
        self._gemm_impl = dispatch_unquantized_gemm(linear_backend)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        # sum(output_partition_sizes) * input_size_per_partition.
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            # MLA's kv_b_proj (see `_cpu_skip_gemm_dispatch`): not
            # perf-critical, so skip packing and use a plain fallback.
            if getattr(layer, "_cpu_skip_gemm_dispatch", False):
                layer.cpu_linear = torch.nn.functional.linear
                return

            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=True)
        elif current_platform.is_rocm():
            layer.weight.data = maybe_pad_weight_avoid_cache_aliasing(layer.weight.data)
        elif self._gemm_impl is cuda_unquantized_gemm:
            maybe_use_skinny_gemm(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT and (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            return linear_batch_invariant(x, layer.weight, bias)
        return self._gemm_impl(layer, x, layer.weight, bias)


def apply_flattened(
    fn: Callable[[Any], torch.Tensor], x: torch.Tensor | QuantizedActivation
) -> torch.Tensor:
    """Call `fn` on `x` flattened to 2D, if that can be done without a copy.

    Linear methods and kernels may assume 2D activations, but callers such as
    the Transformers modeling backend pass `[batch, seq, hidden]` activations.
    """
    if isinstance(x, QuantizedActivation):
        if len(x.orig_shape) <= 2:
            return fn(x)
        return fn(x.flatten()).view(*x.orig_shape[:-1], -1)
    if x.ndim <= 2 or not x.is_contiguous():
        return fn(x)
    return fn(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)


class LinearBase(PluggableLayer):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: Prefix for parameter names.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, tensor parallelism will be disabled for this layer.

    """

    requires_unquantized_input = False

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.allow_fp8_block_shape_mismatch = False
        self.quant_method: QuantizeMethodBase
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := resolve_quant_method(quant_config, self, prefix=prefix):
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )

    def update_param_tp_status(self):
        # Single source of truth for a parameter's TP state. BasevLLMParameter
        # stamps self.tp_rank with the *global* rank in __init__; this reconciles
        # every child parameter to the *layer's* tp_rank/tp_size (which correctly
        # accounts for disable_tp -> replicated weights with tp_rank == 0).
        #
        # Must be re-run whenever parameters are (re-)created after construction,
        # e.g. after quant_method.process_weights_after_loading() swaps in fresh
        # Parameters. Otherwise a later load_weights()/weight-refit would narrow a
        # replicated weight at global_rank * shard_size and overflow.
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


# --8<-- [start:replicated_linear]
@PluggableLayer.register("replicated_linear")
class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.

    """

    # --8<-- [end:replicated_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                },
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        output = apply_flattened(partial(self.quant_method.apply, self, bias=bias), x)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


# --8<-- [start:column_parallel_linear]
@PluggableLayer.register("column_parallel_linear")
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
        tp_rank: Override the tensor-parallel rank used for sharding. Defaults to
            the global TP rank. Used to shard at a coarser granularity than one
            shard per rank (see `DCPGroupColumnParallelLinear`).
        tp_size: Override the tensor-parallel world size used for sharding.
            Defaults to the global TP world size.

    """

    # --8<-- [end:column_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        # Divide the weight matrix along the last dimension.
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        self._maybe_allow_fp8_block_shape_mismatch()
        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
        quant_config = getattr(self, "quant_config", None)
        weight_block = getattr(quant_config, "weight_block_size", None)
        if (
            weight_block is None
            or len(weight_block) < 1
            or len(self.output_partition_sizes) <= 1
        ):
            return

        try:
            block_n = int(weight_block[0])
        except (ValueError, TypeError):
            return

        if block_n <= 0:
            return

        if any(size % block_n != 0 for size in self.output_partition_sizes):
            self.allow_fp8_block_shape_mismatch = True
            logger.debug(
                "Allowing FP8 block shape mismatch for %s (block_n=%d, partitions=%s)",
                getattr(self, "prefix", "<unknown>"),
                block_n,
                self.output_partition_sizes,
            )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        apply = partial(self.quant_method.apply, self, bias=bias)
        output_parallel = apply_flattened(apply, input_)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class DCPGroupColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear whose weight is sharded across DCP groups.

    With Decode Context Parallelism (DCP) the KV cache is sharded across a DCP
    group, so MLA decode must attend the group's full head set. This layer shards
    its output across DCP *groups* (effective tp size `tp_size //
    dcp_world_size`) rather than across every rank, so each rank in a group
    holds the whole group's heads, letting decode skip the query all-gather.

    `forward` returns the group's full head set. `_local_view`
    extracts this rank's TP shard for prefill.
    """

    def __init__(self, *args, **kwargs):
        dcp_world_size = (
            get_current_vllm_config().parallel_config.decode_context_parallel_size
        )
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        self.group_size = max(dcp_world_size, 1)
        self.qrep_active = self.group_size > 1
        self.rank_in_group = rank % self.group_size
        super().__init__(
            *args,
            **kwargs,
            tp_rank=rank // self.group_size,
            tp_size=world_size // self.group_size,
        )

    def _local_view(self, out: torch.Tensor) -> torch.Tensor:
        """Slice this rank's tp head shard from a group-heads output.

        `out` is head-shaped, i.e. `(..., group_heads, head_dim)`.
        """
        if self.group_size == 1:
            return out
        n = out.shape[-2] // self.group_size
        start = self.rank_in_group * n
        return out[..., start : start + n, :].contiguous()


class StackedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear whose weight is several checkpoint tensors stacked
    along the output dimension, each addressed by a shard id.
    """

    shard_ids: tuple[str | int, ...]
    """Every shard this layer can receive."""

    def _get_shard_size_mapping(self, loaded_shard_id: Any) -> int | None:
        """Rows of this rank's parameter that belong to the shard."""
        raise NotImplementedError


class MergedColumnParallelLinear(StackedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, all weights matrix won't be sharded, this layer
                    will be treated as a "Replicated" MergedLinear.
        padding_size: Extra output rows on each TP rank, appended after the
            real output_sizes. These rows are zeroed and are not checkpoint shards.

    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        padding_size: int = 0,
    ):
        self.padding_size = padding_size
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.output_sizes = output_sizes
        if padding_size:
            self.output_sizes = output_sizes + [padding_size * self.tp_size]
        self.shard_ids = tuple(range(len(output_sizes)))

        assert all(output_size % self.tp_size == 0 for output_size in self.output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        if padding_size:
            for param in (self.weight, self.bias):
                if param is None:
                    continue
                param.data[-padding_size:].zero_()
                set_weight_attrs(
                    param, {"weight_loader_numel": param[:-padding_size].numel()}
                )
                # The loader zeroes the padding too, but a load plan skips loader
                # fills: direct reload zero-fills it at begin instead.
                param.fill_on_reload = True

    def _get_shard_size_mapping(self, loaded_shard_id: int) -> int:
        return self.output_sizes[loaded_shard_id] // self.tp_size


class QKVParallelLinear(StackedColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.

    """

    shard_ids: tuple[str, ...] = ("q", "k", "v")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
        output_size = sum(self.output_sizes)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)


class MinimaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """QKV projection fused with a lightning-indexer's index_q/index_k.

    NOTE: MiniMax-M3-specific. This is tailored to the M3 sparse-attention
    layers (it assumes the indexer's head count equals the KV head count and
    shares the main head_dim); it is not a general-purpose linear layer. It
    lives here only to sit alongside QKVParallelLinear, whose sharding /
    weight-loading machinery it reuses.

    A single column-parallel GEMM emits, per rank::

        [q | k | v | index_q | index_k]

    `index_q` must have the same head count as the KV heads
    (`total_num_index_heads == total_num_kv_heads`) and `index_head_size ==
    head_size`, so it shards exactly like K/V -- including the KV-head
    *replication* path when `tp_size > total_num_kv_heads` (this is what makes
    a TP size greater than the KV-head count work). `index_k` is a single
    shared head, replicated to every rank.

    `replicate_index_q` moves `index_q` onto that replicated path too, so
    every rank projects all the index heads rather than its own. That is what
    indexer context parallelism needs: it inverts the split, scoring every head
    against a shard of the blocks instead of one head against all of them.
    """

    shard_ids = ("q", "k", "v", "index_q", "index_k")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        replicate_index_q: bool = False,
    ) -> None:
        # index_q rides the KV-head sharding/replication path, so its head count
        # must match the KV heads.
        assert total_num_index_heads == total_num_kv_heads, (
            "MinimaxM3QKVParallelLinearWithIndexer requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size
        self.replicate_index_q = replicate_index_q

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        # index_q shards identically to the KV heads, unless CP has asked for
        # every head on every rank.
        self.num_index_heads = (
            self.total_num_index_heads if replicate_index_q else self.num_kv_heads
        )

        # Global per-group sizes (replicated groups counted x tp_size, matching
        # the QKVParallelLinear convention). index_k is a single replicated head.
        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        iq = self.num_index_heads * self.index_head_size
        ik = self.index_head_size
        self.output_sizes = [
            q * tp_size,  # q
            kv * tp_size,  # k
            kv * tp_size,  # v
            iq * tp_size,  # index_q
            ik * tp_size,  # index_k (replicated)
        ]

        # Skip QKVParallelLinear.__init__ (3-group layout); build the 5-group
        # column-parallel weight directly.
        StackedColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        return {
            "q": self.num_heads * h,
            "k": self.num_kv_heads * h,
            "v": self.num_kv_heads * h,
            "index_q": self.num_index_heads * h,
            "index_k": self.index_head_size,
        }.get(loaded_shard_id)


class KimiK3MergedQKVGateLinear(MergedColumnParallelLinear):
    """Kimi-K3 QKV-A projection fused with its output-gate projection.

    NOTE: Kimi-K3-specific. This is tailored to its replicated Q/KV latent
    projections and tensor-parallel output gate; it is not a general-purpose
    linear layer. It lives here alongside `MergedColumnParallelLinear`, whose
    sharding and weight-loading machinery it reuses.

    The per-rank shard layout is `[q_a | kv_a | gate]`. The latent shards
    stay replicated across tensor-parallel ranks while the gate shard is
    tensor-parallel, matching the standalone projections they replace.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        total_num_heads: int,
        v_head_dim: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()

        # MergedColumnParallelLinear divides every logical shard by TP size.
        # Inflate the replicated latent shards to retain their checkpoint widths.
        output_sizes = [
            q_lora_rank * tp_size,
            (kv_lora_rank + qk_rope_head_dim) * tp_size,
            total_num_heads * v_head_dim,
        ]
        super().__init__(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )


# --8<-- [start:row_parallel_linear]
@PluggableLayer.register("row_parallel_linear")
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.

    """

    # --8<-- [end:row_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        apply = partial(self.quant_method.apply, self, bias=bias_)
        output_parallel = apply_flattened(apply, input_parallel)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
