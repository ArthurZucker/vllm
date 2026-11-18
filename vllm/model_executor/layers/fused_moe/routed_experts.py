# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsSink,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.model_executor.utils import kernel_parameters

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts


logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


@PluggableLayer.register("routed_experts")
class RoutedExperts(PluggableLayer):
    """Container for routed expert weights and execution logic.

    This module owns the expert weight parameters (w13_weight, w2_weight, scales, etc.)
    and handles:
    - Loading checkpoint weights into parameters
    - Executing routed experts via quant_method.apply()
    """

    # Hash-routing table, owned by the gate.
    hash_indices_table: torch.Tensor | None = None

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str | None = "up_proj",
        is_fused_checkpoint_transposed: bool = False,
        #
        # Extra params that are needed by quant_methods, pass along for now
        # Prefer getting these from other sources, e.g. moe_config or
        # router object
        #
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        hash_indices_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        if moe_config.is_act_and_mul == (ckpt_up_proj_name is None):
            raise ValueError(
                f"{layer_name}: ckpt_up_proj_name={ckpt_up_proj_name!r} is "
                f"inconsistent with activation {moe_config.activation.value!r}. "
                "Gated MoE needs an up projection name; non-gated MoE must "
                "pass None."
            )
        self.is_fused_checkpoint_transposed = is_fused_checkpoint_transposed
        self.expert_map_manager = expert_map_manager
        self.hidden_size = moe_config.hidden_dim
        self.global_num_experts = moe_config.num_experts
        self.local_num_experts = moe_config.num_local_experts
        self.params_dtype = params_dtype

        # Register buffers for state_dict compatibility
        self.update_expert_map_info()

        self.rocm_aiter_fmoe_enabled = moe_config.rocm_aiter_fmoe_enabled

        # It would be good to eventually codify these in FusedMoEConfig
        # or some other config.
        self.top_k = self.moe_config.experts_per_token
        self.activation = self.moe_config.activation
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.e_score_correction_bias = e_score_correction_bias
        # Set by bind_routed_experts_capturer for monolithic kernels.
        self.routing_sink: RoutedExpertsSink | None = None
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # The gate owns this Parameter: keep it out of this module's parameters.
        object.__setattr__(self, "hash_indices_table", hash_indices_table)
        # End random parameters
        self._loaded_expert_biases: set[str] = set()

        self.quant_method = self._get_quant_method(
            self.layer_name,
            self.quant_config,
            self.moe_config,
        )

        # Only set when FSE go through online quantization, with pre-quantized routed.
        self.shared_expert_online_loader = None
        online_config = getattr(self.quant_config, "online_quantization_config", None)
        if (
            online_config is not None
            and self.expert_map_manager.num_fused_shared_experts > 0
        ):
            from vllm.model_executor.layers.quantization.online.shared_expert import (
                OnlineSharedExpertLoader,
            )

            self.shared_expert_online_loader = OnlineSharedExpertLoader(
                self.quant_config,
                online_config,
                self.quant_method,
                self.moe_config.shared_expert_prefix,
                self.moe_config.num_logical_experts,
                self.expert_map_manager.num_fused_shared_experts,
            )

        # Round up hidden size and update moe_config.
        # TODO: move roundup to _get_quant_method?
        self.hidden_size, self.intermediate_size_per_partition = (
            self.quant_method.maybe_roundup_sizes(
                self.hidden_size,
                self.moe_config.intermediate_size_per_partition,
                self.moe_config.in_dtype,
                self.moe_config.moe_parallel_config,
            )
        )
        self.moe_config.hidden_dim = self.hidden_size
        self.moe_config.intermediate_size_per_partition = (
            self.intermediate_size_per_partition
        )

        if (
            self.moe_config.moe_parallel_config.enable_eplb
            and not self.quant_method.supports_eplb
        ):
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}."
            )

        moe_quant_params: dict[str, Any] = {
            "num_experts": moe_config.num_local_experts,
            "hidden_size": self.hidden_size,
            "unpadded_hidden_size": self.moe_config.hidden_dim_unpadded,
            "intermediate_size_per_partition": (
                self.moe_config.intermediate_size_per_partition
            ),
            "params_dtype": params_dtype,
            "global_num_experts": moe_config.num_experts,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        loader = self.shared_expert_online_loader
        if loader is not None and loader.stages_shared_experts:
            loader.create_staging(self, params_dtype, weight_loader=None)

        self.lora_base_layer_prefix = ""
        self._flag_padding_for_reload()

    def _flag_padding_for_reload(self) -> None:
        """Flag the params a direct reload zero-fills whole at begin; decided
        from the sizes at construction, before post-load processing."""
        c = self.moe_config
        inter = c.intermediate_size_per_partition
        if not (
            c.hidden_dim > (c.hidden_dim_unpadded or c.hidden_dim)
            or inter > (c.intermediate_size_per_partition_unpadded or inter)
            # The loader refills this padding too, but a load plan skips loader fills.
            or c.tp_shard_with_padding
        ):
            return
        # Per-expert scalars ((E,), (E, <=2)) aren't padded.
        for p in self._parameters.values():
            if p is not None and p.ndim > 1 and p.numel() > 2 * p.shape[0]:
                p.fill_on_reload = True

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.quant_method = quant_method

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.global_num_experts = new_moe_config.num_experts
        # local experts?

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        """Helper method to ensure quant_method is never None and
        of the proper type.
        """
        quant_method = None
        if quant_config is not None:
            quant_method = resolve_quant_method(quant_config, self, prefix)
        if quant_method is None:
            quant_method = UnquantizedFusedMoEMethod(moe_config)
        assert isinstance(quant_method, FusedMoEMethodBase)
        return quant_method

    def _ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def use_ep(self) -> bool:
        return self.moe_config.moe_parallel_config.use_ep

    @property
    def expert_map(self) -> torch.Tensor | None:
        # AITER fused-MoE kernels consume the 0/1 expert_mask; every other
        # backend consumes the canonical -1/local-slot map. Ask the active
        # experts kernel which it wants (only AITER sets consumes_expert_mask)
        # rather than keying on the global VLLM_ROCM_USE_AITER switch, which
        # does not reflect the per-layer kernel actually selected.
        moe_kernel = getattr(self.quant_method, "moe_kernel", None)
        if moe_kernel is not None and moe_kernel.fused_experts.consumes_expert_mask:
            return self.expert_mask
        return self._expert_map

    def update_expert_map_info(self):
        # Update local attributes from ExpertMapManager
        self.local_num_experts = self.expert_map_manager.local_num_experts
        self.expert_placement_strategy = self.expert_map_manager.placement_strategy
        self.register_buffer(
            "_expert_map", self.expert_map_manager.expert_map, persistent=False
        )
        self.register_buffer(
            "expert_mask", self.expert_map_manager.expert_mask, persistent=False
        )

        # Get routing tables from ExpertMapManager
        routing_tables = self.expert_map_manager.routing_tables
        if routing_tables is not None:
            # Register routing tables as buffers for this layer
            global_to_physical, physical_to_global, local_global = routing_tables
            self.register_buffer(
                "expert_global_to_physical", global_to_physical, persistent=False
            )
            self.register_buffer(
                "expert_physical_to_global", physical_to_global, persistent=False
            )
            self.register_buffer(
                "expert_local_to_global", local_global, persistent=False
            )

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # Return cached routing tables if already registered as buffers
        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )
        return None

    def update_expert_map(self):
        # Update ExpertMapManager with new EP configuration
        # The moe_parallel_config (including ep_size and ep_rank)
        # should already be updated.
        # Note: ExpertMapManager.update() recalculates expert maps and
        # reinitializes routing tables internally.
        self.expert_map_manager.update(
            self.moe_config.moe_parallel_config,
            global_num_experts=self.global_num_experts,
        )

        # Update local attributes from ExpertMapManager
        self.update_expert_map_info()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.expert_map_manager.map_global_to_local(expert_id)

    #
    # Weight Loading Methods
    #

    def get_expert_mapping(
        self,
        ckpt_gate_proj_name: str | None = None,
        ckpt_down_proj_name: str | None = None,
        ckpt_up_proj_name: str | None = None,
        include_fused: bool = False,
    ) -> list[tuple[str, str, int, str]]:
        moe_config = self.moe_config
        num_fused_shared_experts = self.expert_map_manager.num_fused_shared_experts
        num_redundant_experts = moe_config.num_experts - moe_config.num_logical_experts
        return self.build_expert_params_mapping(
            ckpt_gate_proj_name or self.ckpt_gate_proj_name,
            ckpt_down_proj_name or self.ckpt_down_proj_name,
            ckpt_up_proj_name or self.ckpt_up_proj_name,
            num_experts=moe_config.num_logical_experts + num_fused_shared_experts,
            num_redundant_experts=num_redundant_experts,
            routed_experts_prefix="",
            lora_base_layer_prefix=self.lora_base_layer_prefix,
            include_fused=include_fused,
            is_gated=moe_config.is_act_and_mul,
        )

    @staticmethod
    def make_expert_params_mapping(
        model: torch.nn.Module,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str | None,
        num_experts: int,
        num_redundant_experts: int = 0,
        routed_experts_prefix: str = "routed_experts",
    ) -> list[tuple[str, str, int, str]]:
        """Build the expert mapping, detecting the LoRA `base_layer.` prefix by
        scanning `model`'s parameters.

        Legacy entry point for models that still hand-roll `load_weights`; the
        `RoutedExperts` weight loader uses `get_expert_mapping` /
        `build_expert_params_mapping` instead (which take the prefix directly).
        See `build_expert_params_mapping` for the returned tuple format.
        """
        has_base_layer = any(".base_layer." in n for n, _ in model.named_parameters())
        prefix = "base_layer." if has_base_layer else ""
        # These loaders index ``params_dict[full_name]``, so both sides get it.
        mapping = RoutedExperts.build_expert_params_mapping(
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts,
            routed_experts_prefix,
            lora_base_layer_prefix=prefix,
            lora_base_layer_prefix_on_param_name=prefix,
        )

        # All MoE layers of a model share the same FSE and quantization setup.
        routed_experts = next(
            (m for m in model.modules() if isinstance(m, RoutedExperts)), None
        )
        if routed_experts is None or routed_experts.shared_expert_online_loader is None:
            return mapping
        return routed_experts.shared_expert_online_loader.redirect_mapping(mapping)

    @staticmethod
    def build_expert_params_mapping(
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str | None,
        num_experts: int,
        num_redundant_experts: int = 0,
        routed_experts_prefix: str = "routed_experts",
        lora_base_layer_prefix: str = "",
        lora_base_layer_prefix_on_param_name: str = "",
        include_fused: bool = False,
        is_gated: bool = True,
    ) -> list[tuple[str, str, int, str]]:
        """Create expert parameter mapping for weight loading with redundant experts.

        This mapping handles the physical-to-logical expert ID conversion needed
        when loading weights with EPLB redundant experts.

        Args:
            ckpt_gate_proj_name: Name of gate projection in checkpoint
            ckpt_down_proj_name: Name of down projection in checkpoint
            ckpt_up_proj_name: Name of up projection in checkpoint
            num_experts: Number of logical (non-redundant) experts
            num_redundant_experts: Number of redundant experts
            lora_base_layer_prefix: LoRA `base_layer.` prefix for the
                `weight_name` (checkpoint) side
            lora_base_layer_prefix_on_param_name: same, for the `param_name`
                side. Independent because `get_expert_mapping` resolves
                `param_name` via `getattr` against this layer's bare
                `w13_weight`/`w2_weight` (no prefix), while
                `make_expert_params_mapping` indexes the model-wide
                `params_dict` (prefix included).
            include_fused: Prepend the fused pre-fused-checkpoint entries
            is_gated: Whether w13 holds gate and up (gated activation) or a
                single up projection. Selects the fused checkpoint layout.
            routed_experts_prefix: Prefix of the routed experts submodule

        Returns:
            List of tuples (param_name, weight_name, expert_id, shard_id)
            where:
            - param_name: Parameter name in the layer
            - weight_name: Weight name in checkpoint
            - expert_id: Physical expert ID
            - shard_id: Shard identifier (w1, w2, w3)

        """
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        if routed_experts_prefix != "":
            routed_experts_prefix = f"{routed_experts_prefix}."

        w13 = (
            f"experts.{lora_base_layer_prefix_on_param_name}{routed_experts_prefix}w13_"
        )
        w2 = f"experts.{lora_base_layer_prefix_on_param_name}{routed_experts_prefix}w2_"

        fused_mapping = []
        if include_fused:
            gate_up = None
            w13_shards: tuple[str, ...] = ("w1", "w3")
            if not is_gated:
                # Non-gated: the stacked checkpoint tensor is the up projection
                # itself, nothing to split into gate and up.
                gate_up = ckpt_gate_proj_name
                w13_shards = ("w1",)
            elif ckpt_gate_proj_name == "gate_proj" and ckpt_up_proj_name == "up_proj":
                gate_up = "gate_up_proj"
            elif ckpt_gate_proj_name == "w1" and ckpt_up_proj_name == "w3":
                gate_up = "w13"
            else:
                logger.warning(
                    "Unexpected gate/up projection names: %s, %s. "
                    "Fused gate/up mapping will be skipped.",
                    ckpt_gate_proj_name,
                    ckpt_up_proj_name,
                )
            if gate_up is not None:
                fused_mapping = [
                    # (param_name, weight_name, expert_id, shard_id)
                    # expert_id doubles as the chunk index when splitting w13.
                    *(
                        (f"{w13}weight", f"experts.{gate_up}", chunk, shard_id)
                        for chunk, shard_id in enumerate(w13_shards)
                    ),
                    (f"{w2}weight", f"experts.{ckpt_down_proj_name}", 0, "w2"),
                ]
            if gate_up is not None and is_gated:
                # Per-expert gate_up_proj tensors; the non-gated equivalent is
                # already covered by the per-expert mapping below.
                fused_mapping.extend(
                    (
                        w13,
                        (
                            f"experts.{physical_to_logical_map[expert_id]}."
                            f"{gate_up}.{lora_base_layer_prefix}"
                        ),
                        expert_id,
                        shard_id,
                    )
                    for expert_id in range(num_physical_experts)
                    for shard_id in ("w1", "w3")
                )

        per_expert_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                w13 if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name] else w2,
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{lora_base_layer_prefix}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
            if weight_name is not None
        ]

        return fused_mapping + per_expert_mapping

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        # Kernel views of the params: live ones only, under their runtime names
        # (an aliased checkpoint name such as `w13_weight_packed` is reported as
        # `w13_weight`, which the name filters below expect).
        weights = [
            (f"{prefix}.{name}" if prefix else name, p)
            for prefix, module in self.named_modules()
            for name, p in kernel_parameters(module).items()
        ]

        # EPLB must use shards when original weight storage has been released.
        # Use explicit metadata: some tensor subclasses lack untyped_storage().
        localized_freed = getattr(self, "localized_freed_full_weight_names", ())
        if localized_freed:
            weights = [(name, p) for name, p in weights if name not in localized_freed]

        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        # `w13_input_scale` and `w2_input_scale` are global per-tensor
        # activation scales shared across all experts (e.g. NVFP4).
        # They are broadcast views (stride 0) from .expand() and are
        # not actual expert weights, so exclude them from EPLB.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
            "w13_input_scale",
            "w2_input_scale",
            "hash_indices_table",
        }

        # Parameters of non-expert submodules that live inside runner (RoutedExperts).
        # These must be excluded from EPLB weight rearrangement.
        NON_EXPERT_PREFIXES = ()

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith(NON_EXPERT_PREFIXES)
            and name not in NON_EXPERT_WEIGHTS
        )

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith(NON_EXPERT_PREFIXES)
        ]

    #
    # Execution
    #

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """Execute routed experts using the quantization method's apply function.

        This is called by the runner after router selection (for modular kernels)
        quant_method.apply() which accesses the weights on this RoutedExperts
        instance.

        Args:
            x: Input tensor after any transforms
            topk_weights: Routing weights from router (for modular kernels)
            topk_ids: Selected expert IDs from router (for modular kernels)
            shared_experts: The shared experts (if any)
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Finalized routed states or a deferred-finalize output.

        """
        assert not self.quant_method.is_monolithic

        # Modular kernels use pre-computed routing
        return self.quant_method.apply(
            layer=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        x_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """Execute routed experts using the quantization method's apply function.

        This is called by the runner after router selection (for modular kernels)
        or with router logits (for monolithic kernels). It delegates to
        quant_method.apply() which accesses the weights on this RoutedExperts
        instance.

        Args:
            x: Input tensor after any transforms
            router_logits: Router logits (for monolithic kernels)
            input_ids: input ids for DeepSeek V4
            x_scale: Scales of `x` when the caller already quantized it (needs
                `quant_method.accepts_quantized_input`)

        Returns:
            Finalized routed states or a deferred-finalize output.

        """
        assert self.quant_method.is_monolithic
        extra = {} if x_scale is None else {"x_scale": x_scale}

        # Monolithic kernels handle routing internally
        return self.quant_method.apply_monolithic(
            layer=self,
            x=x,
            router_logits=router_logits,
            input_ids=input_ids,
            **extra,
        )

    @property
    def supports_precomputed_routing(self) -> bool:
        """Whether a monolithic kernel can take vLLM's routing instead."""
        moe_kernel = getattr(self.quant_method, "moe_kernel", None)
        return moe_kernel is not None and moe_kernel.supports_precomputed_routing()

    def forward_monolithic_routed(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """Run a monolithic kernel on vLLM's routing, where padding rows are -1
        and activate no expert, which the kernel's own routing cannot do."""
        moe_kernel = self.quant_method.moe_kernel
        assert moe_kernel is not None and moe_kernel.supports_precomputed_routing()
        return moe_kernel.apply_monolithic_routed(
            x,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids,
            activation=self.activation,
            global_num_experts=self.global_num_experts,
            routing_sink=self.routing_sink,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise AssertionError("Call forward_modular or forward_monolithic instead.")


# Mark the RoutedExperts weight_loader as supporting MoE-specific parameters
