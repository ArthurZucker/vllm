# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from contextlib import nullcontext
from weakref import WeakKeyDictionary, WeakSet

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import finalize_model_weights

from .meta import (
    SKIP_LOAD_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
)
from .types import LayerReloadingInfo
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
    has_device_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "max_layer_reload_bytes",
]


# Global dict storing information used for layerwise restoring, loading, and processing.
# For more information regarding what info is stored when, see `LayerReloadingInfo`
#
# Use a weak ref dictionary so that modules can be freed when the model is freed.
# Values are sanitized from references to the layer key in order to avoid circular refs
LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)

# Global set used to track loading for logging purposes only
LOADING_LAYERS: WeakSet[torch.nn.Module] = WeakSet()


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.get_default_device(),
        )

    return LAYERWISE_INFO[layer]


def record_metadata_for_reloading(model: torch.nn.Module):
    """Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        info.restore_metadata = capture_layer_to_meta(layer)
        info.restore_device = torch.get_default_device()


def max_layer_reload_bytes(model: torch.nn.Module) -> int:
    """Bytes of the largest layer in checkpoint format, which a layerwise reload
    materializes on the device, one layer at a time, before processing it."""
    sizes = (
        sum(t.nbytes for tensors in info.restore_metadata for t in tensors.values())
        for layer in model.modules()
        if (info := LAYERWISE_INFO.get(layer)) is not None
    )
    return max(sizes, default=0)


@torch.no_grad()
def initialize_layerwise_reload(
    model: torch.nn.Module, *, defer_processing: bool = False
):
    """Set up layerwise weight loading with deferred processing.

    Must be called after `record_metadata_for_reloading`. This function:
    1. Saves current kernel tensors for later copying
    2. Restores layer parameters/buffers from metadata (on meta device)
    3. Routes `load_with_ops`'s writes through `reload_load_into`, which buffers
       them until all weights of a layer are loaded

    A weight update is then `load_weights` again, with the same conversions and
    plan (quantize ops re-run). When all weights for a layer are loaded:
    1. Materialize the layer onto the target device
    2. Load all cached weights
    3. Run quantization processing if applicable
    4. Copy processed values back to original tensor storage

    With defer_processing=True, only finalize_layerwise_reload triggers processing.
    """
    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_load():
            continue

        # Save current tensors for later copying
        info.kernel_tensors = get_layer_params_buffers(layer)
        # snapshot now: restore_layer_on_meta drops alias buffers from the live set
        info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)

        # Restore layer parameters/buffers onto meta device
        restore_layer_on_meta(layer, info)

        # Buffer the writes into the layer until it is complete
        initialize_online_processing(layer)
        info.defer_processing = defer_processing


def initialize_online_processing(layer: torch.nn.Module):
    """Buffer the writes into `layer`'s tensors (see `reload_load_into`) until all
    of them are loaded, then process it. Called by `initialize_layerwise_reload`
    or an online quantization scheme; a no-op on a layer already buffering."""
    info = get_layerwise_info(layer)
    if info.can_load():
        return
    # Track loading progress to determine when to process/copy
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
    _track(layer)
    from vllm.model_executor.model_loader.ops_loader import install_load_into

    install_load_into(reload_load_into)


# The tensors whose writes a layerwise reload buffers: id -> (tensor, layer, name)
_TRACKED: dict[int, tuple[torch.Tensor, torch.nn.Module, str]] = {}


def _track(layer: torch.nn.Module) -> None:
    for name, tensor in get_layer_tensors(layer).items():
        if name not in SKIP_LOAD_TENSORS:
            _TRACKED[id(tensor)] = (tensor, layer, name)


def _untrack(layers: Iterable[torch.nn.Module]) -> None:
    """Stop buffering writes into `layers`; the last one uninstalls the writer"""
    layers = set(layers)
    for key in [k for k, (_, layer, _) in _TRACKED.items() if layer in layers]:
        del _TRACKED[key]
    if not _TRACKED:
        from vllm.model_executor.model_loader.ops_loader import uninstall_load_into

        uninstall_load_into(reload_load_into)


def reload_load_into(name: str, param: torch.Tensor, value: torch.Tensor) -> None:
    """The write `load_with_ops` uses during a layerwise reload: a write into a
    tracked (checkpoint-layout, on meta) tensor is buffered and counted; once its
    layer's loaded numel reaches the layer's size, the layer is materialized, the
    buffered writes replayed and the layer processed. Others are written as is."""
    from vllm.model_executor.model_loader.ops_loader import copy_into

    tracked = _TRACKED.get(id(param))
    if tracked is None or tracked[0] is not param:
        copy_into(name, param, value)
        return
    _, layer, attr = tracked
    info = get_layerwise_info(layer)
    if not info.can_load():
        # Some qconfigs load the same weight several times (CT_WNA16 loads
        # `weight_shape` for each qkv partition): extra writes after the layer
        # was processed are dropped. `weight_shape` is tiny, so the layer cannot
        # complete without its real weights. See Limitations(3).
        logger.debug("%s: Excessive loading", layer.__class__.__name__)
        return

    # layers may register tensors later (e.g. `bias`): track them, and refresh
    # `load_numel_total` so processing waits until they are loaded too
    info.load_numel_total = get_layer_size(layer)
    _track(layer)

    info.loaded_weights.append((attr, name, value))
    info.load_numel += get_numel_loaded(param, lambda: copy_into(name, param, value))

    logger.debug(
        "%s: %d / %d",
        layer.__class__.__name__,
        info.load_numel,
        info.load_numel_total,
    )

    # Do not online process attention layers, must wait until finalize
    # Preserve placement remapping until checkpoint iteration finishes:
    # expert reload must survive trailing redundant mappings.
    if info.defer_processing or is_deferred_attention_layer(layer):
        return

    # Log warnings allocating excessive buffers on device
    if has_device_tensors([value]) and layer not in LOADING_LAYERS:
        LOADING_LAYERS.add(layer)
        if len(LOADING_LAYERS) == 2:
            names = sorted([layer.__class__.__name__ for layer in LOADING_LAYERS])
            mem_used = sum(
                get_info_size(LAYERWISE_INFO[layer]) for layer in LOADING_LAYERS
            )
            logger.warning_once(
                "Allocating %.1f MB of device memory to buffers to load %s layers. "
                "This extra memory usage can be avoided by ordering weights "
                "by their parent layer when reloading.",
                mem_used / 1e6,
                str(list(names)),
            )

    # Process and copy when all weights are loaded
    if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
        _layerwise_process(layer, info)
        LOADING_LAYERS.discard(layer)


def finalize_layerwise_processing(model: torch.nn.Module, model_config: ModelConfig):
    """Apply processing to any layers which were not layerwise processed during loading.
    This includes attention layers and layers which have weight elements which are not
    loaded (due to padding).

    This function should be applied after `initialize_layerwise_reload`; it stops
    buffering writes into the model's layers.

    Args:
        model: model to finalize processing for
        model_config: config needed for applying processing to attention layers

    """
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []

    for layer in model.modules():
        info = get_layerwise_info(layer)
        if not info.can_load():
            info.reset()
            continue

        # Deferred attention-like layers are processed after all other layers
        if is_deferred_attention_layer(layer):
            deferred_attn.append((layer, info))
            continue

        # No weights were loaded
        if info.load_numel <= 0:
            # first load: checkpoint did not contain weights for this layer
            if info.kernel_tensors is None:
                _layerwise_process(layer, info)
                continue

            # reloading: place kernel tensors back as a fallback. Always place, even
            # when nothing is loadable (load_numel_total == 0), so parameter-alias
            # buffers on such layers are restored rather than left deleted.
            if info.load_numel_total > 0:  # type: ignore[operator]
                logger.warning("%s: Failed to load weights", layer.__class__.__name__)
            _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        elif info.defer_processing or info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info)

        info.reset()

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    LOADING_LAYERS.clear()
    _untrack(model.modules())


def finalize_layerwise_reload(model: torch.nn.Module, model_config: ModelConfig):
    finalize_layerwise_processing(model, model_config)
    finalize_model_weights(model)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    if info.kernel_tensors is None:
        if info.load_numel > 0:
            _layerwise_process(layer, info)
    elif info.load_numel > 0:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info)
    else:
        _place_kernel_tensors(layer, info)
    # its post-load hook runs from finalize_model_weights


def _reload_attention_scales(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None:
        # Re-create scale Parameters with sentinel values so unloaded scales
        # are correctly detected by process_weights_after_loading
        quant_method.create_weights(layer)

    _replay(layer, info)

    if quant_method is not None:
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing_context = (
            online_quantization_timing_context(layer, quant_method)
            if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        with timing_context:
            quant_method.process_weights_after_loading(layer)

    _copy_and_restore_kernel_tensors(layer, info)


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Finalize layer loading after all weights have been buffered.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all buffered weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer tensors onto device
    materialize_layer(layer, info)
    # Load all buffered weights into the materialized layer
    _replay(layer, info)
    _process_loaded_layer(layer, info)
    logger.debug("%s: Processed", layer.__class__.__name__)


def _process_loaded_layer(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Finish a layer whose weights are all loaded: run its quant method's
    processing, copy the result into the persistent kernel storage and reset
    its reload info.

    Shared by `_layerwise_process` and the sharded-RDT engine's quant stream,
    which loads a layer's weights by scattering into it instead of through the
    buffered writes but finishes it exactly the same way.
    """
    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Process weights (quantization, repacking, etc.)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        timing_context = (
            online_quantization_timing_context(layer, quant_method)
            if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
            else nullcontext()
        )
        with timing_context:
            quant_method.process_weights_after_loading(layer)
        # Re-reconcile parameter TP state: process_weights_after_loading may
        # have re-created Parameters (stamped with the global rank), which would
        # otherwise break replicated (disable_tp) weights on a subsequent reload.
        if hasattr(layer, "update_param_tp_status"):
            layer.update_param_tp_status()

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    if info.kernel_tensors is not None:
        _copy_and_restore_kernel_tensors(layer, info)

    info.reset()


def _replay(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Write the buffered values into the layer's current (materialized) tensors"""
    from vllm.model_executor.model_loader.ops_loader import copy_into

    for attr, name, value in info.loaded_weights:
        copy_into(name, getattr(layer, attr), value)


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Copy processed values into original kernel tensor storage and restore
    kernel tensor references on the layer. Preserves cudagraph references."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    loaded_tensor_names = {attr for attr, _, _ in info.loaded_weights}
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        if name in non_persistent and name not in loaded_tensor_names:
            continue
        buffer.data.copy_(getattr(layer, name))

    _place_kernel_tensors(layer, info)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer, persistent=name not in non_persistent)
