# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass(slots=True)
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # materialize on the CPU instead: online quantization of a layer that the
    # UVA offloader targets stages its full-precision weights there
    stage_on_cpu: bool = False

    # track how many elements are ready for loading, used by `reload_load_into`
    load_numel: int = 0
    load_numel_total: int | None = None
    defer_processing: bool = False

    # writes buffered by `reload_load_into` until the layer is complete:
    # (attribute name, name `load_with_ops` wrote, value)
    loaded_weights: list[tuple[str, str, torch.Tensor]] = field(default_factory=list)

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            stage_on_cpu=self.stage_on_cpu,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
