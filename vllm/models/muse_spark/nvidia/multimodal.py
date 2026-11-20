# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark Omni for NVIDIA GPUs: the shared Omni towers over the CUDA text stack."""

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.common.omni import (
    MuseSparkOmniMixin,
    register_omni_processor,
)

from .model import MuseSparkForCausalLM


@register_omni_processor
class MuseSparkForConditionalGeneration(
    MuseSparkOmniMixin, MuseSparkForCausalLM, SupportsMultiModal
):
    """Muse Spark Omni: text stack plus NaViT vision and waveform audio."""


__all__ = ["MuseSparkForConditionalGeneration"]
