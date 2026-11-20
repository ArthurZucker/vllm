# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import Cosmos3OmniConfig

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class Cosmos3ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Cosmos3OmniConfig:
        return self.ctx.get_hf_config(Cosmos3OmniConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Cosmos3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower
    # alongside a generation tower in a flat key layout. This mapper drops
    # the generation tower weights and rewrites the understanding tower keys
    # into the nested form expected by Qwen3VLForConditionalGeneration.

    # Cosmos3 unified Diffusers checkpoints store reasoner weights across
    # transformer/ and vision_encoder/. Match both while excluding VAE and
    # sound_tokenizer weights; the mapper drops generation-only tensors.
    allow_patterns_overrides = ["[tv]*er/*.safetensors"]
