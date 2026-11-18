# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    # WeightsMapper applies all matching patterns sequentially (no break on first
    # match). This is safe here because every pattern is anchored at both ends
    # (\A...\Z) and after substitution the resulting key always starts with
    # "model." or "lm_head.", so no later pattern can accidentally match again.
    pass

    # Bypass super().load_weights() and construct AutoWeightsLoader(self)
    # directly (same pattern as Qwen2ForCausalLM). Any logic in the parent
    # class's load_weights is a thin wrapper around AutoWeightsLoader, and
    # we must apply hf_to_vllm_mapper before the loader walks the tree.
