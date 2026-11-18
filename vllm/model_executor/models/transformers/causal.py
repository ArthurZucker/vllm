# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers modeling backend mixin for causal language models."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import checkpoint_shapes
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class CausalMixin(VllmModelForTextGeneration, Base):
    embedding_modules = {"lm_head": "output_embeddings"}

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        tie_word_embeddings = self._get_tie_word_embeddings()

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                bias=bool(checkpoint_shapes(r"(^|\.)lm_head\.bias$")),  # Phi
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if tie_word_embeddings:
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size,
                scale=getattr(self.text_config, "logit_scale", 1.0),
                soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
            )
        else:
            self.lm_head = PPMissingLayer()

    def compute_logits_local(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """Logits for this TP rank's vocab shard, for batch-sharded sampling."""
        bias = self.lm_head.bias
        return self.logits_processor(
            self.lm_head, hidden_states, bias, skip_gather=True
        )

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
