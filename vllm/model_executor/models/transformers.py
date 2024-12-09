# coding=utf-8
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

"""Wrapper around `transformers` models"""
from typing import List, Optional, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from transformers import AutoModel
from transformers.utils import generic  
from transformers import modeling_flash_attention_utils
from typing import TypedDict


class VllmKwargsForCausalLM(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        kv_cache

        maxattn_metadata_length
    """
    kv_cache: torch.Tensor
    attn_metadata: AttentionMetadata


def vllm_flash_attn_varlen_func(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_interface
    ):
    return attention_interface.attn(query, key, value, kv_cache, attn_metadata=attn_metadata)

modeling_flash_attention_utils.flash_attn_varlen_func = vllm_flash_attn_varlen_func
generic.KwargsForCausalLM = VllmKwargsForCausalLM


class TransformerModel(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config
        self.attention_interface = Attention(
            config.num_heads_per_partition,
            config.head_dim,
            config.scale,
            num_kv_heads=config.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        self.model = AutoModel.from_config(config)
        if get_pp_group().is_last_rank:
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, attention_interface = self.attention_interface
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)
