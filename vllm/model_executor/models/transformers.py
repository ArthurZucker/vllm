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
from typing import List, Optional, Union, Dict

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (get_pp_group)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, make_empty_intermediate_tensors_factory,
                    maybe_prefix)

from torch.nn.parameter import Parameter, UninitializedParameter


from transformers.models.auto import AutoModel
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


def vllm_flash_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        kv_caches: torch.Tensor=None,
        attn_metadata: AttentionMetadata=None,
        attention_interface=None,
        layer_idx=0,
        **kwargs
    ):
    hidden = query.shape[1]
    return attention_interface(query.reshape(hidden,-1), key.reshape(hidden,-1), value.reshape(hidden,-1), kv_cache=kv_caches[layer_idx],attn_metadata=attn_metadata)

modeling_flash_attention_utils._flash_attention_forward = vllm_flash_attention_forward
generic.KwargsForCausalLM = VllmKwargsForCausalLM

# TODO, LLAMA_ATTENTION_FUNCTION would be the best place to put this
# TODO, should we also add a VllmCacheClass? Because otherwise the DynamicCache initiated by default is used
# and we have to set `use_cache=False` while we are actually using it

class TransformersModel(nn.Module, SupportsLoRA, SupportsPP):
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *, vllm_config, prefix: str = ""
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config
        self.config = config
        self.lora_config = lora_config

        self.attention_interface = Attention(
            config.num_attention_heads,
            config.head_dim,
            config.head_dim**-0.5,
            num_kv_heads=config.num_key_value_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
        )
        config._attn_implementation_internal="flash_attention_2"


        self.model = self._init_model(vllm_config=config, prefix=prefix)
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.STEP,
            normalize=False,
            softmax=False)

    def _init_model(self, vllm_config, prefix: str = ""):
        model = AutoModel.from_config(vllm_config)
        # if get_pp_group().is_last_rank:
        #     # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        #     device_type = torch._C._get_accelerator().type
        #     device_module = torch.get_device_module(device_type)
        #     # Get device with index assuming equal number of devices per host
        #     tp_device = torch.device(device_type, torch.distributed.get_rank() % device_module.device_count())
        #     world_size = torch.distributed.get_world_size()
        #     device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))
        #     # Apply Tensor Parallelism
        #     model.tensor_parallel(device_mesh)
        return model
    


    def _autoset_attn_implementation(self, config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,):
        config._attn_implementation = "flash_attention_2"
        config._attn_implementation_autoset = True
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids[None,...], use_cache=False, position_ids=positions[None,...], kv_caches=kv_caches, attn_metadata=attn_metadata, intermediate_tensors=intermediate_tensors, attention_interface = self.attention_interface.forward, return_dict=False
        )[0][0,...] # we remove batch dimension for now
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata,
    ):
        logits = self.compute_logits(hidden_states, None)
        return self._pooler(logits, pooling_metadata)


    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        if isinstance(param, UninitializedParameter):
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = shape[output_dim] // self.tp_size
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)
        return param
    
    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:            
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # Materialize GGUF UninitializedParameter
            if isinstance(param, UninitializedParameter):
                param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)
            # Load the weight into the parameter
            try:
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            except Exception as e:
                print(f"Error loading weight for parameter '{name}': {e}")
                loaded_params.add(name)
        return loaded_params