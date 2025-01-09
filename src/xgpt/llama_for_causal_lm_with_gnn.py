# llama_for_causal_lm_with_gnn.py

from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM
from .llama_model_with_gnn import LlamaModelWithGNN  # Ensure correct import path

import torch.nn as nn

import torch.nn.functional as F
from transformers.models.llama.modeling_llama import *
from transformers.generation import GenerationMixin
from transformers import GenerationMixin  # , CausalLMOutputWithPast
import torch.nn as nn
from transformers.models.llama.modeling_llama import *


class LlamaForCausalLMWithGNN(LlamaForCausalLM, PreTrainedModel, GenerationMixin,):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):

        # note: TODO try to remove super call for better memory management
        super().__init__(config)
        self.model = LlamaModelWithGNN(config,)
