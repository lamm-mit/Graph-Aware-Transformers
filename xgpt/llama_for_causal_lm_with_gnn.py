# llama_for_causal_lm_with_gnn.py

from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM
from .llama_model_with_gnn import LlamaModelWithGNN  # Ensure correct import path
from .gnn_config import GNNConfig
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import *
from .graph_neural_network import CausalGraphNeuralNetwork  
from transformers.generation import GenerationMixin
from transformers import GenerationMixin#, CausalLMOutputWithPast
import torch.nn as nn
from transformers.models.llama.modeling_llama import *
from transformers.processing_utils import Unpack

#from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
#from transformers.utils import LossKwargs

#class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class LlamaForCausalLMWithGNN(LlamaForCausalLM, PreTrainedModel, GenerationMixin,):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig ):

        #note: TODO try to remove super call for better memory management
        super().__init__(config)
        self.model = LlamaModelWithGNN(config,)

    '''
    ################################################################
    ### TO BE TESTED... THIS IS SO THAT INSPECTS FINDS THE ADDITIONAL_ADJ OPTION....
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        additional_adj: Optional[torch.Tensor] = None,
        **kwargs,#: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs  # Pass additional arguments as needed
        )
        #return super.forward (input_ids,attention_mask,position_ids,past_key_values,inputs_embeds,labels,use_cache,output_attentions,output_hidden_states,return_dict,
        #               cache_position,num_logits_to_keep,additional_adj,**kwargs)
    '''