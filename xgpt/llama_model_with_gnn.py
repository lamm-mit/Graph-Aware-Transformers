# llama_model_with_gnn.py

from transformers import LlamaConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, BaseModelOutputWithPast
from .llama_decoder_layer_with_gnn import LlamaDecoderLayerWithGNN  # Ensure correct import path
from .gnn_config import GNNConfig
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import *
from transformers.processing_utils import *
from transformers import LlamaConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, BaseModelOutputWithPast
from .llama_decoder_layer_with_gnn import LlamaDecoderLayerWithGNN  # Ensure correct import path
from .gnn_config import GNNConfig
import torch
import torch.nn as nn
 
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
#from transformers.modeling_flash_attention_utils import FlashAttentionKwargs#, _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
   # LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class LlamaModelWithGNN(LlamaModel,LlamaPreTrainedModel):

    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args: 
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig,  ):
        super().__init__(config)
        self.config = config
        
        # Ensure `config.gnn_config` is a GNNConfig instance
        if isinstance(config.gnn_config, dict):
            config.gnn_config = GNNConfig(**config.gnn_config)

        self.gnn_config = config.gnn_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        '''
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        '''
        self.layers = nn.ModuleList([
            LlamaDecoderLayerWithGNN(config, layer_idx, )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
 