import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from typing import List, Optional, Tuple, Union

from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaRMSNorm,
    Cache,
    LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)
from transformers.models.llama.modeling_llama import *

from torch_geometric.data import Batch, Data
from .gnn_config import GNNConfig
from .graph_neural_network import CausalGraphNeuralNetwork

#from .AttentionPerceiver import *
from .Attention_GNN import *

from typing import Optional, Tuple
import matplotlib.pyplot as plt

class AggregatedLearnableAdjacencyTransformer(nn.Module):
    def __init__(self, num_heads, adj_transform_hidden_dim, activation='sigmoid'):
        super(AggregatedLearnableAdjacencyTransformer, self).__init__()
        self.num_heads = num_heads
        if activation not in ['sigmoid', 'softmax']:
            raise ValueError("Activation must be either 'sigmoid' or 'softmax'")
        self.activation = activation

        # Define the MLP without the final activation
        self.mlp = nn.Sequential(
            nn.Linear(num_heads, adj_transform_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(adj_transform_hidden_dim, 1)
        )

    def forward(self, attn_weights):
        batch_size, num_heads, seq_len, _ = attn_weights.size()
        attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous()
        attn_flat = attn_weights.view(batch_size * seq_len * seq_len, self.num_heads)
        adj_flat = self.mlp(attn_flat)  # [batch_size * seq_len * seq_len, 1]
        adj = adj_flat.view(batch_size, seq_len, seq_len)

        if self.activation == 'sigmoid':
            adj = torch.sigmoid(adj)
        elif self.activation == 'softmax':
            adj = F.softmax(adj, dim=-1)

        return adj


class LlamaAttention_Original(nn.Module):  
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        print ("q_proj size: ", self.num_heads * self.head_dim)
        print ("k/v_proj size: ", self.num_key_value_heads * self.head_dim)
        print ("num_key_value_groups: ", self.num_key_value_groups)
        print ("hidden size in/out: ", self.hidden_size)
        
        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # Plot initial adjacency matrix if debugging is enabled
        if self.config.gnn_config.plot_for_debugging:
            print ('attn_weights', attn_weights.shape)
            head_adj_mean = attn_weights.mean(dim=1).cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx})")
            plt.tight_layout()
            plt.show()

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
########################### END REGULAR ATTENTION ########################


########################### MULTILAYER GCN ########################
class LlamaAttentionMultiLayer(nn.Module):
    """
    Multi-layer multi-headed attention with shared key/query projections and unique value/output projections per layer.
    Maintains feature parity with original LlamaAttention while supporting multiple attention layers.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Args:
            config (LlamaConfig): Model configuration.
            layer_idx (Optional[int]): Index of the layer (used for caching and debugging).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        #self.o_proj_last_only = self.config.gnn_config.GNN_from_attention_layers_o_proj_last_only
        self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Attention parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

        # Shared projections for key and query
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        # Per-layer value projections
        self.v_proj_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias) 
             for _ in range(self.num_attention_layers)]
        )

        self.o_proj_layers = nn.ModuleList(
            [nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
             for _ in range(self.num_attention_layers)]
        )

        # Output projection - single layer (keep here as reference)
        #self.o_proj_last = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary embeddings for positional encoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _split_tensor_for_tp(self, tensor: torch.Tensor, dim: int) -> List[torch.Tensor]:
        """Helper method to split tensors for tensor parallelism"""
        if self.pretraining_tp > 1:
            return tensor.split(tensor.size(dim) // self.pretraining_tp, dim=dim)
        return [tensor]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass incorporating all features from original LlamaAttention.
        """
        bsz, q_len, _ = hidden_states.size()
        last_attn_weights = None
        if self.pretraining_tp > 1:
            # Split projections for tensor parallelism
            q_slices = self._split_tensor_for_tp(self.q_proj.weight, 0)
            k_slices = self._split_tensor_for_tp(self.k_proj.weight, 0)
            #v_slices = self._split_tensor_for_tp(self.v_proj_layers[layer_idx].weight, 0)

            # Apply split projections
            query_states = torch.cat([F.linear(hidden_states, q_slice) for q_slice in q_slices], dim=-1)
            key_states = torch.cat([F.linear(hidden_states, k_slice) for k_slice in k_slices], dim=-1)
            #value_states = torch.cat([F.linear(hidden_states, v_slice) for v_slice in v_slices], dim=-1)
        else:
            # Standard projections without tensor parallelism
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            #value_states = self.v_proj_layers[layer_idx](hidden_states)

        # Reshape states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # Apply rotary positional embeddings
        if position_embeddings is None:
            if self.layer_idx == 0:  # Only warn once
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Repeat key-value states for groups
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        
        for layer_idx in range(self.num_attention_layers):
            #print ("hidden_states", hidden_states.shape)

            # Handle tensor parallelism for projections
            if self.pretraining_tp > 1:
                v_slices = self._split_tensor_for_tp(self.v_proj_layers[layer_idx].weight, 0)

                # Apply split projections

                value_states = torch.cat([F.linear(hidden_states, v_slice) for v_slice in v_slices], dim=-1)
            else:
                # Standard projections without tensor parallelism
                value_states = self.v_proj_layers[layer_idx](hidden_states)

            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Handle past key-value cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Compute attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # Plot initial adjacency matrix if debugging is enabled
            if self.config.gnn_config.plot_for_debugging:
                head_adj_mean = attn_weights.mean(dim=1).cpu().detach().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
                plt.colorbar(label="Attention Weight")
                plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx}, GCN layer {layer_idx})")
                plt.tight_layout()
                plt.show()

            # compute updated hidden_states
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            if self.pretraining_tp > 1:
                o_proj_slices = self._split_tensor_for_tp(self.o_proj_layers[layer_idx].weight, 1)
                attn_outputs = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
                attn_output = sum([F.linear(attn_outputs[i], o_proj_slices[i]) 
                                    for i in range(self.pretraining_tp)])
            else:
                attn_output = self.o_proj_layers[layer_idx](attn_output)
                

            hidden_states = attn_output 

        # Keep track of attention weights if needed
        if output_attentions:
            last_attn_weights = attn_weights

        if not output_attentions:
            last_attn_weights = None

        return hidden_states, last_attn_weights, past_key_value
########################### END MULTILAYER GCN ########################


########################### MULTILAYER GCN with Scaling ########################
class LlamaAttentionMultiLayer_with_Scaling(nn.Module):
    """
    Multi-layer multi-headed attention with shared key/query projections and unique value/output projections per layer.
    Maintains feature parity with original LlamaAttention while supporting multiple attention layers.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Args:
            config (LlamaConfig): Model configuration.
            layer_idx (Optional[int]): Index of the layer (used for caching and debugging).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        #self.o_proj_last_only = self.config.gnn_config.GNN_from_attention_layers_o_proj_last_only
        self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Attention parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

        # Shared projections for key and query
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        # Per-layer value projections
        self.v_proj_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias) 
             for _ in range(self.num_attention_layers)]
        )

        self.o_proj_layers = nn.ModuleList(
            [nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
             for _ in range(self.num_attention_layers)]
        )

        #scaling parameters, similar to GIN 
        #self.epsilon = nn.Parameter(torch.zeros(self.num_attention_layers)) 
        self.epsilon = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0) if i == 0 else torch.tensor(0.01)) for i in range(self.num_attention_layers)
        ])

        # Output projection - single layer (keep here as reference)
        #self.o_proj_last = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary embeddings for positional encoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _split_tensor_for_tp(self, tensor: torch.Tensor, dim: int) -> List[torch.Tensor]:
        """Helper method to split tensors for tensor parallelism"""
        if self.pretraining_tp > 1:
            return tensor.split(tensor.size(dim) // self.pretraining_tp, dim=dim)
        return [tensor]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass incorporating all features from original LlamaAttention.
        """
        bsz, q_len, _ = hidden_states.size()
        last_attn_weights = None
        if self.pretraining_tp > 1:
            # Split projections for tensor parallelism
            q_slices = self._split_tensor_for_tp(self.q_proj.weight, 0)
            k_slices = self._split_tensor_for_tp(self.k_proj.weight, 0)
            #v_slices = self._split_tensor_for_tp(self.v_proj_layers[layer_idx].weight, 0)

            # Apply split projections
            query_states = torch.cat([F.linear(hidden_states, q_slice) for q_slice in q_slices], dim=-1)
            key_states = torch.cat([F.linear(hidden_states, k_slice) for k_slice in k_slices], dim=-1)
            #value_states = torch.cat([F.linear(hidden_states, v_slice) for v_slice in v_slices], dim=-1)
        else:
            # Standard projections without tensor parallelism
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            #value_states = self.v_proj_layers[layer_idx](hidden_states)

        # Reshape states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # Apply rotary positional embeddings
        if position_embeddings is None:
            if self.layer_idx == 0:  # Only warn once
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # Repeat key-value states for groups
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        
        for layer_idx in range(self.num_attention_layers):
            #print ("hidden_states", hidden_states.shape)

            # Handle tensor parallelism for projections
            if self.pretraining_tp > 1:
                v_slices = self._split_tensor_for_tp(self.v_proj_layers[layer_idx].weight, 0)

                # Apply split projections

                value_states = torch.cat([F.linear(hidden_states, v_slice) for v_slice in v_slices], dim=-1)
            else:
                # Standard projections without tensor parallelism
                value_states = self.v_proj_layers[layer_idx](hidden_states)

            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Handle past key-value cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Compute attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # Plot initial adjacency matrix if debugging is enabled
            if self.config.gnn_config.plot_for_debugging:
                head_adj_mean = attn_weights.mean(dim=1).cpu().detach().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
                plt.colorbar(label="Attention Weight")
                plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx}, GCN layer {layer_idx})")
                plt.tight_layout()
                plt.show()

            # compute updated hidden_states
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            if self.pretraining_tp > 1:
                o_proj_slices = self._split_tensor_for_tp(self.o_proj_layers[layer_idx].weight, 1)
                attn_outputs = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
                attn_output = sum([F.linear(attn_outputs[i], o_proj_slices[i]) 
                                    for i in range(self.pretraining_tp)])
            else:
                attn_output = self.o_proj_layers[layer_idx](attn_output)
                

            #scaling=torch.signmoid (self.epsilon[layer_idx])
            #scaling=self.epsilon[layer_idx]
            scaling = torch.clamp(self.epsilon[layer_idx], 0.0, 1.0)

            hidden_states = attn_output*scaling+ (1.-scaling)*hidden_states
            
            #hidden_states = attn_output 


        # Keep track of attention weights if needed
        if output_attentions:
            last_attn_weights = attn_weights

        if not output_attentions:
            last_attn_weights = None

        return hidden_states, last_attn_weights, past_key_value
########################### END MULTILAYER GCN with SCALING ########################
 






#BEST ONE Nov 20  -- the one to be used
########################### MULTILAYER GIN with SCALING ########################



#BEST ONE Nov 20  -- the one to be used
########################### MULTILAYER GIN with SCALING ########################
#>> EDITED FOR EXPERMENT

def FlowThreshold(x, threshold, sharpness_1=5.0, sharpness_2=50.0):
    """
    Smooth custom activation function:
    - Zero below 0.
    - Smoothly rises from 0 at x=0 to 1 at x >= threshold.
    
    Args:
        x (torch.Tensor): Input tensor.
        threshold (float): Threshold above which the output is 1.
        sharpness_1 (float): Controls steepness near x = 0.
        sharpness_2 (float): Controls steepness near the threshold.

    Returns:
        torch.Tensor: Transformed output.
    """
    # Sigmoid for transition near x=0
    sigmoid_0 = F.sigmoid(sharpness_1 * x)
    
    # Sigmoid for transition near threshold
    sigmoid_threshold = F.sigmoid(sharpness_2 * (x - threshold))
    
    # Combine the two, ensuring output is 0 below 0
    smooth_output = sigmoid_0 * sigmoid_threshold
    
    return smooth_output

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

class LlamaAttentionMultiLayerGIN_with_Scaling (nn.Module):
    """
    Multi-layer, multi-headed attention with hybrid GIN-inspired updates.
    Includes dynamic self-loop scaling (attention epsilon) and residual blending (residual epsilon).
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Args:
            config: Model configuration.
            layer_idx: Index of the layer (used for caching and debugging).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers

        if layer_idx is None:
            raise ValueError(
                f"Layer index (layer_idx) must be provided for {self.__class__.__name__} to function correctly."
            )

        # Attention parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)


        self.attention_GIN_MLP_o_proj_at_end= getattr(config.gnn_config, "attention_GIN_MLP_o_proj_at_end", True)  #self.config.gnn_config('attention_GIN_MLP_o_proj_at_end', True)
        if self.attention_GIN_MLP_o_proj_at_end:
            if not self.hidden_size == self.num_heads * self.head_dim:
                print ('If self.attention_GIN_MLP_o_proj_at_end=True, must ensure that hidden_size = num_heads * head_dim. Adapting to fit, but may fail due to RoPE.')
                self.head_dim = self.hidden_size // self.num_heads  # Ensure no mismatch
                print ('new head_dim=', self.head_dim)
                assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        

        # Shared projections for key and query
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        # [EDIT START] Differential Attention Mechanism Parameters
        self.use_differential_attention = getattr(self.config.gnn_config, "use_differential_attention", False)


        if self.use_differential_attention:
            print ("Use differential attention.")
            self.q_proj_2 = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj_2 = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            #self.lambda_param = nn.Parameter(torch.tensor(1.0))  # Learnable lambda parameter

            #lambda_init = 0.8
            lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (self.layer_idx - 1))

            self.lambda_param = nn.ParameterList([
                nn.Parameter(torch.tensor(lambda_init)) for i in range(self.num_attention_layers)
            ])

        '''# Per-layer value MLPs (GIN-style updates with LayerNorm)
        self.v_proj_layers = nn.ModuleList(
            [self._build_mlp(self.hidden_size, self.num_key_value_heads * self.head_dim)
             for _ in range(self.num_attention_layers)]
        )'''
        
        self.attention_mix_mode= getattr(self.config.gnn_config, "attention_GIN_MLP_attention_mix_mode", "A")      
        
        self.GIN_MLP_mode = getattr(config.gnn_config, "attention_GIN_MLP_GIN_MLP_mode", 'shared') #shared=all GINs shared over all heads

        self.layer_norm_after_linear_and_MLP_aggregration = nn.LayerNorm(self.head_dim)  # Per-head LayerNorm

        if self.GIN_MLP_mode == 'shared':
            # Per-layer value MLPs (GIN-style updates with LayerNorm)
            self.GIN_MLP_layers = nn.ModuleList(
                [self._build_mlp(self.head_dim, self.head_dim)
                for _ in range(self.num_attention_layers)]
            )

        elif self.GIN_MLP_mode == 'per_head':
            #Initialize per-head GIN layers
            self._init_head_specific_gin()

        elif self.GIN_MLP_mode == 'per_head_loop':
            # Create a separate GIN MLP for each head at each layer
            self.GIN_MLP_layers = nn.ModuleList([
                nn.ModuleList([
                    self._build_mlp(self.head_dim, self.head_dim)
                    for _ in range(self.num_heads)
                ])
                for _ in range(self.num_attention_layers)
            ])
    
        elif self.GIN_MLP_mode == 'none':
            # Create a separate GIN MLP for each head at each layer
            print ("No GIN used, for reference experiment, attn_mix = C.")
            self.attention_mix_mode='C'
    
        else:
            print ("Unknown attention_GIN_MLP_GIN_MLP_mode option: ", self.GIN_MLP_mode)
            print ("Reverting to default: shared over all head, 'shared'.")
            self.GIN_MLP_layers = nn.ModuleList(
                [self._build_mlp(self.head_dim, self.head_dim)
                for _ in range(self.num_attention_layers)]
            )

        ####### parameters for scaling adjacency used for GIN ############
        # Retrieve mode and parameters from config
        self.threshold_mode = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_mode", "none")
        threshold = nn.Parameter(torch.tensor(getattr( self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", 0.2) ))  # Initialize threshold
        learnable_threshold = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_learnable_threshold", False)
        if learnable_threshold:
            #self.threshold = nn.Parameter(torch.tensor(getattr( self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", threshold) ))  # Initialize threshold
            self.threshold = nn.ParameterList([
                nn.Parameter(torch.tensor(threshold)) for _ in range(self.num_attention_layers)
                ])

            print ("Learnable threshold for adjaceny scaling for GIN.")
        else:
            print ("Threshold for adj scaling for GIN is not learnable.")
            #self.threshold = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", threshold)
            self.threshold = [
                    torch.tensor(threshold, dtype=torch.float32) for _ in range(self.num_attention_layers)
                ]

        self.binary_scale = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_binary_scale", 1.0)
        self.top_k_frac = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length", 0.1) #%10 of sequence length is used
       
        ####### parameters for scaling adjacency used for GIN ############
        

        #define linear layer for parallel

        self.v_proj_layers = nn.ModuleList([
            nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=config.attention_bias) for _ in range(self.num_attention_layers)
        ])
        #alternative, MLPs...
        '''# Per-layer value MLPs (GIN-style updates with LayerNorm)
        self.v_proj_layers = nn.ModuleList(
            [self._build_mlp(self.hidden_size, self.num_key_value_heads * self.head_dim)
             for _ in range(self.num_attention_layers)]
        )'''



        if not self.attention_GIN_MLP_o_proj_at_end:
            # Per-layer output projections
            self.o_proj_layers = nn.ModuleList(
                [nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
                for _ in range(self.num_attention_layers)]
            )
        else:
            self.o_proj_layer = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias) 
            

        # Dynamic epsilon initialization for attention self-loops
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_epsilon_strategy'):
            attention_strategy = self.config.gnn_config.attention_epsilon_strategy
        else:
            attention_strategy = "default"  # Fallback to default strategy

        if attention_strategy == "default":
            # Default: Full self-contribution in the first layer, none in subsequent layers
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) if i == 0 else nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
                for i in range(self.num_attention_layers)
            ])
        elif attention_strategy == "all_attention_epsilon_zero":
            # No self-loops in any layer
            print ("no self interactions, epsilon = 0.0")
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "no_self_interactions":
            # No self-loops in any layer
            print ("no self interactions, epsilon = -1.0")
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(-1.0)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "progressive":
            # Gradually decrease epsilon (linear decay)
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(-1.0 - i / self.num_attention_layers)) for i in range(self.num_attention_layers)
            ])
        elif attention_strategy == "uniform":
            # Same epsilon for all layers
            uniform_value = getattr(self.config.gnn_config, "attention_epsilon_uniform_value", 0.5)
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "learnable":
            # Fully learnable epsilon with random initialization
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.rand(())) for _ in range(self.num_attention_layers)
            ])
        else:
            raise ValueError(f"Unknown attention epsilon strategy: {attention_strategy}")

        # Dynamic epsilon initialization for residual blending
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'residual_epsilon_strategy'):
            residual_strategy = self.config.gnn_config.residual_epsilon_strategy
        else:
            residual_strategy = "default"  # Fallback to default strategy

        if residual_strategy == "default":
            # Default: Full blending in the first layer, light blending in subsequent layers
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0)) if i == 0 else nn.Parameter(torch.tensor(0.01))
                for i in range(self.num_attention_layers)
            ])
        elif residual_strategy == "progressive":
            # Gradually increase blending (linear growth)
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(i / self.num_attention_layers)) for i in range(self.num_attention_layers)
            ])
        elif residual_strategy == "uniform":
            # Same blending factor for all layers
            uniform_value = getattr(self.config.gnn_config, "residual_epsilon_uniform_value", 0.1)
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
            ])
        elif residual_strategy == "learnable":
            # Fully learnable blending with random initialization
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.rand(())) for _ in range(self.num_attention_layers)
            ])
        elif residual_strategy == "not_learnable_no_skip":
            # Constant value of 1 for residual_epsilon, not trainable
            self.residual_epsilon = [torch.tensor(1.0) for _ in range(self.num_attention_layers)]

        else:
            raise ValueError(f"Unknown residual epsilon strategy: {residual_strategy}")

        # LayerNorm for GIN updates
        #self.layer_norm = nn.LayerNorm(self.hidden_size)
        #self.layer_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Add sharpening parameters for each GIN layer with configurable initial values
        # Check for sharpening configuration
        self.use_sharpening = getattr(self.config.gnn_config, "use_sharpening", False)
        
        if self.use_sharpening:        

            sharpening_value_init = getattr(self.config.gnn_config, "sharpening_value_init", "value")
            
            if sharpening_value_init == 'value':
                
                initial_sharpening_value = getattr(self.config.gnn_config, "initial_sharpening_value", 1.0)
                self.sharpening_parameters = nn.ParameterList([
                    nn.Parameter(torch.tensor(initial_sharpening_value)) for _ in range(self.num_attention_layers)
                ])
            elif sharpening_value_init == 'random':

                self.sharpening_parameters = nn.ParameterList([
                            nn.Parameter(torch.rand(1) + 0.5)  # random between 0.5 and 1.5
                            for _ in range(self.num_attention_layers)
                        ])
            


        # [EDIT START] Add soft masking parameters
        self.use_soft_masking = getattr(self.config.gnn_config, "use_soft_masking", False)

        if self.use_soft_masking:
            self.soft_masking_initial_threshold = getattr(self.config.gnn_config, "soft_masking_initial_threshold", 0.01)
            self.soft_masking_tau = nn.ParameterList([
                nn.Parameter(torch.tensor(self.soft_masking_initial_threshold)) for _ in range(self.num_attention_layers)
            ])


        if getattr(config.gnn_config, "use_hierarchical_attention", False):
            print ("Use hiearchical attention.")
            
            self.hierarchical_enc_dec_type = getattr(self.config.gnn_config, "hierarchical_enc_dec_type", 'PerceiverAR')
        
            if self.hierarchical_enc_dec_type == 'PerceiverAR':
                print ('Hierarchical: PerceiverAR enc/dec')
                self.perceiver_causal = PerceiverAR(config)
            '''elif self.hierarchical_enc_dec_type == 'CausalDynamicCoarseGraining':
                print ('Hierarchical: CausalDynamicCoarseGraining enc/dec')
                self.perceiver_causal = CausalDynamicCoarseGraining(config)'''
                

        self.soft_masking_k = getattr(self.config.gnn_config, "soft_masking_k", 10.0)  # Default sharpness
        # [EDIT END]        


        # Optional rotary embeddings for positional encoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

############## BUILD MLP OPTIONS ###############
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """
        Builds a simple GIN-inspired MLP with hidden layers, ReLU activation, and configurable normalization.
        Args:
            input_dim: Input feature size.
            output_dim: Output feature size.
        Returns:
            nn.Sequential: The MLP module.
        """
        
        hidden_dim = (input_dim + output_dim) // 2


        # Check for attention_GIN_MLP_multiplier in gnn_config
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_multiplier = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_multiplier = 2
        
        hidden_dim_MLP = MLP_multiplier * hidden_dim

        # Choose normalization layer based on gnn_config
        if hasattr(self.config.gnn_config, 'use_no_norm_in_GIN_MLP') and self.config.gnn_config.use_no_norm_in_GIN_MLP:
            norm_layer = nn.Identity()  # No normalization
        else:
            norm_layer = (
                nn.LayerNorm(hidden_dim_MLP) 
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP 
                else LlamaRMSNorm(hidden_dim_MLP, eps=self.config.rms_norm_eps)
            )
            norm_layer_end = (
                nn.LayerNorm(hidden_dim) 
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP 
                else LlamaRMSNorm(hidden_dim, eps=self.config.rms_norm_eps)
            )
 
        # Optional dropout
        dropout_rate = (
            self.config.gnn_config.dropout_rate
            if hasattr(self.config.gnn_config, 'dropout_rate')
            else 0.1
        )
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim_MLP, bias=self.config.attention_bias),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            norm_layer,
            nn.Linear(hidden_dim_MLP, output_dim, bias=self.config.attention_bias),
            #norm_layer_end,
            
    )

    def _get_normalization_layer(self, hidden_dim):
        """
        Returns the appropriate normalization layer based on gnn_config.
        Args:
            hidden_dim (int): Dimension of the input features for normalization.
        Returns:
            nn.Module: The normalization layer.
        """
        if hasattr(self.config.gnn_config, 'use_no_norm_in_GIN_MLP') and self.config.gnn_config.use_no_norm_in_GIN_MLP:
            return nn.Identity()  # No normalization
        else:
            return (
                nn.LayerNorm(hidden_dim)
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP
                else LlamaRMSNorm(hidden_dim, eps=self.config.rms_norm_eps)
            )

    def _compute_hidden_dim_MLP(self):
        """
        Computes the hidden dimension for the GIN MLP layers based on config and multiplier.
        """
        hidden_dim = (self.head_dim + self.head_dim) // 2
        if hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_multiplier = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_multiplier = 2
        return MLP_multiplier * hidden_dim
        
    def _init_head_specific_gin(self):
        """
        Initializes the parameters for per-head GIN layers with proper initialization and dimensions.
        """
        hidden_dim_MLP = self._compute_hidden_dim_MLP()
        
        # Calculate proper initialization scaling factors
        fan_in = self.head_dim
        fan_out = hidden_dim_MLP
        bound = 1 / math.sqrt(fan_in)
        
        # First MLP layer parameters (properly scaled initialization)
        self.GIN_MLP_layers_head_weights_1 = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_heads, self.head_dim, hidden_dim_MLP).uniform_(-bound, bound))
            for _ in range(self.num_attention_layers)
        ])
        self.GIN_MLP_layers_head_biases_1 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_heads, hidden_dim_MLP))
            for _ in range(self.num_attention_layers)
        ])
        
        # Second layer initialization scaling
        bound = 1 / math.sqrt(hidden_dim_MLP)
        
        # Second MLP layer parameters
        self.GIN_MLP_layers_head_weights_2 = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_heads, hidden_dim_MLP, self.head_dim).uniform_(-bound, bound))
            for _ in range(self.num_attention_layers)
        ])
        self.GIN_MLP_layers_head_biases_2 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            for _ in range(self.num_attention_layers)
        ])
        
        # Per-head normalization layers
        self.GIN_MLP_layers_norms = nn.ModuleList([
            nn.ModuleList([
                self._get_normalization_layer(hidden_dim_MLP)
                for _ in range(self.num_heads)
            ])
            for _ in range(self.num_attention_layers)
        ])

    def _apply_per_head_mlp(self, features, layer_idx):
        """
        Applies the per-head MLP transformation with proper normalization.
        
        Args:
            features: Input tensor of shape (batch, num_heads, q_len, head_dim)
            layer_idx: Current layer index
        
        Returns:
            Transformed tensor of shape (batch, num_heads, q_len, head_dim)
        """
        batch_size, num_heads, seq_len, _ = features.shape
        
        # First linear layer
        transformed = torch.einsum(
            'bhlq,hqm->bhlm',
            features,
            self.GIN_MLP_layers_head_weights_1[layer_idx]
        )
        
        # Add bias and apply ReLU
        transformed = transformed + self.GIN_MLP_layers_head_biases_1[layer_idx].unsqueeze(0).unsqueeze(2)
        transformed = F.relu(transformed)
        
        # Apply per-head normalization
        normalized = torch.zeros_like(transformed)
        for head in range(num_heads):
            head_features = transformed[:, head, :, :]  # (batch, seq_len, hidden_dim)
            normalized[:, head, :, :] = self.GIN_MLP_layers_norms[layer_idx][head](head_features)
        
        # Second linear layer
        output = torch.einsum(
            'bhlm,hmf->bhlf',
            normalized,
            self.GIN_MLP_layers_head_weights_2[layer_idx]
        )
        
        # Add bias
        output = output + self.GIN_MLP_layers_head_biases_2[layer_idx].unsqueeze(0).unsqueeze(2)
        
        return output
    

    def _apply_linear(self, attn_weights, hidden_states, layer_idx):
        """
        Apply the output projection layer to hidden_states.

        Args:
            hidden_states (torch.Tensor): Shape (batch, num_heads, q_len, head_dim)
            layer_idx (int): The layer index to select the corresponding o_proj.

        Returns:
            torch.Tensor: Transformed hidden states with the same shape.
        """

        '''apply linear,  hidden_states (batch, num_heads, q_len, head_dim):  torch.Size([1, 8, 9, 70]) hidden_dim= 560
        apply linear, hidden_states after view  torch.Size([1, 9, 560])
        apply linear, projected_states at end  torch.Size([1, 8, 9, 70])'''

        batch_size, num_heads, q_len, head_dim = hidden_states.shape
        hidden_dim = num_heads * head_dim

        #hidden_states = (batch, num_heads, q_len, head_dim)
        #print ("apply linear,  hidden_states (batch, num_heads, q_len, head_dim): ", hidden_states.shape, "hidden_dim=", hidden_dim)

        # Reshape to combine heads for linear transformation
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # (batch, q_len, num_heads, head_dim)
        hidden_states = hidden_states.view(batch_size, q_len, hidden_dim)  # (batch, q_len, hidden_dim)

        #print ("apply linear, hidden_states after view ", hidden_states.shape,  )


        # Apply the output projection for the specified layer
        projected_states = self.v_proj_layers[layer_idx](hidden_states)  # (batch, q_len, hidden_dim)

        # Reshape back to per-head format
        projected_states = projected_states.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

        projected_states = torch.matmul(attn_weights, projected_states)  # (batch, num_heads, q_len, head_dim)
        #print ("apply linear, projected_states at end ", projected_states.shape,  )

        return projected_states

     

    def process_adjacency_with_modes(self, adj_matrix, layer_idx):
        """
        Process the adjacency matrix using various modes (SiLU, binary thresholding, top-k, or combined).

        Args:
            adj_matrix (torch.Tensor): Adjacency matrix of shape (batch, num_heads, q_len, q_len).

        Returns:
            torch.Tensor: Processed adjacency matrix.
        """

        threshold = self.threshold[layer_idx]
        binary_scale = self.binary_scale 

        top_k = int(adj_matrix.shape[-1] * self.top_k_frac)

        if self.threshold_mode == "none":
             
            return adj_matrix

        elif self.threshold_mode == "silu":
            # Apply SiLU-based thresholding
            processed_adj = F.silu(adj_matrix - threshold)

        elif self.threshold_mode == "relu":
            # Apply ReLU-based thresholding
            processed_adj = F.relu(adj_matrix - threshold)

        elif self.threshold_mode == "softplus":
            # Apply SiLU-based thresholding
            processed_adj = F.softplus(adj_matrix - threshold)

        elif self.threshold_mode == "flowthreshold":
            # Apply FlowThreshold-based thresholding
            processed_adj = FlowThreshold(adj_matrix, threshold)

        elif self.threshold_mode == "binary":
            # Apply binary thresholding with binary values (0 or 1)
            processed_adj = (adj_matrix > threshold).float()

        elif self.threshold_mode == "top_k":
            # Retain top-k elements per row with binary values (0 or 1)
            if top_k <= 0:
                raise ValueError("Top-k sparsity mode requires `top_k` to be greater than 0.")
            _, indices = torch.topk(adj_matrix, top_k, dim=-1)
            binary_adj = torch.zeros_like(adj_matrix).scatter_(-1, indices, 1.0)  # Set top-k positions to 1
            processed_adj = binary_adj

        elif self.threshold_mode == "silu+binary":
            # Apply SiLU attenuation
            silu_adj = F.silu(adj_matrix - threshold)
            # Apply binary mask (0 or 1 values) 
            binary_adj = (adj_matrix > threshold).float()
            # Combine both
            processed_adj = silu_adj + binary_scale * binary_adj
       
        elif self.threshold_mode == "silu_below_binary_above":
            # Apply SiLU below the threshold
            silu_adj = F.silu(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = silu_adj * (1 - binary_adj) + binary_adj
        
        elif self.threshold_mode == "relu_below_binary_above":
            # Apply SiLU below the threshold
            relu_adj = F.relu(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = relu_adj * (1 - binary_adj) + binary_adj

        elif self.threshold_mode == "softplus_below_binary_above":
            # Apply SiLU below the threshold
            softplus_adj = F.softplus(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = softplus_adj * (1 - binary_adj) + binary_adj

        else:
            raise ValueError(f"Unknown threshold mode: {self.threshold_mode}")
    
        # Ensure causality with a lower triangular mask
        seq_len = adj_matrix.size(-1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=adj_matrix.device), diagonal=0)  # Lower triangular
        processed_adj = processed_adj * causal_mask
        
        return processed_adj





########### END GIN-MLP OPTIONS ########### 



    def forward( #TODO -------> remove scaling, just create a clean copy in the other class, also, play with how adj_matrix is computed - once or every GNN layer (probably not), ...
            # use different training set - e.g. regular text vs. proteins?
            # compare in detail with regular attention... 
            # maybe add MLP back in, or alternative to MLP
            # 
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass incorporating all features from original attention.
        """

        if getattr(self.config.gnn_config, "use_hierarchical_attention", False):
            #print ("hidden_states before encoding", hidden_states.shape)
            #print("attention_mask before", attention_mask.shape)  # Debug print
            #print ("position_ids", position_ids.shape)
            #print (position_ids)
            # Apply rotary embeddings to hidden_states befo
            #if position_embeddings is not None:
            #    cos, sin = position_embeddings
            #    hidden_states, _ = apply_rotary_pos_emb(hidden_states, hidden_states, cos, sin)

            # Process with Perceiver
            #hidden_states = self.perceiver_causal.encode(hidden_states, attention_mask=attention_mask)
            original_attention_mask = attention_mask

            hidden_states, latent_attention_mask = self.perceiver_causal.encode(hidden_states, attention_mask=attention_mask)
            attention_mask=latent_attention_mask

            # Update position embeddings for latent length if they exist
            if position_embeddings is not None:
                cos, sin = position_embeddings

                latent_len = self.config.gnn_config.num_latents
                latent_len = torch.tensor(latent_len, device=position_ids.device)

                input_len = position_ids.size(1)
                if input_len < latent_len:
                    #Extending position_ids from {input_len} to {latent_len}
                    # Extend position_ids sequentially
                    batch_size = position_ids.size(0)
                    new_position_ids = torch.arange(latent_len, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
                    position_ids = new_position_ids 

                # Recalculate positional embeddings if latent length exceeds input length
                if cos.size(1) < latent_len:
                    #print(f"Extending positional embeddings from {cos.size(1)} to {latent_len}.")
                    cos, sin = self.rotary_emb(latent_len, position_ids)

                # Take just what we need for latent length
                latent_cos = cos[:, :self.config.gnn_config.num_latents, :]
                latent_sin = sin[:, :self.config.gnn_config.num_latents, :]
                position_embeddings = (latent_cos, latent_sin)

            #print ("hidden_states after encoding", hidden_states.shape)
            #print("attention_mask after", attention_mask.shape)  # Debug print
            


        bsz, q_len, _ = hidden_states.size()

        # Compute shared query and key projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)

        # [EDIT START] Additional Projections for Differential Attention
        if self.use_differential_attention:
            query_states_2 = self.q_proj_2(hidden_states)
            key_states_2 = self.k_proj_2(hidden_states)
            query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states_2 = repeat_kv(key_states_2, self.num_key_value_groups)
        # [EDIT END]

        # Apply positional embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # Multi-layer attention computation
        for layer_idx in range(self.num_attention_layers):

            #print ("hidden_states at beginning: ", hidden_states.shape)    
            if self.config.gnn_config.plot_for_debugging:
                print ("hidden_states at beginning: ", hidden_states.shape)          

            # Compute value projections using GIN-style MLP
            
            #before - MISTAKE ERROR
            #value_states = self.v_proj_layers[layer_idx](hidden_states)
            #value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            #value_states = repeat_kv(value_states, self.num_key_value_groups)

            # Compute attention weights
            if self.use_differential_attention:  # [EDIT START]
                # Compute attention scores for the two attention maps
                attn_scores_1 = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_scores_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim)

                # Apply attention mask to each set of scores
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                    attn_scores_1 = attn_scores_1 + causal_mask
                    attn_scores_2 = attn_scores_2 + causal_mask

                # Apply sharpening if enabled
                if self.use_sharpening:
                    alpha = self.sharpening_parameters[layer_idx]
                    attn_scores_1 = alpha * attn_scores_1
                    attn_scores_2 = alpha * attn_scores_2
                     
                # Apply softmax to each set of scores
                attn_probs_1 = nn.functional.softmax(attn_scores_1, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_probs_2 = nn.functional.softmax(attn_scores_2, dim=-1, dtype=torch.float32).to(query_states.dtype)

                # [EDIT START] Modulate attn_scores_2 with graph properties
                if getattr(self.config.gnn_config, "use_graph_property_modulation", False):
                    # Original implementation
                    # Create causal mask for degrees and clustering
                    seq_len = attn_scores_2.shape[-1]
                    causal_mask_graph = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores_2.device))

                    # Compute causal degrees (only consider previous positions)
                    degrees = (attn_scores_2 * causal_mask_graph.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)

                    # Compute causal clustering (strictly causal)
                    masked_attn = attn_scores_2 * causal_mask_graph.unsqueeze(0).unsqueeze(0)
                    causal_clustering = torch.matmul(masked_attn, masked_attn.transpose(-2, -1))
                    causal_clustering = causal_clustering * causal_mask_graph.unsqueeze(0).unsqueeze(0)  # Apply mask again
                    clustering = causal_clustering.sum(dim=-1, keepdim=True)

                    # Combine into modulation factor
                    modulation_factor = 1 + degrees + clustering

                    # Apply modulation
                    attn_scores_2 = attn_scores_2 + modulation_factor  # Note: Original code multiplies attn_probs_2, here we add to attn_scores_2

                # New normalized modulation approach
                # [EDIT START] Modulate attn_scores_2 with graph properties
                elif getattr(self.config.gnn_config, "use_graph_property_modulation_with_norm", False):
                    # Create causal mask for degrees and clustering
                    seq_len = attn_scores_2.shape[-1]
                    causal_mask_graph = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores_2.device))
                    
                    # Compute attention probabilities for attn_scores_2
                    attn_probs_2 = nn.functional.softmax(attn_scores_2, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
                    # Compute degrees per key token (sum over queries)
                    degrees = (attn_probs_2 * causal_mask_graph.unsqueeze(0).unsqueeze(0)).sum(dim=-2)
                    # degrees shape: (batch_size, num_heads, seq_len)
                    
                    # Normalize degrees to prevent large values
                    degrees = degrees / (degrees.max(dim=-1, keepdim=True)[0] + 1e-6)
                    
                    # Initialize modulation term with degrees
                    modulation = torch.log1p(degrees)
                    
                    # Check if causal clustering is enabled
                    if getattr(self.config.gnn_config, "use_graph_property_modulation_with_norm_use_causal_clustering", False):
                        # Compute causal clustering coefficients
                        # Masked attention probabilities (causal)
                        masked_attn_probs = attn_probs_2 * causal_mask_graph.unsqueeze(0).unsqueeze(0)
                        
                        # Compute second-order attentions
                        causal_clustering = torch.matmul(masked_attn_probs, masked_attn_probs.transpose(-2, -1))
                        causal_clustering = causal_clustering * causal_mask_graph.unsqueeze(0).unsqueeze(0)
                        # Sum over queries to get clustering coefficients per key
                        clustering = causal_clustering.sum(dim=-2)
                        # clustering shape: (batch_size, num_heads, seq_len)
                        
                        # Normalize clustering coefficients
                        clustering = clustering / (clustering.max(dim=-1, keepdim=True)[0] + 1e-6)
                        
                        # Apply logarithmic scaling to clustering coefficients
                        clustering_modulation = torch.log1p(clustering)
                        
                        # Add clustering modulation to the existing modulation term
                        modulation += clustering_modulation
                    
                    # Expand modulation to match the shape of attn_scores_2
                    modulation = modulation.unsqueeze(-2)  # Shape: (batch_size, num_heads, 1, seq_len)
                    
                    # Add modulation to attn_scores_2 before softmax
                    attn_scores_2 = attn_scores_2 + modulation
                # [EDIT END]
                

                # [EDIT END]                

                # Compute the differential attention
                #lambda_param = torch.clamp(self.lambda_param, 0.0, 1.0)
                lambda_param =  self.lambda_param [layer_idx]
                attn_weights = attn_probs_1 - lambda_param * attn_probs_2


            else:  # Logic without differential attention 
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                if self.use_sharpening:
                    alpha = self.sharpening_parameters[layer_idx]
                    attn_weights = nn.functional.softmax(alpha * attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                else:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # [EDIT END]            
                 

            # Apply soft masking if enabled
            if self.use_soft_masking:
                tau = self.soft_masking_tau[layer_idx]
                k = self.soft_masking_k
                soft_mask = torch.sigmoid(k * (attn_weights - tau))
                attn_weights = attn_weights * soft_mask

            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)


            ######## NEW !!! #####

            #hidden_states = hidden_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            ####hidden_states = hidden_states.contiguous().view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            ####hidden_states = repeat_kv(hidden_states, self.num_key_value_groups)
            if layer_idx==0:
                hidden_states = hidden_states.contiguous().view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                hidden_states = repeat_kv(hidden_states, self.num_key_value_groups)

            linear_output = self._apply_linear(attn_weights, hidden_states, layer_idx)  # (batch, num_heads, q_len, head_dim)

            #print (f"layer_idx={layer_idx}, hidden_states processed (batch, num_heads, q_len, head_dim):", hidden_states.shape)
            # Create identity matrix for self-loops


            if self.attention_mix_mode == 'C':
                attn_output=  linear_output #+ hidden_states
                continue
            ################################################################################################
            # PROCESS for potential sparsity etc. <<<<<<<<<<<< EXPERIMENT!!!!!!
            attn_weights=self.process_adjacency_with_modes(attn_weights, layer_idx)
            ################################################################################################

            identity = torch.eye(q_len, device=attn_weights.device).unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, q_len)
            #print ("identity", identity.shape)
            # Add (1 + epsilon) to the diagonal of the adjacency matrix
            #adj_with_self_loop = attn_weights + (1 + self.attention_epsilon[layer_idx]) * identity  # (batch, num_heads, q_len, q_len)

            #adj_with_self_loop = attn_weights + self.attention_epsilon[layer_idx] * identity  # (batch, num_heads, q_len, q_len)
            
            if self.attention_mix_mode == 'A':
                #V31
                adj_with_self_loop = attn_weights + self.attention_epsilon[layer_idx] * identity  # (batch, num_heads, q_len, q_len)
                
                #use output form v_proj for GIN
                hidden_states=linear_output

                #print ("adj_with_self_loop", adj_with_self_loop.shape)
            elif self.attention_mix_mode == 'B':
                adj_with_self_loop = attn_weights + identity  # (batch, num_heads, q_len, q_len)
                #use output form v_proj for GIN
                hidden_states=linear_output
            

            
            

            # Sum aggregation using the modified adjacency matrix
            aggregated_features = torch.matmul(adj_with_self_loop, hidden_states)  # (batch, num_heads, q_len, head_dim)

            #print ("aggregated_features:", aggregated_features.shape)
            
             
            if self.GIN_MLP_mode == 'shared':
                attn_output = self.GIN_MLP_layers[layer_idx](aggregated_features)

            elif self.GIN_MLP_mode == 'per_head':
                attn_output = self._apply_per_head_mlp(aggregated_features, layer_idx) #+ aggregated_features

            elif self.GIN_MLP_mode == 'per_head_loop':
                batch_size, num_heads, seq_len, head_dim = aggregated_features.shape
                output = torch.zeros_like(aggregated_features)
                for h in range(num_heads):
                    head_features = aggregated_features[:, h]  # (batch, seq_len, head_dim)
                    transformed = self.GIN_MLP_layers[layer_idx][h](head_features)  # use head-specific MLP
                    output[:, h] = transformed
                attn_output = output


            
            
            if self.attention_mix_mode == 'A':
                attn_output=(1.-self.residual_epsilon[layer_idx])*attn_output + self.residual_epsilon[layer_idx]*linear_output #+ hidden_states V31

            if self.attention_mix_mode == 'B':
                attn_output=attn_output #+ linear_output #+ hidden_states




            if layer_idx+1 < self.num_attention_layers:
                attn_output=self.layer_norm_after_linear_and_MLP_aggregration(attn_output) + hidden_states

            hidden_states = attn_output

            # Residual blending with residual epsilon
            #scaling_residual = torch.clamp(self.residual_epsilon[layer_idx], 0.0, 1.0)
            #hidden_states = scaling_residual * attn_output + (1.0 - scaling_residual) * hidden_states

            #DEBUG JUST LINEAR
            #hidden_states = linear_output


            ####### END NEW !!! #####


 

            if self.config.gnn_config.plot_for_debugging:
                
                # Attention adjacency matrix
                head_adj_mean = adj_with_self_loop.mean(dim=1).cpu().detach().numpy()
                # Plot adjacency matrix
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
                plt.colorbar(label="Attention Weight")
                plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx}, GIN layer {layer_idx})")

                # Plot epsilon values
                plt.subplot(1, 2, 2)
                eps_values = [param.item() for param in self.attention_epsilon]
                residual_eps_values = [param.item() for param in self.residual_epsilon]
                plt.plot(eps_values, label="Attention Epsilon", marker="o")
                plt.plot(residual_eps_values, label="Residual Epsilon", marker="x")
                plt.title("Epsilon Values Across Layers (GPT layer {self.layer_idx})")
                plt.xlabel("Layer Index")
                plt.ylabel("Epsilon Value")
                plt.legend()

                plt.tight_layout()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_filename = f"./GNN_ATTN_adj_plots_{timestamp}.svg"
                #plt.savefig(combined_filename, format="svg")

                plt.show()

            # Compute attention output
            #attn_output = torch.matmul(attn_weights, value_states)
            #print ("attn_weights: ", attn_weights.shape, "value_states", value_states.shape)
            #print ("attn_output: ", attn_output.shape, "hidden_dim", self.hidden_size)

            #attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

            #print ("attn_output after contigous etc.: ", attn_output.shape, )

 

            # Apply additional LayerNorm
            
            #hidden_states = self.layer_norm(hidden_states)
            

            if self.config.gnn_config.plot_for_debugging:
                print ("hidden_states at end: ", hidden_states.shape)

        if self.attention_GIN_MLP_o_proj_at_end: #do o_project only at end

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = self.o_proj_layer(attn_output)
            hidden_states=attn_output

             

        # Optional attention weights output
        if output_attentions:
            last_attn_weights = attn_weights
        else:
            last_attn_weights = None


        if getattr(self.config.gnn_config, "use_hierarchical_attention", False):
            
            '''print ("hidden_states before decoding", hidden_states.shape)
            print ("original_attention_mask", original_attention_mask.shape)
            '''
                
            hidden_states = self.perceiver_causal.decode(hidden_states, attention_mask=original_attention_mask)
            if input_len < latent_len:
                hidden_states=hidden_states[:,:input_len,:]
            #print ("hidden_states after decoding", hidden_states.shape)
            

        return hidden_states, last_attn_weights, past_key_value

### >EDITED FOR EXPERIMENT 







###### >>> SINGLE GIN MLP AFTER WE HAVE REGULAR ATTENTION <<<<<

class LlamaAttentionMultiLayer_Single_GIN (nn.Module):
    """
    Multi-layer, multi-headed attention with hybrid GIN-inspired updates.
    Includes dynamic self-loop scaling (attention epsilon) and residual blending (residual epsilon).
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Args:
            config: Model configuration.
            layer_idx: Index of the layer (used for caching and debugging).
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers

        if layer_idx is None:
            raise ValueError(
                f"Layer index (layer_idx) must be provided for {self.__class__.__name__} to function correctly."
            )

        # Attention parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)


        # Shared projections for key and query
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj= nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias) 


        self.separate_attention_for_GIN = getattr(self.config.gnn_config, 'attention_GIN_MLP_separate_attention', 'none')
         
        #If i wanted to create a second att matrix just for GIN
        self.q_proj_GIN = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj_GIN = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        print ("Created separate q/k project for GIN.")
        
  
        self.GIN_MLP_mode = getattr(config.gnn_config, "attention_GIN_MLP_GIN_MLP_mode", 'shared') #shared=all GINs shared over all heads

        #self.layer_norm_after_linear_and_MLP_aggregration = nn.LayerNorm(self.head_dim)  # Per-head LayerNorm

        self.pre_GIN_layernorm = LlamaRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps) # nn.LayerNorm(self.hidden_size)


        self.GIN_MLP_layers = nn.ModuleList(
                [self._build_mlp(self.hidden_size, self.hidden_size)
                for _ in range(self.num_attention_layers)]
        )

        ####### parameters for scaling adjacency used for GIN ############
        # Retrieve mode and parameters from config
        self.threshold_mode = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_mode", "none")
        threshold = nn.Parameter(torch.tensor(getattr( self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", 0.2) ))  # Initialize threshold
        learnable_threshold = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_learnable_threshold", False)
        if learnable_threshold:
            #self.threshold = nn.Parameter(torch.tensor(getattr( self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", threshold) ))  # Initialize threshold
            self.threshold = nn.ParameterList([
                nn.Parameter(torch.tensor(threshold)) for _ in range(self.num_attention_layers)
                ])

            print ("Learnable threshold for adjaceny scaling for GIN.")
        else:
            print ("Threshold for adj scaling for GIN is not learnable.")
            #self.threshold = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", threshold)
            self.threshold = [
                    torch.tensor(threshold, dtype=torch.float32) for _ in range(self.num_attention_layers)
                ]

        self.binary_scale = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_binary_scale", 1.0)
        self.top_k_frac = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length", 0.1) #%10 of sequence length is used
       
        ####### parameters for scaling adjacency used for GIN ############
        

        #define linear layer  

        # self.v_proj_layer =  nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=config.attention_bias) 
        #alternative, MLPs...
         


            

        # Dynamic epsilon initialization for attention self-loops
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_epsilon_strategy'):
            attention_strategy = self.config.gnn_config.attention_epsilon_strategy
        else:
            attention_strategy = "default"  # Fallback to default strategy

        if attention_strategy == "default":
            # Default: Full self-contribution in the first layer, none in subsequent layers
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) if i == 0 else nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
                for i in range(self.num_attention_layers)
            ])
        elif attention_strategy == "all_attention_epsilon_zero":
            # No self-loops in any layer
            print ("no self interactions, epsilon = 0.0")
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "no_self_interactions":
            # No self-loops in any layer
            print ("no self interactions, epsilon = -1.0")
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(-1.0)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "progressive":
            # Gradually decrease epsilon (linear decay)
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(-1.0 - i / self.num_attention_layers)) for i in range(self.num_attention_layers)
            ])
        elif attention_strategy == "uniform":
            # Same epsilon for all layers
            uniform_value = getattr(self.config.gnn_config, "attention_epsilon_uniform_value", 0.5)
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
            ])
        elif attention_strategy == "learnable":
            # Fully learnable epsilon with random initialization
            self.attention_epsilon = nn.ParameterList([
                nn.Parameter(torch.rand(())) for _ in range(self.num_attention_layers)
            ])
        else:
            raise ValueError(f"Unknown attention epsilon strategy: {attention_strategy}")

        # Dynamic epsilon initialization for residual blending
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'residual_epsilon_strategy'):
            residual_strategy = self.config.gnn_config.residual_epsilon_strategy
        else:
            residual_strategy = "default"  # Fallback to default strategy

        if residual_strategy == "default":
            # Default: Full blending in the first layer, light blending in subsequent layers
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0)) if i == 0 else nn.Parameter(torch.tensor(0.01))
                for i in range(self.num_attention_layers)
            ])
        elif residual_strategy == "progressive":
            # Gradually increase blending (linear growth)
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(i / self.num_attention_layers)) for i in range(self.num_attention_layers)
            ])
        elif residual_strategy == "uniform":
            # Same blending factor for all layers
            uniform_value = getattr(self.config.gnn_config, "residual_epsilon_uniform_value", 0.1)
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
            ])
        elif residual_strategy == "learnable":
            # Fully learnable blending with random initialization
            self.residual_epsilon = nn.ParameterList([
                nn.Parameter(torch.rand(())) for _ in range(self.num_attention_layers)
            ])
        elif residual_strategy == "not_learnable_no_skip":
            # Constant value of 1 for residual_epsilon, not trainable
            self.residual_epsilon = [torch.tensor(1.0) for _ in range(self.num_attention_layers)]

        else:
            raise ValueError(f"Unknown residual epsilon strategy: {residual_strategy}")

        # LayerNorm for GIN updates
        #self.layer_norm = nn.LayerNorm(self.hidden_size)
        #self.layer_norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Add sharpening parameters for each GIN layer with configurable initial values
        # Check for sharpening configuration
        self.use_sharpening = getattr(self.config.gnn_config, "use_sharpening", False)
        
        if self.use_sharpening:        

            sharpening_value_init = getattr(self.config.gnn_config, "sharpening_value_init", "value")
            
            if sharpening_value_init == 'value':
                
                initial_sharpening_value = getattr(self.config.gnn_config, "initial_sharpening_value", 1.0)
                self.sharpening_parameters = nn.ParameterList([
                    nn.Parameter(torch.tensor(initial_sharpening_value)) for _ in range(self.num_attention_layers)
                ])
            elif sharpening_value_init == 'random':

                self.sharpening_parameters = nn.ParameterList([
                            nn.Parameter(torch.rand(1) + 0.5)  # random between 0.5 and 1.5
                            for _ in range(self.num_attention_layers)
                        ])
            


        # [EDIT START] Add soft masking parameters
        self.use_soft_masking = getattr(self.config.gnn_config, "use_soft_masking", False)

        if self.use_soft_masking:
            self.soft_masking_initial_threshold = getattr(self.config.gnn_config, "soft_masking_initial_threshold", 0.01)
            self.soft_masking_tau = nn.ParameterList([
                nn.Parameter(torch.tensor(self.soft_masking_initial_threshold)) for _ in range(self.num_attention_layers)
            ])
 
        self.soft_masking_k = getattr(self.config.gnn_config, "soft_masking_k", 10.0)  # Default sharpness
        # [EDIT END]        


        # Optional rotary embeddings for positional encoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

############## BUILD MLP OPTIONS ###############
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """
        Builds a simple GIN-inspired MLP with hidden layers, ReLU activation, and configurable normalization.
        Args:
            input_dim: Input feature size.
            output_dim: Output feature size.
        Returns:
            nn.Sequential: The MLP module.
        """
        
        hidden_dim = (input_dim + output_dim) // 2


        # Check for attention_GIN_MLP_multiplier in gnn_config
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_multiplier = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_multiplier = 2
        
        hidden_dim_MLP = MLP_multiplier * hidden_dim

        # Choose normalization layer based on gnn_config
        if hasattr(self.config.gnn_config, 'use_no_norm_in_GIN_MLP') and self.config.gnn_config.use_no_norm_in_GIN_MLP:
            norm_layer = nn.Identity()  # No normalization
        else:
            norm_layer = (
                nn.LayerNorm(hidden_dim_MLP) 
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP 
                else LlamaRMSNorm(hidden_dim_MLP, eps=self.config.rms_norm_eps)
            )
            norm_layer_end = (
                nn.LayerNorm(hidden_dim) 
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP 
                else LlamaRMSNorm(hidden_dim, eps=self.config.rms_norm_eps)
            )
 
        # Optional dropout
        dropout_rate = (
            self.config.gnn_config.dropout_rate
            if hasattr(self.config.gnn_config, 'dropout_rate')
            else 0.1
        )
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim_MLP, bias=self.config.attention_bias),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            norm_layer,
            nn.Linear(hidden_dim_MLP, output_dim, bias=self.config.attention_bias),
            #norm_layer_end,
            
    )

    def _get_normalization_layer(self, hidden_dim):
        """
        Returns the appropriate normalization layer based on gnn_config.
        Args:
            hidden_dim (int): Dimension of the input features for normalization.
        Returns:
            nn.Module: The normalization layer.
        """
        if hasattr(self.config.gnn_config, 'use_no_norm_in_GIN_MLP') and self.config.gnn_config.use_no_norm_in_GIN_MLP:
            return nn.Identity()  # No normalization
        else:
            return (
                nn.LayerNorm(hidden_dim)
                if hasattr(self.config.gnn_config, 'use_layer_norm_in_GIN_MLP') and self.config.gnn_config.use_layer_norm_in_GIN_MLP
                else LlamaRMSNorm(hidden_dim, eps=self.config.rms_norm_eps)
            )
  

     
    def process_adjacency_with_modes(self, adj_matrix, layer_idx):
        """
        Process the adjacency matrix using various modes (SiLU, binary thresholding, top-k, or combined).

        Args:
            adj_matrix (torch.Tensor): Adjacency matrix of shape (batch, num_heads, q_len, q_len).

        Returns:
            torch.Tensor: Processed adjacency matrix.
        """

        threshold = self.threshold[layer_idx]
        binary_scale = self.binary_scale 

        top_k = int(adj_matrix.shape[-1] * self.top_k_frac)

        if self.threshold_mode == "none":
             
            return adj_matrix

        elif self.threshold_mode == "silu":
            # Apply SiLU-based thresholding
            processed_adj = F.silu(adj_matrix - threshold)

        elif self.threshold_mode == "relu":
            # Apply ReLU-based thresholding
            processed_adj = F.relu(adj_matrix - threshold)

        elif self.threshold_mode == "softplus":
            # Apply SiLU-based thresholding
            processed_adj = F.softplus(adj_matrix - threshold)

        elif self.threshold_mode == "flowthreshold":
            # Apply FlowThreshold-based thresholding
            processed_adj = FlowThreshold(adj_matrix, threshold)

        elif self.threshold_mode == "binary":
            # Apply binary thresholding with binary values (0 or 1)
            processed_adj = (adj_matrix > threshold).float()

        elif self.threshold_mode == "top_k":
            # Retain top-k elements per row with binary values (0 or 1)
            if top_k <= 0:
                raise ValueError("Top-k sparsity mode requires `top_k` to be greater than 0.")
            _, indices = torch.topk(adj_matrix, top_k, dim=-1)
            binary_adj = torch.zeros_like(adj_matrix).scatter_(-1, indices, 1.0)  # Set top-k positions to 1
            processed_adj = binary_adj

        elif self.threshold_mode == "silu+binary":
            # Apply SiLU attenuation
            silu_adj = F.silu(adj_matrix - threshold)
            # Apply binary mask (0 or 1 values) 
            binary_adj = (adj_matrix > threshold).float()
            # Combine both
            processed_adj = silu_adj + binary_scale * binary_adj
       
        elif self.threshold_mode == "silu_below_binary_above":
            # Apply SiLU below the threshold
            silu_adj = F.silu(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = silu_adj * (1 - binary_adj) + binary_adj
        
        elif self.threshold_mode == "relu_below_binary_above":
            # Apply SiLU below the threshold
            relu_adj = F.relu(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = relu_adj * (1 - binary_adj) + binary_adj

        elif self.threshold_mode == "softplus_below_binary_above":
            # Apply SiLU below the threshold
            softplus_adj = F.softplus(adj_matrix - threshold)
            # Binary above the threshold (set all above-threshold values to 1)
            binary_adj = (adj_matrix > threshold).float()
            # Combine: SiLU below threshold and binary (1) above threshold
            processed_adj = softplus_adj * (1 - binary_adj) + binary_adj

        else:
            raise ValueError(f"Unknown threshold mode: {self.threshold_mode}")
    
        # Ensure causality with a lower triangular mask
        seq_len = adj_matrix.size(-1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=adj_matrix.device), diagonal=0)  # Lower triangular
        processed_adj = processed_adj * causal_mask
        
        return processed_adj
 

    def forward( #TODO -------> remove scaling, just create a clean copy in the other class, also, play with how adj_matrix is computed - once or every GNN layer (probably not), ...
            # use different training set - e.g. regular text vs. proteins?
            # compare in detail with regular attention... 
            # maybe add MLP back in, or alternative to MLP
            # 
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass incorporating all features from original attention.
        """
 
        bsz, q_len, _ = hidden_states.size()

        # Compute shared query and key projections
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Apply positional embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        if self.use_sharpening:
            alpha = self.sharpening_parameters[layer_idx]
            attn_weights = nn.functional.softmax(alpha * attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)


        attn_output = torch.matmul(attn_weights, value_states)


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        hidden_states = self.o_proj(attn_output)


        residual = hidden_states

        hidden_states= self.pre_GIN_layernorm (hidden_states)

        # Multi-layer attention computation
        for layer_idx in range(self.num_attention_layers):

            #print ("hidden_states at beginning: ", hidden_states.shape)    
            if self.config.gnn_config.plot_for_debugging:
                print ("hidden_states at beginning: ", hidden_states.shape)          
 

            ######## NEW !!! #####

            #hidden_states = hidden_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            ####hidden_states = hidden_states.contiguous().view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            ####hidden_states = repeat_kv(hidden_states, self.num_key_value_groups)
            if layer_idx==0:
                  
                # use hidden states from linear attention to compute new adjacency matrix: attn_weights_GIN - which is then used for   

                # Compute GIN-specific query and key states
                query_states_GIN = self.q_proj_GIN(hidden_states)
                key_states_GIN = self.k_proj_GIN(hidden_states)

                # Compute GIN-specific attention weights
                attn_weights_GIN = torch.matmul(query_states_GIN, key_states_GIN.transpose(1, 2)) / math.sqrt(self.head_dim)
                
                if self.config.gnn_config.plot_for_debugging:
                    print ("attn_weights_GIN before mask", attn_weights_GIN.shape)

                if attention_mask is not None:
                    causal_mask = attention_mask[:, 0, :, :key_states_GIN.shape[-2]]
                    attn_weights_GIN = attn_weights_GIN + causal_mask
                if self.config.gnn_config.plot_for_debugging:
                    print ("attn_weights_GIN after mask", attn_weights_GIN.shape)
                
                attn_weights_GIN = nn.functional.softmax(attn_weights_GIN, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_GIN = nn.functional.dropout(attn_weights_GIN, p=self.attention_dropout, training=self.training)

                ################################################################################################
                # PROCESS for potential sparsity etc. <<<<<<<<<<<< EXPERIMENT!!!!!!
                attn_weights_GIN=self.process_adjacency_with_modes(attn_weights_GIN, layer_idx)
                ################################################################################################

                # Add self-loops to GIN adjacency matrix
                identity = torch.eye(q_len, device=attn_weights_GIN.device).unsqueeze(0)
                attn_weights_GIN = attn_weights_GIN + self.attention_epsilon[0] * identity

                
            # Sum aggregation using the modified adjacency matrix
            aggregated_features = torch.matmul(attn_weights_GIN, hidden_states)  # (batch, q_len, hidden_dim)

             
            attn_output = self.GIN_MLP_layers[layer_idx](aggregated_features) # (batch, q_len, hidden_dim)
  
            
            if layer_idx+1 < self.num_attention_layers: #if not first, but more than one GIN do norm 
                #attn_output=self.layer_norm_after_linear_and_MLP_aggregration(attn_output) #+ hidden_states
                attn_output = attn_output + hidden_states

            hidden_states = attn_output # (batch, q_len, hidden_dim)

            
            ####### END NEW !!! #####



            if self.config.gnn_config.plot_for_debugging:
                
                # Attention adjacency matrix
                head_adj_mean = attn_weights_GIN.cpu().detach().numpy()
                # Plot adjacency matrix
                plt.figure(figsize=(6, 6))
                plt.subplot(1, 1, 1)
                plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
                plt.colorbar(label="Attention Weight")
                plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx}, GIN-MLP layer {layer_idx})")

                plt.tight_layout()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_filename = f"./GNN_ATTN_adj_plots_{timestamp}.svg"
                #plt.savefig(combined_filename, format="svg")

                plt.show()
 
                print ("hidden_states at end: ", hidden_states.shape)

        hidden_states = hidden_states + residual

        # Optional attention weights output
        if output_attentions:
            last_attn_weights = attn_weights
        else:
            last_attn_weights = None
    

        if self.config.gnn_config.plot_for_debugging:
            print ("return hidden_states", hidden_states.shape)
        return hidden_states, last_attn_weights, past_key_value


 
  
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple 

class LlamaAttentionMultiLayerGIN(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

        self.use_rope_every_layer = getattr(config.gnn_config, "use_GNN_from_attention_add_RoPE_at_every_layer", False)

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = getattr(config, "is_causal", True)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        self.epsilon = nn.Parameter(torch.zeros(self.num_attention_layers, self.num_heads))
        self.gin_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.head_dim, 2 * self.head_dim),
                    nn.ReLU(),
                    nn.Linear(2 * self.head_dim, self.head_dim)
                ) for _ in range(self.num_heads)
            ]) for _ in range(self.num_attention_layers)
        ])
        self.o_proj_last = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
    # [Unchanged helper methods]
    def _split_tensor_for_tp(self, tensor: torch.Tensor, dim: int) -> List[torch.Tensor]:
        if self.pretraining_tp > 1:
            return tensor.split(tensor.size(dim) // self.pretraining_tp, dim=dim)
        return [tensor]

    def _create_causal_mask(self, bsz: int, q_len: int, k_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.full((q_len, k_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1).expand(bsz, 1, q_len, k_len)
        return mask.to(dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # PART 1: Standard attention processing [unchanged]
        # Project and reshape query/key/value states
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values and repeat KV heads
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if self.is_causal:
            causal_mask = self._create_causal_mask(bsz, q_len, key_states.shape[-2], attn_weights.dtype, attn_weights.device)
            attn_weights = attn_weights + causal_mask

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # PART 2: Initial attention computation and transformation for GIN
        attn_output = torch.matmul(attn_weights, value_states)  # [bsz, num_heads, q_len, head_dim]
        
        # Plot initial attention weights if debugging
        if self.config.gnn_config.plot_for_debugging:
            head_adj_mean = attn_weights.mean(dim=1).cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Initial Adjacency Matrix (GPT layer {self.layer_idx})")
            plt.tight_layout()
            plt.show()

        # PART 3: GIN Processing
        hidden_layer = attn_output.transpose(1, 2)  # [bsz, q_len, num_heads, head_dim]
        initial_hidden = hidden_layer

        # GIN layers
        for layer_idx in range(self.num_attention_layers):
            new_hidden_states = []
            
            for head in range(self.num_heads):
                head_hidden = hidden_layer[:, :, head, :]  # [bsz, q_len, head_dim]
                head_adj = attn_weights[:, head]           # [bsz, q_len, k_len]
                
                if self.config.gnn_config.plot_for_debugging:
                    head_adj_mean = head_adj.mean(dim=0).cpu().detach().numpy()
                    plt.figure(figsize=(8, 8))
                    plt.imshow(head_adj_mean, cmap="viridis", aspect="auto")
                    plt.colorbar(label="Attention Weight")
                    plt.title(f"Head Adjacency Matrix (GPT layer {self.layer_idx}, GIN Layer {layer_idx}, Head {head})")
                    plt.tight_layout()
                    plt.show()

        
                neighbor_sum = torch.bmm(head_adj, head_hidden)
                gin_input = (1 + self.epsilon[layer_idx, head]) * head_hidden + neighbor_sum
                head_output = self.gin_mlps[layer_idx][head](gin_input)

                if self.use_rope_every_layer:
                    head_output_pre_rope = head_output.unsqueeze(1)
                    head_output_with_rope = apply_rotary_pos_emb(head_output_pre_rope, head_output_pre_rope, cos, sin)[0]
                    head_output = head_output_with_rope.squeeze(1)

                new_hidden_states.append(head_output)

            hidden_layer = torch.stack(new_hidden_states, dim=2)  # [bsz, q_len, num_heads, head_dim]

        # PART 4: Final Processing
        if getattr(self.config.gnn_config, "use_gin_residual", False):
            hidden_layer = hidden_layer + initial_hidden

        # Make sure tensor is contiguous before reshaping
        attn_output = hidden_layer.contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Final projection with tensor parallelism support
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj_last.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj_last(attn_output)

        # Cache handling
        if use_cache:
            past_key_value = Cache(
                key_states=key_states,
                value_states=value_states,
                layer_idx=self.layer_idx,
            )

        if not output_attentions:
            attn_weights = None

        # FIXED: Return attn_output instead of hidden_states
        return attn_output, attn_weights, past_key_value

##################  ######### END MULTILAYER GIN ########################


class LlamaSimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        #self.intermediate_size = config.intermediate_size
        #self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.mlp_bias)

        #self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        #proj = self.proj(self.act_fn(x))
        proj = self.act_fn(self.proj(x))
        return proj
    
class ShallowLlamaMLP (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.proj_out = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        
        inter = self.act_fn(self.proj(x))
        proj = self.proj_out(inter)

        return proj


class LlamaDecoderLayerWithGNN(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.gnn_config = config.gnn_config
        self.layer_idx = layer_idx

    
        # Initialize self-attention layers based on the configuration
        if self.gnn_config.use_GNN_from_attention in ['LlamaAttention_Original',
                                                      'LlamaAttentionGIN',
                                                      'LlamaAttentionPNA',
                                                      'LlamaAttentionPNA_LM',
                                                      ]:
            
            if self.gnn_config.use_GNN_from_attention == 'LlamaAttentionGIN':  #o1pro
                self.self_attn = LlamaAttentionGIN(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttentionPNA':  #o1pro - PNA variant!
                self.self_attn = LlamaAttentionPNA(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttentionPNA_LM':  #o1pro - PNA variant!
                self.self_attn = LlamaAttentionPNA_LM(config=config, layer_idx=layer_idx)
                
            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttention_Original':
                self.self_attn = LlamaAttention_Original(config=config, layer_idx=layer_idx)

        else:
            print (f"Not found ({self.gnn_config.use_GNN_from_attention}), falling back to standard.")
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        
        self.skip_around_MLP = True #usually use skip around MLP, except for some MLP choices
        if self.gnn_config.MLP_type == 'linear_MLP':
            self.mlp= LlamaSimpleMLP(config)
            print ("Linear/simple MLP")
        elif self.gnn_config.MLP_type == 'standard_MLP':
            self.mlp = LlamaMLP(config)
            print ("Standard MLP")
        elif self.gnn_config.MLP_type == 'shallow_MLP':
            self.mlp = ShallowLlamaMLP(config)
        elif self.gnn_config.MLP_type == 'no_MLP':
            self.mlp = nn.Identity()
            self.skip_around_MLP = False #No MLP
            print ("No MLP, also removed skip connection.")

        ######## MLP-GIN OPTIONS ############
        elif self.gnn_config.MLP_type == 'LlamaMLP_HalfwayGIN':
            self.mlp = LlamaMLP_HalfwayGIN(config)
        elif self.gnn_config.MLP_type == 'LlamaMLP_HalfwayGIN_MultiHop':
            self.mlp = LlamaMLP_MultiHop(config)
            print ("LlamaMLP_MultiHop, does GIN inside MLP, A, A2, A3, multi-hope.")
        elif self.gnn_config.MLP_type == 'LlamaMLP_HalfwayGIN_MultiAggregration':
            self.mlp = LlamaMLP_HalfwayGIN_MultiAggregration(config)
            print ("LlamaMLP_HalfwayGIN_MultiAggregration, multiple aggregators.")
        ######## MLP-GIN OPTIONS ############
        
        else:
            print (f"Unknown MLP type: {self.gnn_config.MLP_type}, falling back to standart MLP type.")
            self.mlp = LlamaMLP(config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Calculate linear scale based on layer index
        num_layers = config.num_hidden_layers
        initial_scale = self.gnn_config.lambda_GNN_initial if self.gnn_config.lambda_GNN_initial is not None else self.gnn_config.lambda_GNN

        final_scale = self.gnn_config.lambda_GNN
        if num_layers>1:
            linear_scale = initial_scale + (final_scale - initial_scale) * (layer_idx / (num_layers - 1))
        else:
            linear_scale=initial_scale

        self.gnn_mode = self.gnn_config.gnn_mode  # 'single' or 'per_head'

        if self.gnn_mode == 'per_head':
            self.gnns = nn.ModuleList([
                CausalGraphNeuralNetwork(self.gnn_config, transformer_hidden_dim=self.head_dim) for _ in range(self.num_heads)
            ])
            self.ff_layer = nn.Linear(self.num_heads * self.head_dim, self.hidden_size) if self.gnn_config.per_head_ff else nn.Identity()
            if self.gnn_config.adj_construction_method == 'learnable_aggregate':
                self.adj_transform = AggregatedLearnableAdjacencyTransformer(
                    num_heads=self.num_heads,
                    adj_transform_hidden_dim=self.gnn_config.adj_transform_hidden_dim,
                    activation=self.gnn_config.learnable_aggregate_activation,
                )
            # Initialize trainable_scale with the calculated linear scale
            self.trainable_scale = nn.Parameter(torch.tensor(linear_scale, dtype=torch.float32))


        elif self.gnn_mode == 'single':
            self.gnn = CausalGraphNeuralNetwork(self.gnn_config, transformer_hidden_dim=config.hidden_size)
            self.gnn_norm = LlamaRMSNorm(config.hidden_size, eps=self.gnn_config.rms_norm_eps) if self.gnn_config.use_layer_norm else nn.Identity()
            if self.gnn_config.adj_construction_method == 'learnable_aggregate':
                self.adj_transform = AggregatedLearnableAdjacencyTransformer(
                    num_heads=config.num_attention_heads,
                    adj_transform_hidden_dim=self.gnn_config.adj_transform_hidden_dim,
                    activation=self.gnn_config.learnable_aggregate_activation,
                )
            # Initialize trainable_scale with the calculated linear scale
            self.trainable_scale = nn.Parameter(torch.tensor(linear_scale, dtype=torch.float32))

        elif self.gnn_mode == 'none':
            self.gnn = nn.Identity()
            self.gnn_norm = nn.Identity()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.gnn_config.use_original_hidden_states:
            hidden_states_original = hidden_states #save since we need it in the case we want to use the original hidden states

        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=True,
            **{k: v for k, v in kwargs.items() if k != 'output_attentions'},
        )
        
        if self.gnn_config.use_original_hidden_states: 
            transformer_hidden_states = hidden_states_original #in this case the residual is added later
        else:
            hidden_states = residual + hidden_states
            transformer_hidden_states = hidden_states #just use output of self attention with residual

        additional_adj = kwargs.get("additional_adj", None)
        input_ids = None

        if self.gnn_config.gnn_logic == 'before_MLP':
            adj = self.construct_adjacency(self_attn_weights, attention_mask, additional_adj, input_ids)
            combined_hidden_states = self.apply_gnn(transformer_hidden_states, adj, attention_mask, position_embeddings)

            if self.gnn_config.use_original_hidden_states:
                #here i replace self-attention with GNN, and the residual is added now
                combined_hidden_states=combined_hidden_states+residual
                if self.gnn_config.use_original_hidden_states_add_attention:
                    combined_hidden_states=combined_hidden_states #+hidden_states_attention

            ############ MLP BLOCK with RESIDULAL ##############
            residual = combined_hidden_states
            combined_hidden_states = self.post_attention_layernorm(combined_hidden_states)

            if self.gnn_config.MLP_type in ['LlamaMLP_HalfwayGIN', 'LlamaMLP_HalfwayGIN_MultiHop', 'LlamaMLP_HalfwayGIN_MultiAggregration']:
                mlp_out = self.mlp(combined_hidden_states, self_attn_weights)
            else: #standard
                mlp_out = self.mlp(combined_hidden_states)
            
            #hidden_states = residual + mlp_out
            if self.skip_around_MLP:
                hidden_states = residual + mlp_out
            else:
                hidden_states =  mlp_out
            
            

            ############ MLP BLOCK with RESIDULAL ##############

        elif self.gnn_config.gnn_logic == 'after_MLP':
            # Apply MLP first

            ############ MLP BLOCK with RESIDULAL ##############
            residual = transformer_hidden_states
            hidden_states = self.post_attention_layernorm(transformer_hidden_states)
            mlp_out = self.mlp(hidden_states)
            
            #combined_hidden_states = residual + mlp_out  # Add residual for MLP
            if self.skip_around_MLP:
                combined_hidden_states = residual + mlp_out  # Add residual for MLP
            else:
                combined_hidden_states = mlp_out

            ############ MLP BLOCK with RESIDULAL ##############

            # Construct adjacency matrix
            adj = self.construct_adjacency(self_attn_weights, attention_mask, additional_adj, input_ids)

            # Then apply GNN after MLP
            combined_hidden_states = self.apply_gnn(combined_hidden_states, adj, attention_mask, position_embeddings)
            hidden_states = combined_hidden_states  # Final hidden states include GNN output

        elif self.gnn_config.gnn_logic == 'parallel_GNN_MLP':
            # Initial Layer Norm and Residual for MLP
            residual_mlp = transformer_hidden_states
            hidden_states_mlp = self.post_attention_layernorm(transformer_hidden_states)

            # Pass through MLP and add residual
            mlp_out = self.mlp(hidden_states_mlp)
            hidden_states_mlp = residual_mlp + mlp_out  # First residual path

            # Initial Layer Norm for GNN
            hidden_states_gnn = self.post_attention_layernorm(transformer_hidden_states)

            # Construct adjacency matrix
            adj = self.construct_adjacency(self_attn_weights, attention_mask, additional_adj)

            # Apply GNN and add second residual
            hidden_states_gnn = self.apply_gnn(hidden_states_gnn, adj, attention_mask, position_embeddings)

            # Combine both residual outputs (MLP and GNN)
            hidden_states = hidden_states_mlp + hidden_states_gnn  # Final output

            hidden_states = self.post_attention_layernorm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def construct_adjacency(self, attn_weights, attention_mask=None, additional_adj=None, input_ids=None):

        #if GNN is not used, no processing needed
        if self.gnn_mode == 'none':
            return None
        
        method = self.gnn_config.adj_construction_method
        batch_size, num_heads, seq_len, _ = attn_weights.size()

        # Step 1: Construct adjacency matrix based on chosen method
        if self.gnn_mode == 'single':
            if method == 'mean':
                adj = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
            elif method == 'sum':
                adj = attn_weights.sum(dim=1).clamp(max=1.0)
            elif method == 'learnable_aggregate':
                adj = self.adj_transform(attn_weights)
            elif method == 'threshold_any':
                tau = self.gnn_config.threshold_any_tau
                current_threshold = self.gnn_config.threshold
                soft_mask = torch.sigmoid((attn_weights - current_threshold) / tau)
                adj = soft_mask.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown adj_construction_method: {method}")
            
        elif self.gnn_mode == 'per_head':
            if method in ['mean', 'sum']:
                adj = attn_weights  # [batch_size, num_heads, seq_len, seq_len]
            elif method == 'learnable_aggregate':
                adj = self.adj_transform(attn_weights)  # [batch_size, seq_len, seq_len]
            elif method == 'threshold_any':
                tau = self.gnn_config.threshold_any_tau
                current_threshold = self.gnn_config.threshold
                soft_mask = torch.sigmoid((attn_weights - current_threshold) / tau)
                adj = soft_mask
            else:
                raise ValueError(f"Unknown adj_construction_method: {method}")

        # Step 2: Apply distance scaling if configured
        if self.gnn_config.use_distance_scaling:
            distance_weights = torch.arange(seq_len, device=adj.device).float().unsqueeze(0) - torch.arange(seq_len, device=adj.device).float().unsqueeze(1)
            distance_weights = torch.abs(distance_weights) * self.gnn_config.distance_weight_strength

            # Apply scaling method
            if self.gnn_config.distance_scaling_method == 'sigmoid':
                distance_weights = torch.sigmoid(distance_weights)
            elif self.gnn_config.distance_scaling_method == 'exp':
                distance_weights = torch.exp(distance_weights)
            elif self.gnn_config.distance_scaling_method == 'power':
                distance_weights = torch.pow(distance_weights + 1, self.gnn_config.distance_weight_strength)

            distance_weights = distance_weights / distance_weights.max()

            if self.gnn_mode == 'per_head':
                distance_weights = distance_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)
            else:
                distance_weights = distance_weights.unsqueeze(0).expand(batch_size, seq_len, seq_len)

            adj = adj * distance_weights

        # Step 3: Apply attention mask
        if attention_mask is not None:
            binary_mask = (attention_mask > -1e10).float()
            if self.gnn_mode == 'per_head':
                binary_mask = binary_mask.expand(batch_size, num_heads, seq_len, seq_len)
            else:
                binary_mask = binary_mask.squeeze(1).expand(batch_size, seq_len, seq_len)
            adj = adj * binary_mask

        # Additional transformations
        if self.gnn_config.continuous_transform_alpha > 0:
            adj = torch.sigmoid(self.gnn_config.continuous_transform_alpha * (adj - self.gnn_config.threshold))

        if self.gnn_config.zero_below_epsilon_threshold:
            adj = torch.where(adj > self.gnn_config.epsilon_threshold, adj, torch.zeros_like(adj))

        if self.gnn_config.remove_self_connections:
            if self.gnn_mode == 'single':
                diag_mask = torch.eye(seq_len, device=adj.device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
                adj = adj.masked_fill(diag_mask.bool(), 0.0)
            elif self.gnn_mode == 'per_head':
                diag_mask = torch.eye(seq_len, device=adj.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)
                adj = adj.masked_fill(diag_mask.bool(), 0.0)

        if self.gnn_config.enforce_causality:
            if self.gnn_config.remove_self_connections:
                diag_offset = -1
            else:
                diag_offset = 0
            if self.gnn_mode == 'single':
                adj = torch.tril(adj, diagonal=diag_offset)
            elif self.gnn_mode == 'per_head':
                adj = torch.tril(adj, diagonal=diag_offset)

        # Check dimensions and add additional adjacency if provided
        if self.gnn_mode == 'single':
            expected_dims = (batch_size, seq_len, seq_len)
        elif self.gnn_mode == 'per_head':
            expected_dims = (batch_size, self.num_heads, seq_len, seq_len)
        else:
            raise ValueError(f"Unknown gnn_mode: {self.gnn_mode}")

        if additional_adj is not None:
            if additional_adj.shape != expected_dims:
                raise ValueError(f"Expected additional_adj to have dimensions {expected_dims} but got {additional_adj.shape}")
            
            #TODO check if clamping works, may need other measures
            adj += additional_adj.clamp(max=1.0)

        # Debugging plot for adjacency
        if self.gnn_config.plot_for_debugging:
            if self.gnn_mode == 'single':
                # Mean is used to average over batch
                adj_mean = adj.mean(dim=0).cpu().detach().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(adj_mean, cmap='viridis')
                plt.colorbar()
                plt.title('Adjacency Matrix (Single Mode)')

                # If tokenizer is provided, use tokens as labels
                if self.gnn_config.tokenizer is not None and input_ids is not None:
                    # Assuming input_ids shape is [batch_size, seq_len]
                    # Use the first example in the batch for labels
                    token_ids = input_ids[0].cpu().numpy()
                    tokens = [self.gnn_config.tokenizer.decode([token_id]).strip() for token_id in token_ids]

                    # Truncate tokens to improve readability
                    max_token_length = 10  # Adjust as needed
                    tokens = [token[:max_token_length] for token in tokens]

                    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
                    plt.yticks(ticks=range(len(tokens)), labels=tokens)
                    plt.xlabel('Tokens')
                    plt.ylabel('Tokens')

                plt.tight_layout()
                plt.show()
            elif self.gnn_mode == 'per_head':
                cols = 4
                rows = (self.num_heads + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
                fig.suptitle('Adjacency Matrices for Each Head (Per Head Mode)', fontsize=16)
                for head in range(self.num_heads):
                    row, col = divmod(head, cols)
                    ax = axes[row, col] if rows > 1 else axes[col]
                    # Mean is used to average over batch
                    adj_mean_per_head = adj[:, head, :, :].mean(dim=0).cpu().detach().numpy()
                    im = ax.imshow(adj_mean_per_head, cmap='viridis')
                    ax.set_title(f'Head {head}')
                    fig.colorbar(im, ax=ax)

                    # If tokenizer is provided, use tokens as labels
                    if self.gnn_config.tokenizer is not None and input_ids is not None:
                        token_ids = input_ids[0].cpu().numpy()
                        tokens = [self.gnn_config.tokenizer.decode([token_id]).strip() for token_id in token_ids]

                        # Truncate tokens to improve readability
                        max_token_length = 10  # Adjust as needed
                        tokens = [token[:max_token_length] for token in tokens]

                        ax.set_xticks(range(len(tokens)))
                        ax.set_xticklabels(tokens, rotation=90)
                        ax.set_yticks(range(len(tokens)))
                        ax.set_yticklabels(tokens)
                        ax.set_xlabel('Tokens')
                        ax.set_ylabel('Tokens')

                # Remove any extra subplots
                for i in range(self.num_heads, rows * cols):
                    fig.delaxes(axes.flatten()[i])

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
       
        return adj

    def apply_gnn(self, hidden_states, adj, attention_mask, position_embeddings):
        if self.gnn_mode == 'single':
            return self._apply_single_gnn(hidden_states, adj, attention_mask, position_embeddings)
        elif self.gnn_mode == 'per_head':
            return self._apply_per_head_gnn(hidden_states, adj, attention_mask, position_embeddings)
        elif self.gnn_mode == 'none':
            return hidden_states
        

    def _apply_single_gnn(self, hidden_states, adj, attention_mask, position_embeddings):
        batch_size, seq_len, hidden_dim = hidden_states.size()
        residual=hidden_states

        data_list = []
        for batch_idx in range(batch_size):
            x = hidden_states[batch_idx]  # [seq_len, hidden_dim]
            adj_matrix = adj[batch_idx]   # [seq_len, seq_len]

            edge_indices = adj_matrix.nonzero(as_tuple=False).t()
            edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]

            data = Data(x=x, edge_index=edge_indices, edge_weight=edge_weights)
            data_list.append(data)


        batch = Batch.from_data_list(data_list).to(hidden_states.device)

        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        batch_vector = batch.batch

        # Apply RoPE if configured
        if self.gnn_config.add_rope:
            x = x.view(batch_size, seq_len, -1)
            cos, sin = position_embeddings
            x, _ = apply_rotary_pos_emb(x, x, cos, sin, position_ids=None)
            x = x.view(-1, hidden_dim)

        # Pass through GNN
        gnn_out = self.gnn(x, edge_index, edge_weight=edge_weight, batch=batch_vector)

        # Split the output back into individual graphs
        gnn_out_list = gnn_out.split(batch.batch.bincount().tolist(), dim=0)
        gnn_out = torch.stack(gnn_out_list, dim=0)  # [batch_size, seq_len, hidden_dim]

        # Add normalization to hidden states if configured
        if self.gnn_config.norm_to_hidden_states:
            t_rms = torch.sqrt(torch.mean(hidden_states ** 2))
            g_rms = torch.sqrt(torch.mean(gnn_out ** 2))
            gnn_out = gnn_out * (t_rms / g_rms)

        gnn_out = self.gnn_norm(gnn_out)

        return residual + gnn_out * self.trainable_scale

    def _apply_per_head_gnn(self, hidden_states, adj, attention_mask, position_embeddings):
        residual = hidden_states
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states_per_head = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        hidden_states_per_head = hidden_states_per_head.permute(2, 0, 1, 3)  # [num_heads, batch_size, seq_len, head_dim]

        gnn_outputs = []
        for head_idx in range(self.num_heads):
            head_hidden = hidden_states_per_head[head_idx]  # [batch_size, seq_len, head_dim]
            head_adj = adj[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]

            data_list = []
            for batch_idx in range(batch_size):
                x = head_hidden[batch_idx]  # [seq_len, head_dim]
                adj_matrix = head_adj[batch_idx]  # [seq_len, seq_len]

                # Enforce causality and remove self-loops if not already done
                # (This should have been handled in construct_adjacency)

                edge_indices = adj_matrix.nonzero(as_tuple=False).t()
                edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]

                data = Data(x=x, edge_index=edge_indices, edge_weight=edge_weights)
                data_list.append(data)

            batch = Batch.from_data_list(data_list)

            x = batch.x
            edge_index = batch.edge_index
            edge_weight = batch.edge_weight
            batch_vector = batch.batch

            # Apply GNN for this head
            gnn_out = self.gnns[head_idx](x, edge_index, edge_weight=edge_weight, batch=batch_vector)

            # Split and stack outputs
            gnn_out_list = gnn_out.split(batch.batch.bincount().tolist(), dim=0)
            gnn_out = torch.stack(gnn_out_list, dim=0)  # [batch_size, seq_len, head_dim]

            gnn_outputs.append(gnn_out)

        # Stack outputs from all heads
        gnn_outputs = torch.stack(gnn_outputs, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        gnn_outputs = gnn_outputs.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        gnn_out = self.ff_layer(gnn_outputs)

        # Add normalization to hidden states if configured
        if self.gnn_config.norm_to_hidden_states:
            t_rms = torch.sqrt(torch.mean(hidden_states ** 2))
            g_rms = torch.sqrt(torch.mean(gnn_out ** 2))
            gnn_out = gnn_out * (t_rms / g_rms)

        return residual + gnn_out * self.trainable_scale
