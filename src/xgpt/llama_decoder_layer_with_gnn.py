 # llama_decoder_layer_with_gnn.py
  
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from typing import Optional, Tuple

from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaRMSNorm,
    Cache,
    LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)
from transformers.models.llama.modeling_llama import *

from torch_geometric.data import Batch, Data

from .graph_neural_network import CausalGraphNeuralNetwork

from .CG_Attention import *
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

### Original Attention Layer
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

### MLP variants for Transformer layer    
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


### Main class: The LlamaDecoderLayerWithGNN layer that incorporates the various GNN flavors
class LlamaDecoderLayerWithGNN(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.gnn_config = config.gnn_config
        self.layer_idx = layer_idx
    
        # Initialize self-attention layers based on the configuration
        if self.gnn_config.use_GNN_from_attention in ['LlamaAttention_Original', # original Llama attention (fallback option yields similar behavior, albeit using the LLAMA_ATTENTION_CLASSES functio from the original code)
                                                      'LlamaAttentionGIN', # GIN-Attention
                                                      'LlamaAttentionPNA', # PNA-Attention
                                                      'LlamaAttentionPNA_LM', #PNA_LM-Attention, a variant of PNA
                                                      'CG_Attention', # CG-Attention (CG to latent space, GNN in latent space, decode to sequence space) - uses PerceiverAR_Fixed_Token_Per_Latent
                                                      'CG_Attention_Interpolate', # CG-Attention with interpolation (compute adjacency matrix in latent space, then upscale via interpolation, and GNN in sequence space) - uses PerceiverAR_Fixed_Token_Per_Latent_Scaling
                                                      ]:
            
            if self.gnn_config.use_GNN_from_attention == 'LlamaAttentionGIN':  
                self.self_attn = LlamaAttentionGIN(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttentionPNA': 
                self.self_attn = LlamaAttentionPNA(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttentionPNA_LM': 
                self.self_attn = LlamaAttentionPNA_LM(config=config, layer_idx=layer_idx)
                
            elif self.gnn_config.use_GNN_from_attention == 'CG_Attention': 
                self.self_attn = CG_Attention(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'CG_Attention_Interpolate': 
                self.self_attn = CG_Attention_Interpolate(config=config, layer_idx=layer_idx)

            elif self.gnn_config.use_GNN_from_attention == 'LlamaAttention_Original':
                self.self_attn = LlamaAttention_Original(config=config, layer_idx=layer_idx)

        else:
            print (f"Not found ({self.gnn_config.use_GNN_from_attention}), falling back to standard.")
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        ######## Initialize MLP based on the configuration #######         
        self.skip_around_MLP = True # Usually we use skip around MLP, except for some MLP choices
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
            self.skip_around_MLP = False # No MLP means no skip connection
            print ("No MLP, also removed skip connection.")
        ######## Initialize MLP based on the configuration #######         

        ######## MLP-GIN OPTIONS: Additional aggregrations before FF MLP layer ############
        elif self.gnn_config.MLP_type == 'LlamaMLP_HalfwayGIN':
            self.mlp = LlamaMLP_HalfwayGIN(config)
        elif self.gnn_config.MLP_type == 'LlamaMLP_HalfwayGIN_MultiHop':
            self.mlp = LlamaMLP_MultiHop(config)
            print ("LlamaMLP_MultiHop, does GIN inside MLP, A, A2, A3, multi-hop.")
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
            
            if self.skip_around_MLP:
                hidden_states = residual + mlp_out
            else:
                hidden_states =  mlp_out
            ############ MLP BLOCK with RESIDULAL ##############

        elif self.gnn_config.gnn_logic == 'after_MLP':
            # Apply MLP first, then GNN

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
