# Attention_GNN.py

# Contains three classes:
# - LlamaAttentionGIN: GIN-based attention mechanism.   
# - LlamaAttentionPNA: PNA-based attention mechanism.
# - LlamaAttentionPNA_LM: PNA-based attention mechanism adapted for LM.

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)
from transformers.models.llama.modeling_llama import *

import matplotlib.pyplot as plt

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


def scaled_topk_causal_4d(adj_matrix, sparsity_frac=0.5, threshold=0.0):
    """
    Custom function to apply top-k selection with scaling and thresholding for 4D causal adjacency matrices.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix of shape (batch_size, num_heads, seq_len, seq_len).
                                   Must be causal (upper triangular mask).
        sparsity_frac (float): Fraction of available connections to retain (0 < sparsity_frac <= 1).
        threshold (float): Minimum value for connections to be considered.

    Returns:
        torch.Tensor: Processed adjacency matrix with scaled top-k sparsity and threshold applied.
    """
    # Validate inputs
    if not (0 < sparsity_frac <= 1):
        raise ValueError("`sparsity_frac` must be in the range (0, 1].")

    if adj_matrix.dim() != 4:
        raise ValueError("`adj_matrix` must be a 4D tensor of shape (batch_size, num_heads, seq_len, seq_len).")

    # Get shape information
    batch_size, num_heads, seq_len, _ = adj_matrix.shape

    # Initialize the processed adjacency matrix
    processed_adj = torch.zeros_like(adj_matrix)

    # Iterate over each position in the sequence
    for node_idx in range(1, seq_len):
        # Calculate the dynamic top_k for this node based on sparsity fraction
        available_predecessors = node_idx  # Number of available predecessors
        top_k = max(1, math.ceil(sparsity_frac * available_predecessors))

        # Select valid predecessors for this node (values above the threshold)
        valid_connections = adj_matrix[:, :, node_idx, :node_idx] >= threshold

        # Mask adj_matrix values below the threshold
        filtered_adj = adj_matrix[:, :, node_idx, :node_idx] * valid_connections

        # Select the top-k strongest connections (after applying the threshold)
        _, indices = torch.topk(filtered_adj, top_k, dim=-1)

        # Scatter top-k connections into the processed adjacency matrix
        processed_adj[:, :, node_idx, :node_idx].scatter_(-1, indices, 1.0)

    return processed_adj

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjRMSNorm(nn.Module):
    def __init__(self, eps=1e-8, scalar_weight=True):
        super().__init__()
        self.eps = eps
        self.scalar_weight = scalar_weight
        if scalar_weight:
            # Just one learnable scalar to scale everything.
            self.weight = nn.Parameter(torch.tensor(1.0))
        else:
            # If you need more complicated parameterization, you need to know dims.
            # But for fully flexible shape, stick to a scalar.
            pass

    def forward(self, x):
        # x: (b, ...)
        # We want to normalize across all dimensions except batch.
        # Compute RMS over all non-batch dims:
        # First, compute mean of squares:
        # Reshape x to treat all non-batch dims as features:
        b = x.shape[0]
        feature_dims = x.dim() - 1
        # Flatten all dims except batch into one dimension for mean calculation
        x_flat = x.view(b, -1)
        
        # Mean of squares
        rms = x_flat.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        # rms shape: (b, 1)
        
        # Normalize
        x_norm = x_flat / rms  # broadcast across all features per batch
        x_norm = x_norm.view(*x.shape)  # reshape back to original shape

        if self.scalar_weight:
            x_norm = x_norm * self.weight

        return x_norm

class GINLayer(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, epsilon=0.0):
        super().__init__()
        self.config=config
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.normlayer = LlamaRMSNorm(hidden_dim, eps=self.config.rms_norm_eps)
        dropout_rate = (
            self.config.gnn_config.dropout_rate
            if hasattr(self.config.gnn_config, 'dropout_rate')
            else 0.1
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            self.normlayer,
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)  # project back to input_dim
        )

    def forward(self, x, adjacency):
        # x: (batch, num_nodes, head_dim)
        # adjacency: (batch, num_nodes, num_nodes)
        # GIN aggregation: h^(k) = MLP((1+epsilon)*h^(k-1) + sum_{neighbors} A_ij * h^(k-1)_j)
        neighbor_sum = torch.matmul(adjacency, x)  # (batch, num_nodes, head_dim)
        #out = (1.0 + self.epsilon) * x + neighbor_sum
        out =  self.epsilon * x + neighbor_sum
        out = self.mlp(out)
        return out
    
def reset_threshold_parameters(model, new_value):
    """
    Resets all 'threshold' parameters in the model to the specified value.
    
    Args:
        model: The model containing the 'threshold' parameters.
        new_value (float): The new value to set for each 'threshold' parameter.
    """
    print ("ORIGINAL")
    for name, param in model.named_parameters():
        if "threshold" in name:
            print(f"Name: {name}")
            print(f"Shape: {param.shape}")
            print(f"Value: {param.data}")

    for name, param in model.named_parameters():
        if "threshold" in name:
            with torch.no_grad():
                param.fill_(new_value)
            print(f"Reset {name} to {new_value}")

def gumbel_sigmoid(logits, tau=1.0, eps=1e-10):
    # For a single Bernoulli variable, we can treat this like a 2-class Gumbel-Softmax.
    # logits is (b, L, L) representing log-odds of edge presence.
    # Convert to two-class logits: [logits_no_edge=0, logits_edge=logits].
    logits_edge = logits
    logits_no_edge = torch.zeros_like(logits)  # "no-edge" class
    two_class_logits = torch.stack([logits_no_edge, logits_edge], dim=-1)  # (b, L, L, 2)
    
    U = torch.rand_like(two_class_logits)
    gumbel = -torch.log(-torch.log(U + eps) + eps)
    y = (two_class_logits + gumbel) / tau
    return F.softmax(y, dim=-1)[..., 1]  # return the probability of "edge" class after sampling

def sharp_softplus(x, beta=1.0):
    return F.softplus(beta * x) / beta

class LlamaAttentionGIN(nn.Module):
    """Multi-headed attention replaced by GIN-based aggregation."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # TODO - add multiple GIN attention layer as option
        # self.num_attention_layers = self.config.gnn_config.N_GNN_from_attention_layers
        self.num_attention_layers = 1 #only 1
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        self.diff_attention=  getattr(self.config.gnn_config, "use_differential_attention", False)

        self.use_sharpening = getattr(self.config.gnn_config, "use_sharpening", False)
        
        if self.use_sharpening:        

            sharpening_value_init = getattr(self.config.gnn_config, "sharpening_value_init", "value")
            print ("Use sharpening. Init mode: ", sharpening_value_init )
            
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
                
        if self.diff_attention:

            print ("Use differential attention.")
            self.q_proj_2 = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj_2 = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

            lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (self.layer_idx - 1))
            self.lambda_init=lambda_init

            self.lambda_param = nn.ParameterList([
                nn.Parameter(torch.tensor(lambda_init)) for i in range(self.num_attention_layers)
            ])

            self.diff_attention_group_norm=getattr(self.config.gnn_config, "use_differential_attention_group_norm", False)

            if self.diff_attention_group_norm:
                print ("use_differential_attention_group_norm is TRUE!")
                # Define GroupNorm with num_heads as the groups
                self.groupnorm = nn.GroupNorm(
                    num_groups=self.num_heads,  # Group for each head
                    num_channels=self.num_heads  # Match num_heads
                )

        self.attention_GIN_MLP_o_proj_at_end= getattr(config.gnn_config, "attention_GIN_MLP_o_proj_at_end", True)  #self.config.gnn_config('attention_GIN_MLP_o_proj_at_end', True)
        
        if self.attention_GIN_MLP_o_proj_at_end:
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
            print ("Use o_proj after GIN")
        else:
            print ("Do not use o_proj after GIN")

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # Check for attention_GIN_MLP_multiplier in gnn_config
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_multiplier = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_multiplier = 2


        self.GIN_mode = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_mode", 'default')

        if self.GIN_mode == 'GINLayer_Var_A':
            print ("GIN_mode: GIN_layer_Var_A")
            self.gin_layers = nn.ModuleList([GINLayer_Var_A(config, self.head_dim, int(MLP_multiplier*self.head_dim) ) for _ in range(self.num_heads)])
        else: # 'default':
            print ("Default GIN mode.")
            # GIN layer per head
            self.gin_layers = nn.ModuleList([GINLayer(config, self.head_dim, int(MLP_multiplier*self.head_dim) ) for _ in range(self.num_heads)])
        

        self.use_softmax = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_use_softmax", False)

        if self.use_softmax:
            print ("Use softmax.")
        else:
            print ("Do not use softmax.")
        
        self.use_scoring_fnct=getattr(self.config.gnn_config, "attention_GIN_MLP_use_scoring_fnct", True)
        if self.use_scoring_fnct:
            print ("Do not use standard attention, but instead, scoring function.")

            hidden_dim = self.hidden_size   # Define hidden_dim
            mlp_hidden_dim = getattr(self.config.gnn_config, "attention_GIN_MLP_scoring_hidden_dim", 512)             # Or from config            
            print ("Scoring MLP mlp_hidden_dim: ", mlp_hidden_dim)
            
            self.mlp_scoring = nn.Sequential(
                nn.Linear(2 * hidden_dim, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, self.num_heads)
            )
        else:
            print ("Use standard attention.")

        self.attention_GIN_MLP_GIN_sharp_softplus_beta=getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_sharp_softplus_beta", 10.0)

        ####### parameters for scaling adjacency used for PNA ############
        self.threshold_mode = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_mode", "none")
        threshold = nn.Parameter(torch.tensor(getattr( self.config.gnn_config, "attention_GIN_MLP_GIN_threshold_value", 0.2) ))  # Initialize threshold
        learnable_threshold = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_learnable_threshold", False)
        if learnable_threshold:
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
        #self.blend_alpha = nn.Parameter(torch.tensor(-2.0))

        uniform_value = getattr(self.config.gnn_config, "residual_epsilon_uniform_value", 0.1) #10% PNA
        self.residual_epsilon = nn.ParameterList([
            nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
        ])
    
        self.rmsnorm = AdjRMSNorm()

        self.attention_GIN_MLP_GIN_softmax_temperature=getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_softmax_temperature", 1.0)
        #self.gate_param = nn.Parameter(torch.tensor(-2.))  # Initialize gate around 0.5


        ####### parameters for scaling adjacency used for PNA ############


    def compute_attn_weights_SINGLEHEAD(self, hidden_states, attention_mask):
        """
        Compute attention weights using an MLP-based scoring mechanism.
        
        Args:
            hidden_states (torch.Tensor): shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            attn_weights (torch.Tensor): shape (batch_size, seq_len, seq_len)
                A softmax-normalized adjacency matrix representing attention weights.
        """
        b, L, D = hidden_states.shape
        # Expand to create pairs (i, j)
        h_i = hidden_states.unsqueeze(2)  # (b, L, 1, D)
        h_j = hidden_states.unsqueeze(1)  # (b, 1, L, D)
        # Concatenate token representations pairwise
        pairs = torch.cat([h_i.expand(b, L, L, D), h_j.expand(b, L, L, D)], dim=-1)  # (b, L, L, 2D)
        
        # Apply the MLP scoring
        scores = self.mlp_scoring(pairs).squeeze(-1)  # (b, L, L)
        
        #print ("scores", so)
        if attention_mask is not None:
            # If attention_mask is (b, 1, 1, L), make it (b, L) then (b, L, L)
            attention_mask = attention_mask.squeeze(1).squeeze(1)  # now (b, L)
            #attention_mask = attention_mask.unsqueeze(1).expand(b, L, L)  # now (b, L, L)
            scores = scores + attention_mask
        else:
            causal_mask = torch.tril(torch.ones(L, L, device=scores.device)).unsqueeze(0).unsqueeze(0)
            scores = scores + (1 - causal_mask) * (-1e9)

        # Softmax normalization over the last dimension
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)


        ## SINGLE HEAD....
        attn_weights = attn_weights.unsqueeze(1)  # (b, 1, L, L)
        attn_weights = attn_weights.expand(b, self.num_heads, L, L)  # (b, h, L, L)
        
        #attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype) 
    


        return attn_weights
       
    def compute_attn_weights(self, hidden_states, attention_mask):
        b, L, D = hidden_states.shape
        
        # Expand to create pairs (b, L, L, 2D)
        h_i = hidden_states.unsqueeze(2)  # (b, L, 1, D)
        h_j = hidden_states.unsqueeze(1)  # (b, 1, L, D)
        pairs = torch.cat([h_i.expand(b, L, L, D), h_j.expand(b, L, L, D)], dim=-1)  # (b, L, L, 2D)

        # Now, the mlp_scoring should output (b, L, L, num_heads) instead of (b, L, L, 1).
        # For this, redefine mlp_scoring in __init__:
        #
        # self.mlp_scoring = nn.Sequential(
        #     nn.Linear(2 * hidden_dim, mlp_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_hidden_dim, self.num_heads)  # Note: output dimension = num_heads
        # )
        #
        # This means each pair (i,j) now maps to num_heads scores.
        
        scores = self.mlp_scoring(pairs)  # (b, L, L, h)

        # Rearrange to (b, h, L, L) so we can apply softmax per head easily
        scores = scores.permute(0, 3, 1, 2)  # (b, h, L, L)

        # Handle attention_mask
        # If attention_mask is (b, 1, 1, L), it will broadcast correctly over (b, h, L, L)
        # Just ensure that the mask aligns with (b, h, L, L)
        if attention_mask is not None:
            # Typically attention_mask is (b, 1, 1, L) where non-masked = 0, masked = -inf
            # This will broadcast over h, L as needed
            scores = scores + attention_mask  # Broadcasting should work if mask is (b, 1, 1, L)

        else:
            # Add causal mask if needed
            causal_mask = torch.tril(torch.ones(L, L, device=scores.device))
            causal_mask = (1 - causal_mask) * (-1e9)  # masked positions get -inf
            # causal_mask is (L,L), broadcast to (b,h,L,L)
            scores = scores + causal_mask

        # Softmax along the last dimension (over the second L)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype) 
        # attn_weights is now (b, h, L, L) with a distribution over each token's neighbors per head

        return attn_weights
         
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

        
        if self.threshold_mode == "none":
             
            return adj_matrix

        elif self.threshold_mode == "silu":
            # Apply SiLU-based thresholding
            processed_adj = F.silu(adj_matrix - threshold)

        elif self.threshold_mode == "relu":
            # Apply ReLU-based thresholding
            processed_adj = F.relu(adj_matrix - threshold)

        elif self.threshold_mode == "softplus":
            # Apply softplus-based thresholding
            processed_adj = F.softplus(adj_matrix - threshold)

         
        elif self.threshold_mode == "sharp_softplus":
            # Apply softplus-based thresholding
            #processed_adj = F.softplus(adj_matrix - threshold)

            #beta=10.
            beta=self.attention_GIN_MLP_GIN_sharp_softplus_beta

            processed_adj = sharp_softplus (adj_matrix - threshold, beta)
        
        elif self.threshold_mode == "sigmoid_elementwise":
            # Apply sigmoid element-wise with a threshold
            #beta = 10.0  # Adjust scaling factor (controls steepness of the sigmoid curve)
            beta=self.attention_GIN_MLP_GIN_sharp_softplus_beta

            processed_adj = torch.sigmoid(beta * (adj_matrix - threshold))



        elif self.threshold_mode == "flowthreshold":
            # Apply FlowThreshold-based thresholding
            processed_adj = FlowThreshold(adj_matrix, threshold)

        elif self.threshold_mode == "binary":
            # Apply binary thresholding with binary values (0 or 1)
            processed_adj = (adj_matrix > threshold).float()

        elif self.threshold_mode == "top_k":

            top_k = max (int( adj_matrix.shape[-1] * self.top_k_frac ), 1)
           
            # Retain top-k elements per row with binary values (0 or 1)
            if top_k <= 0:
                raise ValueError("Top-k sparsity mode requires `top_k` to be greater than 0.")
            _, indices = torch.topk(adj_matrix, top_k, dim=-1)
            binary_adj = torch.zeros_like(adj_matrix).scatter_(-1, indices, 1.0)  # Set top-k positions to 1
            processed_adj = binary_adj

        elif self.threshold_mode == "scaled_topk_causal_4d":
           
            # Retain top-k elements per row with binary values (0 or 1)
            processed_adj = scaled_topk_causal_4d(adj_matrix, self.top_k_frac, threshold)
            
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
        
    def apply_groupnorm(self, attn_weights_diff):
        """
        Applies GroupNorm to the attention weights.

        Args:
            attn_weights_diff (torch.Tensor): Tensor of shape (batch, num_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        # Reshape to merge seq_len and seq_len for GroupNorm input
        batch, num_heads, seq_len, seq_len_2 = attn_weights_diff.size()
        attn_weights_diff = attn_weights_diff.reshape(batch, num_heads, -1)  # [batch, num_heads, seq_len * seq_len]

        # Apply GroupNorm
        attn_weights_diff = self.groupnorm(attn_weights_diff)

        # Reshape back to original dimensions
        attn_weights_diff = attn_weights_diff.reshape(batch, num_heads, seq_len, seq_len_2)  # [batch, num_heads, seq_len, seq_len]
        return attn_weights_diff


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  
        key_states = self.k_proj(hidden_states)    
        if self.diff_attention:
            query_states_2 = self.q_proj_2(hidden_states)  
            key_states_2 = self.k_proj_2(hidden_states)    
            query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

       
        value_states = hidden_states#self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

 
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.diff_attention:
            query_states_2, key_states_2 = apply_rotary_pos_emb(query_states_2, key_states_2, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)   

        if self.diff_attention: 
            key_states_2 = repeat_kv(key_states_2, self.num_key_value_groups)  

        value_states = repeat_kv(value_states, self.num_key_value_groups)

        layer_idx=0 #only 1 anyway.... 

        # Apply sharpening if enabled
        if self.use_sharpening:
            alpha = self.sharpening_parameters[layer_idx]
             
        else:
            alpha=1.0

        # If using scoring function, override attn_weights
        if self.use_scoring_fnct:
            attn_weights = self.compute_attn_weights(hidden_states, attention_mask)  # (b, L, L)
            # Expand attn_weights to (b, num_heads, L, L) to match per-head structure
             
            attn_weights = attn_weights.expand(bsz, self.num_heads, q_len, q_len)

            if self.diff_attention:
                print ("Diff attention not implemented for scoring funct.")
  
        else: #conventional attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            
            if self.diff_attention:
                attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                if self.diff_attention:
                    attn_weights_2 = attn_weights_2 + attention_mask
            else:
                causal_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device)).unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + (1 - causal_mask) * (-1e9)
                if self.diff_attention:
                    attn_weights_2 = attn_weights_2 + (1 - causal_mask) * (-1e9)
            
            #self.use_softmax=False
            if self.use_softmax:
                #self.attention_GIN_MLP_GIN_softmax_temperature
                #attn_weights_linear = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                attn_weights = F.softmax(attn_weights*alpha / self.attention_GIN_MLP_GIN_softmax_temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
                if self.diff_attention:
                    attn_weights_2 = F.softmax(attn_weights_2*alpha / self.attention_GIN_MLP_GIN_softmax_temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_weights=attn_weights-self.lambda_param[0]*attn_weights_2  

                    if self.diff_attention_group_norm:
                        attn_weights = (1 - self.lambda_init) * self.apply_groupnorm(attn_weights)
                        attn_weights=attn_weights.tril(0)

                #identity = torch.eye(q_len, device=attn_weights.device).unsqueeze(0).unsqueeze(0)
                #attn_weights = (attn_weights + identity).clamp(min=0, max=1.0)


            else:

                 
                if self.diff_attention:
                    attn_weights=attn_weights-self.lambda_param[0]*attn_weights_2   

            
        
        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        
        layer_idx=0 #only 1 anyway....
        ########### PROCESS ATTN_WEIGHTS ###########
        if self.threshold_mode != 'none':
            
            attn_weights=self.process_adjacency_with_modes(attn_weights, layer_idx)
        else:
            attn_weights = attn_weights
        ########### PROCESS ATTN_WEIGHTS ###########



        if self.config.gnn_config.plot_for_debugging:
                
            # Attention adjacency matrix
            head_adj_mean = attn_weights[:,0,:].cpu().detach().numpy()
             
            # Plot adjacency matrix
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Adjacency Matrix (GPT layer {self.layer_idx}, GIN layer {layer_idx})")
            plt.tight_layout()
             

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_filename = f"./GNN_ATTN_adj_plots_{timestamp}.svg"
            #plt.savefig(combined_filename, format="svg")

            plt.show()

            print ("hidden_states at end: ", hidden_states.shape)


        # Apply GIN per head
        # value_states: (bsz, num_heads, q_len, head_dim)
        head_outputs = []
        for h in range(self.num_heads):
            A = attn_weights[:, h, :, :]  # (bsz, q_len, q_len)
            X = value_states[:, h, :, :]  # (bsz, q_len, head_dim)
            updated_X = self.gin_layers[h](X, A)
            head_outputs.append(updated_X.unsqueeze(1))

         
        
        attn_output = torch.cat(head_outputs, dim=1)  # (bsz, num_heads, q_len, head_dim)

    
        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, -1)       # (bsz, q_len, hidden_size)


        if self.attention_GIN_MLP_o_proj_at_end:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LlamaAttentionPNA(nn.Module):
    """Multi-headed attention replaced by sum, mean, max, and variance aggregators."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # Aggregator MLP for up to 4 aggregators: sum, mean, max, var

        num_aggrgs=4
        #MLP_mult=1

        # Check for attention_GIN_MLP_multiplier in gnn_config
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_mult = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_mult = 2

        agg_input_dim = num_aggrgs * self.head_dim
        
        self.shared_aggregrator_mlp=False
        if self.shared_aggregrator_mlp:
            self.aggregator_mlp = nn.Sequential(
                nn.Linear(agg_input_dim, int(self.head_dim*MLP_mult), bias=config.mlp_bias),
                nn.SiLU(),
                nn.Linear( int(self.head_dim*MLP_mult), self.head_dim, bias=config.mlp_bias)
            )
        else:
            self.aggregator_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear( agg_input_dim,  int(self.head_dim*MLP_mult), bias=config.mlp_bias),
                nn.SiLU(),
                nn.Linear( int(self.head_dim*MLP_mult), self.head_dim, bias=config.mlp_bias)
            ) for _ in range(self.num_heads)
        ])

        ####### parameters for scaling adjacency used for PNA ############
        # Retrieve mode and parameters from config

        self.use_softmax = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_use_softmax", False)
        self.use_ReLU_instead_of_softmax = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_use_ReLU_instead_of_softmax", True)
        if self.use_ReLU_instead_of_softmax:
            print ("self.use_ReLU_instead_of_softmax is True.")
        else:
            print ("self.use_ReLU_instead_of_softmax is False. Must use proper threshold mode.")

        self.num_attention_layers=1 #only 1
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
        #self.blend_alpha = nn.Parameter(torch.tensor(-2.0))

        uniform_value = getattr(self.config.gnn_config, "residual_epsilon_uniform_value", 0.1) #10% PNA
        self.residual_epsilon = nn.ParameterList([
            nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
        ])
        self.attention_GIN_MLP_GIN_softmax_temperature=getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_softmax_temperature", 1.0)

        self.attention_GIN_MLP_GIN_sharp_softplus_beta=getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_sharp_softplus_beta", 10.0)



        ####### parameters for scaling adjacency used for PNA ############

    def _normalize_causal_adjacency(self, A, causal_mask):
        """
        Normalize adjacency matrix A while preserving causal structure.
        A: (batch, heads, seq_len, seq_len)
        causal_mask: binary mask (1 where causal, 0 otherwise)
        """
        # Ensure causality
        A = A * causal_mask

        # Compute node degrees
        deg = A.sum(dim=-1)  # (batch, heads, seq_len)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        deg_inv_sqrt_left = deg_inv_sqrt.unsqueeze(-1)  # (batch, heads, seq_len, 1)
        deg_inv_sqrt_right = deg_inv_sqrt.unsqueeze(-2) # (batch, heads, 1, seq_len)

        # Symmetric normalization
        A_norm = deg_inv_sqrt_left * A * deg_inv_sqrt_right
        A_norm = A_norm * causal_mask
        return A_norm



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

        
        if self.threshold_mode == "none":
             
            return adj_matrix

        elif self.threshold_mode == "silu":
            # Apply SiLU-based thresholding
            processed_adj = F.silu(adj_matrix - threshold)

        elif self.threshold_mode == "relu":
            # Apply ReLU-based thresholding
            processed_adj = F.relu(adj_matrix - threshold)

        elif self.threshold_mode == "softplus":
            # Apply softplus-based thresholding
            processed_adj = F.softplus(adj_matrix - threshold)

        elif self.threshold_mode == "flowthreshold":
            # Apply FlowThreshold-based thresholding
            processed_adj = FlowThreshold(adj_matrix, threshold)
            
        elif self.threshold_mode == "sharp_softplus":
            # Apply softplus-based thresholding
            #processed_adj = F.softplus(adj_matrix - threshold)

            #beta=10.
            beta=self.attention_GIN_MLP_GIN_sharp_softplus_beta

            processed_adj = sharp_softplus (adj_matrix - threshold, beta)
        
        elif self.threshold_mode == "sigmoid_elementwise":
            # Apply sigmoid element-wise with a threshold
            beta = 10.0  # Adjust scaling factor (controls steepness of the sigmoid curve)
            processed_adj = torch.sigmoid(beta * (adj_matrix - threshold))

        elif self.threshold_mode == "binary":
            # Apply binary thresholding with binary values (0 or 1)
            processed_adj = (adj_matrix > threshold).float()

        elif self.threshold_mode == "top_k":

            top_k = max (int( adj_matrix.shape[-1] * self.top_k_frac ), 1)
           
            # Retain top-k elements per row with binary values (0 or 1)
            if top_k <= 0:
                raise ValueError("Top-k sparsity mode requires `top_k` to be greater than 0.")
            _, indices = torch.topk(adj_matrix, top_k, dim=-1)
            binary_adj = torch.zeros_like(adj_matrix).scatter_(-1, indices, 1.0)  # Set top-k positions to 1
            processed_adj = binary_adj

        elif self.threshold_mode == "scaled_topk_causal_4d":
           
            # Retain top-k elements per row with binary values (0 or 1)
            processed_adj = scaled_topk_causal_4d(adj_matrix, self.top_k_frac, threshold)
            
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
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (bsz, num_heads, q_len, q_len)

        layer_idx = 0 #there is only 1 GNN layer!

        if attention_mask is not None:
            causal_mask = (attention_mask == 0).type_as(attn_weights)
            attn_weights = attn_weights + attention_mask
        else:
            # Full causal mask if none provided
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + (1 - causal_mask) * (-1e9)

        
        #self.use_softmax=False
        if self.use_softmax:
            #attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.softmax(attn_weights / self.attention_GIN_MLP_GIN_softmax_temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)

            

        else:
            #causal_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device)).unsqueeze(0).unsqueeze(0)
            #attn_weights = attn_weights * causal_mask
            if self.use_ReLU_instead_of_softmax:
                attn_weights = F.relu(attn_weights)



        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)

        

        ########### PROCESS ATTN_WEIGHTS ###########
        if self.threshold_mode != 'none':
            
            attn_weights_processed = attn_weights

            # Add self-loops to GIN adjacency matrix
            #identity = torch.eye(q_len, device=attn_weights_processed.device).unsqueeze(0).unsqueeze(0)
            #attn_weights_processed = (attn_weights_processed + 0.1*identity).clamp(min=0, max=1.0)

            attn_weights_processed=self.process_adjacency_with_modes(attn_weights_processed, layer_idx)
        else:
            attn_weights_processed = attn_weights
        ########### PROCESS ATTN_WEIGHTS ###########

        if self.config.gnn_config.plot_for_debugging:
                
             
            head_adj_mean = attn_weights_processed[:,0,:].cpu().detach().numpy()
             
            # Plot adjacency matrix
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Adjacency Matrix (GPT layer {self.layer_idx}, GIN layer {layer_idx})")
            plt.tight_layout()
             

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_filename = f"./GNN_ATTN_adj_plots_{timestamp}.svg"
            #plt.savefig(combined_filename, format="svg")

            plt.show()

            print ("hidden_states at end: ", hidden_states.shape)



        # value_states: (bsz, num_heads, q_len, head_dim)
        head_outputs = []
        # Compute deg for mean/variance
        #deg = attn_weights.sum(dim=-1)  # (b, h, s)
        deg = attn_weights_processed.sum(dim=-1)  # (b, h, s)
        deg_clamped = deg.clone()
        deg_clamped[deg_clamped == 0] = 1.0  # avoid division by zero



        for h in range(self.num_heads):
            A = attn_weights[:, h, :, :]  # (b, s, s)
            X = value_states[:, h, :, :]  # (b, s, head_dim)
            deg_h = deg_clamped[:, h]     # (b, s)

            # sum_agg = A X
            sum_agg = torch.matmul(A, X)  # (b, s, head_dim) 


            ##############
            # Here, from now on, use processed attn_weights... we leave the original for sum since we want to make sure they are all used as in conventional attention
            A = attn_weights_processed[:, h, :, :]  # (b, s, s)
            #re-calculate sum based on rocessed weights
            sum_agg = torch.matmul(A, X)  # (b, s, head_dim)
            ###########                    

            # mean_agg = (A X) / deg
            mean_agg = sum_agg / deg_h.unsqueeze(-1)
  
            # max_agg: mask invalid neighbors with -inf
            valid_mask = (A > 0)  # (b, s, s)
            X_expanded = X.unsqueeze(2).expand(-1, -1, X.size(1), -1) # (b, s, s, head_dim)

            valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, -1, X.size(-1))  # match head_dim
            masked_max = X_expanded.clone()
            masked_max[~valid_mask_expanded] = float('-inf')
            max_agg, _ = torch.max(masked_max, dim=2)

            ### MIN AGGGREGATOR - NOT CURRENTLY USED
            masked_min = X_expanded.clone()
            masked_min[~valid_mask_expanded] = float('inf')
            min_agg, _ = torch.min(masked_min, dim=2)       
            ### END MIN AGGREG     

            # variance aggregator:
            # var = mean_of_squares - (mean_aggÂ²)
            X_squares = X**2
            sum_of_squares = torch.matmul(A, X_squares)     # (b, s, head_dim)
            mean_of_squares = sum_of_squares / deg_h.unsqueeze(-1)
            var_agg = mean_of_squares - (mean_agg**2)
            var_agg = torch.clamp(var_agg, min=0.0)

            # Combine [sum, mean, max, var]
            combined = torch.cat([sum_agg, mean_agg, 
                                  max_agg, var_agg
                                  ], dim=-1)  # (b, s, 4*head_dim)

            if self.shared_aggregrator_mlp:
                h_out = self.aggregator_mlp(combined)  # (b, s, head_dim)
            else:
                h_out = self.aggregator_mlps[h](combined)
            head_outputs.append(h_out.unsqueeze(1))

        attn_output = torch.cat(head_outputs, dim=1)  # (b, h, s, head_dim)

        #blend_ratio =  self.residual_epsilon[layer_idx].clamp(min=0, max=1.0)
        #attn_output=(1 - blend_ratio) * original_attn_output + blend_ratio * attn_output

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1) # (b, s, hidden_size)
        attn_output = self.o_proj(attn_output)
 
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LlamaAttentionPNA_LM(nn.Module):
    """Multi-headed attention replaced by sum, mean, max, and variance aggregators."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # Aggregator MLP for up to 4 aggregators: sum, max, var, min

        num_aggrgs=4

        # Check for attention_GIN_MLP_multiplier in gnn_config
        if hasattr(self.config, 'gnn_config') and hasattr(self.config.gnn_config, 'attention_GIN_MLP_multiplier'):
            MLP_mult = self.config.gnn_config.attention_GIN_MLP_multiplier
        else:
            MLP_mult = 2

        agg_input_dim = num_aggrgs * self.head_dim
        
        self.shared_aggregrator_mlp=False
        if self.shared_aggregrator_mlp:
            self.aggregator_mlp = nn.Sequential(
                nn.Linear(agg_input_dim, self.head_dim*MLP_mult, bias=config.mlp_bias),
                nn.SiLU(),
                nn.Linear(self.head_dim*MLP_mult, self.head_dim, bias=config.mlp_bias)
            )
        else:
            self.aggregator_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(agg_input_dim, self.head_dim*MLP_mult, bias=config.mlp_bias),
                nn.SiLU(),
                nn.Linear(self.head_dim*MLP_mult, self.head_dim, bias=config.mlp_bias)
            ) for _ in range(self.num_heads)
        ])

        ####### parameters for scaling adjacency used for PNA ############
        self.use_softmax = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_use_softmax", False)
        self.num_attention_layers=1 #only 1
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
            self.threshold = [
                    torch.tensor(threshold, dtype=torch.float32) for _ in range(self.num_attention_layers)
                ]

        self.binary_scale = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_binary_scale", 1.0)
        self.top_k_frac = getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length", 0.1)  #%10 of sequence length is used
        
        uniform_value = getattr(self.config.gnn_config, "residual_epsilon_uniform_value", 0.1) #10% PNA
        self.residual_epsilon = nn.ParameterList([
            nn.Parameter(torch.tensor(uniform_value)) for _ in range(self.num_attention_layers)
        ])
        self.attention_GIN_MLP_GIN_softmax_temperature=getattr(self.config.gnn_config, "attention_GIN_MLP_GIN_softmax_temperature", 1.0)
        ####### parameters for scaling adjacency used for PNA ############

    def _normalize_causal_adjacency(self, A, causal_mask):
        """
        Normalize adjacency matrix A while preserving causal structure.
        A: (batch, heads, seq_len, seq_len)
        causal_mask: binary mask (1 where causal, 0 otherwise)
        """
        # Ensure causality
        A = A * causal_mask

        # Compute node degrees
        deg = A.sum(dim=-1)  # (batch, heads, seq_len)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        deg_inv_sqrt_left = deg_inv_sqrt.unsqueeze(-1)  # (batch, heads, seq_len, 1)
        deg_inv_sqrt_right = deg_inv_sqrt.unsqueeze(-2) # (batch, heads, 1, seq_len)

        # Symmetric normalization
        A_norm = deg_inv_sqrt_left * A * deg_inv_sqrt_right
        A_norm = A_norm * causal_mask
        return A_norm



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

        
        if self.threshold_mode == "none":
             
            return adj_matrix

        elif self.threshold_mode == "silu":
            # Apply SiLU-based thresholding
            processed_adj = F.silu(adj_matrix - threshold)

        elif self.threshold_mode == "relu":
            # Apply ReLU-based thresholding
            processed_adj = F.relu(adj_matrix - threshold)

        elif self.threshold_mode == "softplus":
            # Apply softplus-based thresholding
            processed_adj = F.softplus(adj_matrix - threshold)

        elif self.threshold_mode == "flowthreshold":
            # Apply FlowThreshold-based thresholding
            processed_adj = FlowThreshold(adj_matrix, threshold)
        elif self.threshold_mode == "sharp_softplus":
            # Apply softplus-based thresholding
            #processed_adj = F.softplus(adj_matrix - threshold)

            beta=10.
            processed_adj = sharp_softplus (adj_matrix - threshold, beta)
        
        elif self.threshold_mode == "sigmoid_elementwise":
            # Apply sigmoid element-wise with a threshold
            beta = 10.0  # Adjust scaling factor (controls steepness of the sigmoid curve)
            processed_adj = torch.sigmoid(beta * (adj_matrix - threshold))

        elif self.threshold_mode == "binary":
            # Apply binary thresholding with binary values (0 or 1)
            processed_adj = (adj_matrix > threshold).float()

        elif self.threshold_mode == "top_k":

            top_k = max (int( adj_matrix.shape[-1] * self.top_k_frac ), 1)
           
            # Retain top-k elements per row with binary values (0 or 1)
            if top_k <= 0:
                raise ValueError("Top-k sparsity mode requires `top_k` to be greater than 0.")
            _, indices = torch.topk(adj_matrix, top_k, dim=-1)
            binary_adj = torch.zeros_like(adj_matrix).scatter_(-1, indices, 1.0)  # Set top-k positions to 1
            processed_adj = binary_adj

        elif self.threshold_mode == "scaled_topk_causal_4d":
           
            # Retain top-k elements per row with binary values (0 or 1)
            processed_adj = scaled_topk_causal_4d(adj_matrix, self.top_k_frac, threshold)
            
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
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (bsz, num_heads, q_len, q_len)

        layer_idx = 0 # There is only 1 GNN layer!
        threshold_value = self.threshold[layer_idx].to(attn_weights.device)

        if attention_mask is not None:
            causal_mask = (attention_mask == 0).type_as(attn_weights)
            attn_weights = attn_weights + attention_mask
        else:
            # Full causal mask if none provided
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=attn_weights.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + (1 - causal_mask) * (-1e9)

        if self.use_softmax:
            attn_weights = F.softmax(attn_weights / self.attention_GIN_MLP_GIN_softmax_temperature, dim=-1, dtype=torch.float32).to(query_states.dtype)
        else:
            attn_weights = F.relu(attn_weights)

        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)

        # Process adjacency if threshold_mode != none
        if self.threshold_mode != 'none':
            attn_weights_processed = self.process_adjacency_with_modes(attn_weights, layer_idx)
        else:
            attn_weights_processed = attn_weights

        if self.config.gnn_config.plot_for_debugging:
            head_adj_mean = attn_weights_processed[:,0,:].cpu().detach().numpy()
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(head_adj_mean[0], cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention Weight")
            plt.title(f"Adjacency Matrix (GPT layer {self.layer_idx}, GIN layer {layer_idx})")
            plt.tight_layout()
            plt.show()

        # value_states: (bsz, num_heads, q_len, head_dim)
        head_outputs = []
        # Compute deg for mean/variance
        deg = attn_weights_processed.sum(dim=-1)  # (b, h, s)
        deg_clamped = deg.clone()
        deg_clamped[deg_clamped == 0] = 1.0  # avoid division by zero

        for h in range(self.num_heads):
            A = attn_weights_processed[:, h, :, :]  # (b, s, s)
            X = value_states[:, h, :, :]            # (b, s, head_dim)
            deg_h = deg_clamped[:, h]               # (b, s)

            # sum_agg = A X
            sum_agg = torch.matmul(A, X)  # (b, s, head_dim)

            # mean_agg = (A X) / deg
            mean_agg = sum_agg / deg_h.unsqueeze(-1)

            # variance aggregator:
            X_squares = X**2
            sum_of_squares = torch.matmul(A, X_squares)     # (b, s, head_dim)
            mean_of_squares = sum_of_squares / deg_h.unsqueeze(-1)
            var_agg = mean_of_squares - (mean_agg**2)
            var_agg = torch.clamp(var_agg, min=0.0)

            # Vector-level max/min aggregator:
            # Use threshold_value instead of 0
            valid_mask = (A > threshold_value)  # (b, s, s)
            norms = torch.norm(X, dim=-1)  # (b, s)
            expanded_norms = norms.unsqueeze(1).expand(-1, X.size(1), -1)  # (b, s, s)

            # MAX aggregator based on largest norm among valid neighbors
            masked_norms_max = expanded_norms.clone()
            masked_norms_max[~valid_mask] = float('-inf')
            max_indices = masked_norms_max.argmax(dim=-1)  # (b, s)
            max_agg = torch.gather(X, 1, max_indices.unsqueeze(-1).expand(-1, -1, X.size(-1)))  # (b, s, head_dim)

            # MIN aggregator based on smallest norm among valid neighbors
            masked_norms_min = expanded_norms.clone()
            masked_norms_min[~valid_mask] = float('inf')
            min_indices = masked_norms_min.argmin(dim=-1)  # (b, s)
            min_agg = torch.gather(X, 1, min_indices.unsqueeze(-1).expand(-1, -1, X.size(-1)))  # (b, s, head_dim)

            # Handle deg == 0 case by using node’s own embedding
            no_neighbors_mask = (deg[:, h] == 0)  # (b, s)
            no_neighbors_mask_expanded = no_neighbors_mask.unsqueeze(-1)

            max_agg = torch.where(no_neighbors_mask_expanded, X, max_agg)
            min_agg = torch.where(no_neighbors_mask_expanded, X, min_agg)

            # Combine [sum, mean, max, var, min]
            combined = torch.cat([#sum_agg, 
                                  mean_agg, 
                                  max_agg, var_agg, min_agg], dim=-1)  # (b, s, 5*head_dim)

            if self.shared_aggregrator_mlp:
                h_out = self.aggregator_mlp(combined)  # (b, s, head_dim)
            else:
                h_out = self.aggregator_mlps[h](combined)
            head_outputs.append(h_out.unsqueeze(1))

        attn_output = torch.cat(head_outputs, dim=1)  # (b, h, s, head_dim)
        blend_ratio = self.residual_epsilon[layer_idx].clamp(min=0, max=1.0)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1) # (b, s, hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
