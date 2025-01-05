#CG_Attention.py

# Defines two main classes used outside
# 1. CG_Attention: CG (encode, attention, decode) module
# 2. CG_Attention_Interpolate: Interpolates between supplied attention and local attention

# Several helper classes
# 1. CrossAttention: CrossAttention  with additional causal_mask; LatentTransformerLayer: Transformer layer for latents
# 2. CrossAttentionPlus: CrossAttention with additional parameters (supplied_attn, supplied_attn_mix), LatentTransformerLayerPlus: Transformer layer for latents with attention matrix (uses CrossAttentionPlus) 

# Additional attention/transformer classes
# 1. PerceiverAR_Fixed_Token_Per_Latent: PerceiverAR with fixed number of tokens per latent, used in CG_Attention, uses CrossAttention
# 2. PerceiverAR_Fixed_Token_Per_Latent_Scaling: PerceiverAR with fixed number of tokens per latent and scaling CG_Attention_Interpolate, uses CrossAttentionPlus

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import *
from .CG_Attention import * 
import matplotlib.pyplot as plt
from copy import deepcopy
#import numpy as np
class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # Full causal mask
        causal_mask: Optional[torch.Tensor] = None      # Additional causal constraint
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project to q, k, v
        q = self.to_q(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.to_k(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        v = self.to_v(value).view(batch_size, key_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            
            attention_mask = attention_mask.expand(-1, scores.size(1), -1, -1)
            scores = scores + attention_mask

        # Apply any additional causal constraints
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, -1e9)

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return self.to_out(out)
    
class LatentTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CrossAttention(dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            causal_mask=causal_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x
      
### Class to encode/decode with fixed number of tokens per latent, inspired by PerceiverAR 
class PerceiverAR_Fixed_Token_Per_Latent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Position embedding configurations
        self.max_position_embeddings = getattr(config.gnn_config, "max_position_embeddings", 2048)
        self.num_latents = getattr(config.gnn_config, "num_latents", 32)
        
        # Fixed number of tokens per latent (new option)
        self.use_fixed_number_of_tokens_per_latent = getattr(config.gnn_config, "use_fixed_number_of_tokens_per_latent", False)
        self.tokens_per_latent = getattr(config.gnn_config, "tokens_per_latent", self.max_position_embeddings // self.num_latents)
        
        # Coarse-graining option
        self.group_tokens_for_coarse_graining = getattr(config.gnn_config, "group_tokens_for_coarse_graining", False)
        
        max_tokens_assignable = self.tokens_per_latent * self.num_latents

        print ("Option: use_fixed_number_of_tokens_per_latent=", self.use_fixed_number_of_tokens_per_latent)
        print ("tokens_per_latent", self.tokens_per_latent, "max tokens assignable: ", max_tokens_assignable)

        if self.max_position_embeddings > max_tokens_assignable:
            raise ValueError(
                f"max_position_embeddings ({self.max_position_embeddings}) exceeds the assignable tokens "
                f"({max_tokens_assignable}) when using fixed tokens per latent."
            )
        
        # Input position embeddings for the full sequence
        self.input_pos_emb = nn.Parameter(
            torch.randn(1, self.max_position_embeddings, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        # Latent position embeddings
        self.latent_pos_emb = nn.Parameter(
            torch.randn(1, self.num_latents, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        # Learnable latent array
        self.latent_array = nn.Parameter(
            torch.randn(1, self.num_latents, self.hidden_size) / math.sqrt(self.hidden_size)
        )

        # Decoder queries
        self.decoder_queries = nn.Parameter(
            torch.randn(1, self.max_position_embeddings, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        self.group_tokens_for_coarse_graining =  getattr(config.gnn_config, "group_tokens_for_coarse_graining", False)


        # Layers
        self.input_to_latent = CrossAttention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.attention_dropout
        )
        
        num_latent_layers = getattr(config.gnn_config, "num_latent_layers", 6)
        self.latent_transformer = nn.ModuleList([
            LatentTransformerLayer(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout=config.attention_dropout
            ) for _ in range(num_latent_layers)
        ])
        
        self.latent_to_output = CrossAttention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.attention_dropout
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.final_norm = nn.LayerNorm(self.hidden_size)
 

    def create_causal_mask(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask ensuring strict causality.
        """
        mask = torch.triu(
            torch.ones(query_len, key_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum allowed "
                f"length ({self.max_position_embeddings})"
            )
        
        # Add position embeddings
        input_pos = self.input_pos_emb[:, :seq_len, :]
        hidden_states = hidden_states + input_pos
        
        # Prepare latent array
        latent_pos = self.latent_pos_emb
        latents = self.latent_array + latent_pos
        latents = latents.expand(batch_size, -1, -1)

        # Create a combined attention and causal mask for input-to-latent attention
        if attention_mask is not None:
            # Pad the attention mask to match latent dimensions
            if attention_mask.shape[-2] < self.num_latents or attention_mask.shape[-1] < seq_len:
                padded_attention_mask = torch.full(
                    (attention_mask.size(0), attention_mask.size(1), self.num_latents, seq_len),
                    -1e9,
                    device=device,
                    dtype=attention_mask.dtype,
                )
                
                padded_attention_mask[..., :attention_mask.size(-2), :attention_mask.size(-1)] = attention_mask
                
                attention_mask = padded_attention_mask

        
        
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum allowed "
                f"length ({self.max_position_embeddings})"
            )

        # Add position embeddings
        input_pos = self.input_pos_emb[:, :seq_len, :]
        hidden_states = hidden_states + input_pos

        # Prepare latent array
        latent_pos = self.latent_pos_emb
        latents = self.latent_array + latent_pos
        latents = latents.expand(batch_size, -1, -1)

        
        # Fixed number of tokens per latent
        tokens_per_latent = getattr(self.config.gnn_config, "tokens_per_latent", self.max_position_embeddings // self.num_latents)
        starts = torch.arange(0, seq_len, tokens_per_latent, device=device)
        ends = torch.clamp(starts + tokens_per_latent - 1, max=seq_len - 1)

        #num_active_latents = len(starts)  # Number of latents required for the sequence length

        self.token_to_latent_mapping = {
            'seq_len': seq_len,
            'starts': starts,
            'ends': ends
        }

        # Initialize group mask for fixed tokens per latent
        group_mask = torch.ones(batch_size, 1, self.num_latents, seq_len, device=device).bool()  # Always use all latents

        for i, (start, end) in enumerate(zip(starts, ends)):
            group_mask[:, :, i, start:end + 1] = False  # Mask only valid ranges for active latents

        # Mask unused latents
        #for i in range(num_active_latents, self.num_latents):
        #    group_mask[:, :, i, :] = True  # Fully mask unused latents

        if self.config.gnn_config.plot_for_debugging:
            print ("group_mask Encode", group_mask.shape)
            print (group_mask)

            mask_plot = group_mask[0,0,:].cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_plot, cmap="viridis", aspect="auto")
            plt.colorbar(label="Mask")
            plt.title(f"mask_encode, mask_plot: {mask_plot.shape}")
            plt.tight_layout()
            plt.show()
 

        group_mask_additive = group_mask.float() * -1e9

        #print ("group_mask_additive", group_mask_additive.shape, "group_mask", group_mask.shape, "attention_mask", attention_mask.shape)

        # Adjust attention_mask dimensions
        if attention_mask is not None:
            padded_attention_mask = torch.full(
                (attention_mask.size(0), attention_mask.size(1), self.num_latents, seq_len),
                -1e9,
                device=device,
                dtype=attention_mask.dtype,
            )
            padded_attention_mask[:, :, :self.num_latents, :seq_len] = attention_mask [:, :, :self.num_latents, :seq_len]  # Use all latents
            attention_mask = padded_attention_mask + group_mask_additive
        else:
            attention_mask = group_mask_additive 

        # Input to latent cross-attention
        latents = self.input_to_latent(
            query=self.norm1(latents),
            key=self.norm2(hidden_states),
            value=self.norm2(hidden_states),
            attention_mask=attention_mask
        )
        
        # Create strictly causal mask for latent processing
        latent_causal_mask = torch.triu(
            torch.ones(self.num_latents, self.num_latents, dtype=torch.bool, device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Create new attention mask for latents
        new_attention_mask = torch.triu(
            torch.full((self.num_latents, self.num_latents), -1e9, device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Process through transformer layers
        for layer in self.latent_transformer:
            latents = layer(latents, causal_mask=latent_causal_mask)
                
        return self.final_norm(latents), new_attention_mask

    def decode(self, latents: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = latents.shape[0]
        device = latents.device
        sequence_len = self.token_to_latent_mapping['seq_len']

        # Use learned queries for the sequence
        queries = self.decoder_queries[:, :sequence_len, :].expand(batch_size, -1, -1)

        # Initialize group mask
        #group_mask = torch.ones(batch_size, 1, sequence_len, self.num_latents, device=device).bool()

        starts = self.token_to_latent_mapping['starts']  # Contains all the token-to-latent mapping info
        ends = self.token_to_latent_mapping['ends']

        #print ("starts: ", starts, "ends: ", ends)
        group_mask = torch.ones(batch_size, 1, sequence_len, self.num_latents, #len(starts), 
                                device=device).bool()

        # Create mask in one go
        token_pos = torch.arange(sequence_len, device=device)[:, None]  # [seq_len, 1]
        starts = starts[None, :]  # [1, num_latents]
        ends = ends[None, :]      # [1, num_latents]
        
        # Mask for active latents
        mask = (token_pos >= starts) & (token_pos <= ends)  # [seq_len, len(starts)]
 

        # Add batch and singleton dimensions, expand to match group_mask
        batch_size = group_mask.size(0)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)  # [batch_size, 1, seq_len, len(starts)]



        # Debug shapes
        #print("mask", mask.shape, "group_mask", group_mask.shape, "starts", starts.shape)


        # Assign mask to group_mask
        group_mask[:, :, :, :starts.shape[1]] = ~mask  # Explicitly assign to the latent dimension

        #debug=False
        if self.config.gnn_config.plot_for_debugging:
            print ("group_mask Decode", group_mask.shape)
            print (group_mask)

            mask_plot = group_mask[0,0,:].cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_plot, cmap="viridis", aspect="auto")
            plt.colorbar(label="Mask")
            plt.title(f"mask_decode, mask_plot: {mask_plot.shape}")
            plt.tight_layout()
            plt.show()
    
  
        # Fully mask unused latents
        #if len(starts) < self.num_latents:
        #    group_mask[...:, len(starts):] = True  # Mask unused latent dimensions
        


        # Convert to additive mask
        group_mask_additive = group_mask.float() * -1e9


        # Combine with provided attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :sequence_len, :self.num_latents]
            padded_attention_mask = torch.full(
                (attention_mask.size(0), attention_mask.size(1), sequence_len, self.num_latents),
                -1e9,
                device=device,
                dtype=attention_mask.dtype
            )
            padded_attention_mask[..., :attention_mask.size(-2), :attention_mask.size(-1)] = attention_mask
            attention_mask = padded_attention_mask + group_mask_additive
        else:
            attention_mask = group_mask_additive

        # Perform latent-to-output attention
        output = self.latent_to_output(
            query=queries,
            key=latents,
            value=latents,
            attention_mask=attention_mask
        )

        return output
        #return self.final_norm(output)
    
    
### CG_Attention ####

class CG_Attention (nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
         
        self.config = config
        self.num_latents_list = getattr(self.config.gnn_config, "num_latents_list", [64, 32, 8])
        self.use_skip = getattr(self.config.gnn_config, "CG_Attention_use_skip", True)
        print(f"Hierarchical latents in CG_Attention {layer_idx}: ", self.num_latents_list)
        
        # Number of hierarchical levels
        self.num_hierarchy_levels = len(self.num_latents_list)

        self.use_fixed_number_of_tokens_per_latent = True

        if self.use_fixed_number_of_tokens_per_latent:
            print ("Use fixed number of tokens per latent")
            self.perceiver_layers = nn.ModuleList([
                PerceiverAR_Fixed_Token_Per_Latent(self._create_layer_config(config, num_latents, idx))
                for idx, num_latents in enumerate(self.num_latents_list)
            ])

        # Projection layers for each hierarchical level
        self.level_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(self.num_hierarchy_levels)
        ])

    def _create_layer_config(self, base_config, num_latents, layer_idx):
        """
        Create a layer-specific configuration for the PerceiverAR.
        """
        layer_config = deepcopy(base_config)
        layer_config.gnn_config.num_latents = num_latents
        layer_config.layer_idx = layer_idx
        return layer_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for hierarchical attention with PerceiverAR.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size, seq_len, hidden_dim).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output hidden states after hierarchical processing.
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        # Store original sequence length for potential truncation
        original_seq_len = seq_len
        
        # Store reconstructed sequences from each level
        reconstructed_sequences = []

        # Process through each hierarchical level
        for perceiver, projection in zip(self.perceiver_layers, self.level_projections):
            # Encode sequence to latent representation using the original sequence
            latent_states, _ = perceiver.encode(
                hidden_states,  # Always use the original sequence
                attention_mask=attention_mask  # Always use the original attention mask
            )
            
            # Decode back to sequence length using the original sequence
            decoded_states = perceiver.decode(
                latent_states,
                attention_mask=attention_mask  # Always use the original attention mask
            )
            
            # Truncate if necessary
            if decoded_states.size(1) > original_seq_len:
                decoded_states = decoded_states[:, :original_seq_len, :]
            
            # Project decoded states
            projected_states = projection(decoded_states)

            #projected_states = decoded_states
            
            # Add skip connection with original input
            if self.use_skip:
                skip_connected_states = hidden_states + projected_states
            else:
                skip_connected_states = projected_states #no skip

            # Store reconstruction with skip connection
            reconstructed_sequences.append(skip_connected_states)

        # Sum up all reconstructed sequences (hierarchical integration)
        integrated_output = sum(reconstructed_sequences)

        # Add skip connection with the original input
        #integrated_output = integrated_output + hidden_states

        return integrated_output, None, None


####################################################################################
### Scale adjancency matrix based on latent attention ###
### TODO: test and debug
### Not tested yet, use with caution
####################################################################################
class CrossAttentionPlus(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1, 
                 normalize_after_mixing=True,
                 #to_out_dim: bool=None
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        
        to_out_dim=num_heads * head_dim
        self.to_out = nn.Linear(to_out_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.normalize_after_mixing = normalize_after_mixing,
        
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        supplied_attn: Optional[torch.Tensor] = None,  # New: Supplied attention matrix
        supplied_attn_mix: float=0., #if >0, mixes supplied_attn with attention generated "locally" usning its own q/k
        return_attention: bool = False,  # Handle return_attention flag
    ) -> torch.Tensor:
        
        #in encoding cross attention: key/value = original sequence, query = latent

        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project to q, k, v
        q = self.to_q(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.to_k(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        v = self.to_v(value).view(batch_size, key_len, self.num_heads, self.head_dim)

        #print ("v", v.shape)

        # Transpose for attention
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        # Attention calculation
        
        if supplied_attn is None or supplied_attn_mix > 0.0: # either way, we need to calculate local attention sclaes for mixing
            # Calculate attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply masks
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, -1e9)
            
            if attention_mask is not None:
    
                # Directly use the attention mask for latent-to-latent or output attention
                attention_mask = attention_mask.expand(-1, scores.size(1), -1, -1)
                scores = scores + attention_mask

            # Apply softmax to obtain attention
            local_attn = F.softmax(scores, dim=-1)

        # Mix with supplied attention if needed
        if supplied_attn is not None:
            if supplied_attn_mix>0:

                attn = supplied_attn_mix * local_attn + (1 - supplied_attn_mix) * supplied_attn
                
                # Normalize rows to ensure a valid probability distribution
                if self.normalize_after_mixing:
                    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9) #ensures that even if the sum of a row is 0.0, the denominator will never actually be zero, avoiding division-by-zero errors.

                    if causal_mask is not None:
                        attn = attn.masked_fill(causal_mask, 0) #make sure normalized attn is causal
            else:
                attn = supplied_attn  # Fully use supplied attention
        else:
            attn = local_attn  # Fully use locally computed attention

        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
            
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
    
        if return_attention:
            return self.to_out(out), attn

        else:
            return self.to_out(out), None
            

class LatentTransformerLayerPlus(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CrossAttentionPlus(dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, adj_matrix = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            causal_mask=causal_mask,
            return_attention=True,
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, adj_matrix

def visualize_two_adj_matrices_torch(matrix1, matrix2):
    """
    Visualize and compare two adjacency matrices (PyTorch tensors), averaging over the `head` dimension
    and plotting the 0th batch dimension for both matrices.

    Parameters:
        matrix1 (torch.Tensor): First adjacency matrix with shape (batch, head, x, y).
        matrix2 (torch.Tensor): Second adjacency matrix with shape (batch, head, x, y).
    """
    # Ensure matrices have the correct dimensions
    if matrix1.ndim != 4 or matrix2.ndim != 4:
        raise ValueError("Input matrices must have shape (batch, head, x, y).")

    # Convert tensors to NumPy arrays and average over the head dimension
    avg_matrix1 = matrix1[0].mean(dim=0).numpy()  # Take 0th index of batch and average over head
    avg_matrix2 = matrix2[0].mean(dim=0).numpy()  # Take 0th index of batch and average over head

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot first matrix
    im1 = axes[0].imshow(avg_matrix1, cmap="Blues", interpolation="nearest")
    axes[0].set_title("Original (Averaged over Head)")
    plt.colorbar(im1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)

    # Plot second matrix
    im2 = axes[1].imshow(avg_matrix2, cmap="Blues", interpolation="nearest")
    axes[1].set_title("Scaled (Averaged over Head)")
    plt.colorbar(im2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"./adj_matrix_comparison_{timestamp}.svg"
    plt.savefig(file_path, format="svg")
    plt.show()

class PerceiverAR_Fixed_Token_Per_Latent_Scaling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Position embedding configurations
        self.max_position_embeddings = getattr(config.gnn_config, "max_position_embeddings", 2048)
        self.num_latents = getattr(config.gnn_config, "num_latents", 32)
        self.interpolation_method_for_adj_scaling = getattr(config.gnn_config, "interpolation_method_for_adj_scaling", 'nearest'
                                                            #"bilinear", 
                                                            )
        print ("interpolation_method_for_adj_scaling: ", self.interpolation_method_for_adj_scaling)
        
        if self.num_latents >= self.max_position_embeddings:
            raise ValueError(
                f"num_latents ({self.num_latents}) must be smaller than "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )
        
        # Input position embeddings for the full sequence
        self.input_pos_emb = nn.Parameter(
            torch.randn(1, self.max_position_embeddings, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        # Latent position embeddings
        self.latent_pos_emb = nn.Parameter(
            torch.randn(1, self.num_latents, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        # Learnable latent array
        self.latent_array = nn.Parameter(
            torch.randn(1, self.num_latents, self.hidden_size) / math.sqrt(self.hidden_size)
        )

        # Decoder queries
        self.decoder_queries = nn.Parameter(
            torch.randn(1, self.max_position_embeddings, self.hidden_size) / math.sqrt(self.hidden_size)
        )
        
        # Layers
        self.input_to_latent = CrossAttentionPlus(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.attention_dropout
        )
        
        num_latent_layers = getattr(config.gnn_config, "num_latent_layers", 6)
        print ("num_latent_layers", num_latent_layers)
        self.latent_transformer = nn.ModuleList([
            LatentTransformerLayerPlus(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout=config.attention_dropout
            ) for _ in range(num_latent_layers)
        ])
        
        self.sequence_self_attention = CrossAttentionPlus(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.attention_dropout,
             
        )

        #whether or not to group tokens for CG - otherwise latents can see ALL prior tokens 
        self.group_tokens_for_coarse_graining =  getattr(config.gnn_config, "group_tokens_for_coarse_graining", True)

        # Compute the fixed number of tokens per latent
        self.tokens_per_latent = getattr(self.config.gnn_config, "tokens_per_latent", self.max_position_embeddings // self.num_latents)
        
        max_tokens_assignable = self.tokens_per_latent * self.num_latents

        print ("Option: use_fixed_number_of_tokens_per_latent for interpolation." )
        print ("tokens_per_latent", self.tokens_per_latent, "max tokens assignable: ", max_tokens_assignable)

        if self.max_position_embeddings > max_tokens_assignable:
            raise ValueError(
                f"max_position_embeddings ({self.max_position_embeddings}) exceeds the assignable tokens "
                f"({max_tokens_assignable}) when using fixed tokens per latent."
            )

        #initial_mix=1 means entire matrix from CG latents, = 0 means all original
        initial_mix=num_latent_layers = getattr(config.gnn_config, "mix_weights_initial", 1.)
        self.mix_weights = nn.Parameter(torch.tensor([initial_mix]))  # Learnable mixing weight

        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.final_norm = nn.LayerNorm(self.hidden_size)

    def create_causal_mask(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask ensuring strict causality.
        """
        mask = torch.triu(
            torch.ones(query_len, key_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)


    def encode(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        self.hidden_input=hidden_states
        
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum allowed "
                f"length ({self.max_position_embeddings})"
            )
        
        # Add position embeddings
        input_pos = self.input_pos_emb[:, :seq_len, :]
        hidden_states = hidden_states + input_pos
        
        # Prepare latent array
        latent_pos = self.latent_pos_emb
        latents = self.latent_array + latent_pos
        latents = latents.expand(batch_size, -1, -1)

        # Create a combined attention and causal mask for input-to-latent attention
        if attention_mask is not None:
            # Pad the attention mask to match latent dimensions
            if attention_mask.shape[-2] < self.num_latents or attention_mask.shape[-1] < seq_len:
                padded_attention_mask = torch.full(
                    (attention_mask.size(0), attention_mask.size(1), self.num_latents, seq_len),
                    -1e9,
                    device=device,
                    dtype=attention_mask.dtype,
                )
                
                padded_attention_mask[..., :attention_mask.size(-2), :attention_mask.size(-1)] = attention_mask
                
                attention_mask = padded_attention_mask

    
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum allowed "
                f"length ({self.max_position_embeddings})"
            )

        # Add position embeddings
        input_pos = self.input_pos_emb[:, :seq_len, :]
        hidden_states = hidden_states + input_pos

        # Prepare latent array
        latent_pos = self.latent_pos_emb
        latents = self.latent_array + latent_pos
        latents = latents.expand(batch_size, -1, -1)

        starts = torch.arange(0, seq_len, self.tokens_per_latent, device=device)
        ends = torch.clamp(starts + self.tokens_per_latent - 1, max=seq_len - 1)

        num_active_latents = len(starts)  # Number of latents required for the sequence length

        self.token_to_latent_mapping = {
            'seq_len': seq_len,
            'starts': starts,
            'ends': ends
        }

        # Initialize group mask for fixed tokens per latent
        group_mask = torch.ones(batch_size, 1, self.num_latents, seq_len, device=device).bool()  # Always use all latents
        for i, (start, end) in enumerate(zip(starts, ends)):
            group_mask[:, :, i, start:end + 1] = False  # Mask only valid ranges for active latents

        # Mask unused latents
        if self.config.gnn_config.plot_for_debugging:
            print ("group_mask Encode", group_mask.shape)
            print (group_mask)

            mask_plot = group_mask[0,0,:].cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_plot, cmap="viridis", aspect="auto")
            plt.colorbar(label="Mask")
            plt.title(f"mask_encode, mask_plot: {mask_plot.shape}")
            plt.tight_layout()
            plt.show()

        group_mask_additive = group_mask.float() * -1e9

        # Adjust attention_mask dimensions
        if attention_mask is not None:
            padded_attention_mask = torch.full(
                (attention_mask.size(0), attention_mask.size(1), self.num_latents, seq_len),
                -1e9,
                device=device,
                dtype=attention_mask.dtype,
            )
            padded_attention_mask[:, :, :self.num_latents, :seq_len] = attention_mask [:, :, :self.num_latents, :seq_len]  # Use all latents
            attention_mask = padded_attention_mask + group_mask_additive
        else:
            attention_mask = group_mask_additive 

        # Input to latent cross-attention
        latents, _  = self.input_to_latent(
            query=self.norm1(latents),
            key=self.norm2(hidden_states),
            value=self.norm2(hidden_states),
            attention_mask=attention_mask
        )
        
        # Create strictly causal mask for latent processing
        latent_causal_mask = torch.triu(
            torch.ones(self.num_latents, self.num_latents, dtype=torch.bool, device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Create new attention mask for latents
        new_attention_mask = torch.triu(
            torch.full((self.num_latents, self.num_latents), -1e9, device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Process through transformer layers
        for layer in self.latent_transformer:
            latents, adj_matrix_current  = layer(latents, causal_mask=latent_causal_mask)
                
        self.adj_matrix_current=adj_matrix_current
        self.num_active_latents = num_active_latents # how many latents were used for the data
        return self.final_norm(latents), new_attention_mask

    def decode(self, latents: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode using the adjacency matrix extracted from latents to guide self-attention
        over the original sequence.
        """
         
        original_sequence=self.hidden_input
        batch_size, seq_len, _ = original_sequence.shape
        device = original_sequence.device

       
        adj_matrix=self.adj_matrix_current
        max_len_for_all_latent_tokens=self.max_position_embeddings


        #The algorithm encodes the entire sequence into a fixed number of tokens per maximum sequence length
        #When we provide a shorter sequence only a small number of latents are used. Yet, self attention in latent space is done with ALL latents.
        #Hence, need to select only part of the adj matrix with active latents and scale that to the original sequence length
        #
        num_active_latents=self.num_active_latents # how many latents were used for the data
        size_of_adj_matrix = max(num_active_latents // seq_len, 1)

        # Scale adjacency matrix to sequence length
        scaled_adj_matrix = F.interpolate(
            adj_matrix[:,:,:size_of_adj_matrix, :size_of_adj_matrix], size=(max_len_for_all_latent_tokens, max_len_for_all_latent_tokens), mode=self.interpolation_method_for_adj_scaling, 
            #mode="bilinear", 
        )
        scaled_adj_matrix = scaled_adj_matrix[..., :seq_len, :seq_len]

        # Ensure causality by setting the upper triangular part to zero
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )  # Shape: [seq_len, seq_len]
        scaled_adj_matrix = scaled_adj_matrix.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Incorporate attention_mask into scaled_adj_matrix
        if attention_mask is not None:
            # Convert attention_mask from pre-softmax (-1e9 for invalid) to binary mask
            binary_attention_mask = (attention_mask > -1e5).float()  # 1 for valid, 0 for invalid
            
            # Apply binary mask to scaled_adj_matrix
            scaled_adj_matrix = scaled_adj_matrix * binary_attention_mask

        # Apply the adjacency matrix for sequence-level self-attention
        plot_for_debugging=self.config.gnn_config.plot_for_debugging

        if plot_for_debugging:

            print ("adj_matrix: ",adj_matrix.shape )
            print ("scaled_adj_matrix: ",scaled_adj_matrix.shape )

            svg_path = visualize_two_adj_matrices_torch(adj_matrix.cpu() , scaled_adj_matrix.cpu() )
            print ("saved as: ", svg_path)
            
        output, _ = self.sequence_self_attention(
            query=self.norm1(original_sequence),
            key=self.norm1(original_sequence),
            value=self.norm1(original_sequence),
            supplied_attn=scaled_adj_matrix,  # Use precomputed attention
            supplied_attn_mix=self.mix_weights , #mix old and new
            causal_mask=causal_mask,
        )
  
        '''
        output, _ = self.sequence_self_attention(
            query=original_sequence,
            key=original_sequence,
            value=original_sequence,
            supplied_attn=scaled_adj_matrix,  # Use precomputed attention
        )
        '''
        return self.final_norm(output)
    
class CG_Attention_Interpolate (nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
         
        self.config = config
        self.num_latents_list = getattr(self.config.gnn_config, "num_latents_list", [64, 32, 8])
        self.use_skip = getattr(self.config.gnn_config, "CG_Attention_use_skip", True)
        print(f"Hierarchical latents {layer_idx}: ", self.num_latents_list)
        
        # Number of hierarchical levels
        self.num_hierarchy_levels = len(self.num_latents_list)

        # Initialize hierarchical encoder-decoder layers
        self.perceiver_layers = nn.ModuleList([
            PerceiverAR_Fixed_Token_Per_Latent_Scaling(self._create_layer_config(config, num_latents, idx))
            for idx, num_latents in enumerate(self.num_latents_list)
        ])

        if getattr(self.config.gnn_config, "use_projection", True):

            # Projection layers for each hierarchical level
            self.level_projections = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(self.num_hierarchy_levels)
            ])

        else:
            self.level_projections = nn.ModuleList([
                nn.Identity()
                for _ in range(self.num_hierarchy_levels)
            ])

    def _create_layer_config(self, base_config, num_latents, layer_idx):
        """
        Create a layer-specific configuration for the PerceiverAR.
        """
        layer_config = deepcopy(base_config)
        layer_config.gnn_config.num_latents = num_latents
        layer_config.layer_idx = layer_idx

        return layer_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for hierarchical attention with PerceiverAR.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size, seq_len, hidden_dim).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output hidden states after hierarchical processing.
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        # Store original sequence length for potential truncation
        original_seq_len = seq_len #
        
        # Store reconstructed sequences from each level
        reconstructed_sequences = []

        # Process through each hierarchical level
        for perceiver, projection in zip(self.perceiver_layers, self.level_projections):
            # Encode sequence to latent representation using the original sequence
            latent_states, _ = perceiver.encode(
                hidden_states,  # Always use the original sequence
                attention_mask=attention_mask  # Always use the original attention mask
            )

            # Decode back to sequence length using the original sequence
            decoded_states = perceiver.decode(
                latent_states, #not used
                attention_mask=attention_mask  # Always use the original attention mask
            )
            
            # Truncate if necessary
            if decoded_states.size(1) > original_seq_len:
                decoded_states = decoded_states[:, :original_seq_len, :]
            
            #TODO check, may not need projection layer in this case!
            # Project decoded states
            projected_states = projection(decoded_states)
            #projected_states =  decoded_states
            
            
            # Add skip connection with original input
            if self.use_skip:
                skip_connected_states = hidden_states + projected_states
            else:
                skip_connected_states = projected_states #no skip

            # Store reconstruction with skip connection
            reconstructed_sequences.append(skip_connected_states)

        # Sum up all reconstructed sequences (hierarchical integration)
        integrated_output = sum(reconstructed_sequences)

        # Add final skip connection with the original input
        integrated_output = integrated_output + hidden_states

        return integrated_output, None, None
