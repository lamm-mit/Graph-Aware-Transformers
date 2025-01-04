# MLP_GNN.py contains the implementation of the LlamaMLP_HalfwayGIN, LlamaMLP_MultiHop, and LlamaMLP_HalfwayGIN_MultiAggregration classes.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class LlamaMLP_HalfwayGIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = getattr(config, "mlp_bias", False)

        self.act_fn = ACT2FN[config.hidden_act]

        # Full MLP layers as usual:
        # We'll split the MLP into two halves:
        # First half: gate_proj and up_proj (like normal)
        # After partial activation, we do a GIN step per head
        # Then second half: down_proj
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.mlp_bias)

        # GIN parameters per head
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_heads)
        ])
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_heads)
        ])

    def forward(self, x, adjacency):
        """
        x: (batch, seq_len, hidden_size)
        adjacency: (batch, num_heads, seq_len, seq_len)
        
        Steps:
        1. Apply partial MLP (gate and up projection) in full hidden dimension form.
        2. Reshape to per-head format.
        3. Apply GIN per head on the intermediate representations.
        4. Recombine heads to full dimension.
        5. Apply remaining down_proj to complete the MLP.
        """

        bsz, seq_len, _ = x.size()

        # First half of MLP as usual
        gate = self.gate_proj(x)     # (b, s, int_size)
        up = self.up_proj(x)         # (b, s, int_size)
        h_inter = self.act_fn(gate) * up  # (b, s, int_size)

        # Now we have intermediate_size representation
        # Reshape to per-head format to apply GIN:
        # intermediate_size should be divisible by num_heads for this step to make sense
        # If intermediate_size is not divisible by num_heads, you might consider a different dimension or strategy.
        head_int_dim = self.intermediate_size // self.num_heads
        h_inter_heads = h_inter.view(bsz, seq_len, self.num_heads, head_int_dim).transpose(1, 2)
        # h_inter_heads: (b, h, s, head_int_dim)

        # Apply GIN per head on h_inter_heads
        out_per_head = []
        for h in range(self.num_heads):
            h_h = h_inter_heads[:, h, :, :]        # (b, s, head_int_dim)
            A_h = adjacency[:, h, :, :]            # (b, s, s)
            epsilon_h = self.epsilons[h]
            alpha_h = self.alphas[h]

            # GIN update inside intermediate features:
            h_gin = (1.0 + epsilon_h)*h_h + alpha_h*torch.matmul(A_h, h_h)

            out_per_head.append(h_gin.unsqueeze(1))

        h_gin_combined = torch.cat(out_per_head, dim=1)  # (b, h, s, head_int_dim)
        h_gin_combined = h_gin_combined.transpose(1, 2).contiguous().view(bsz, seq_len, self.intermediate_size)
        # Now we have GIN-enriched intermediate features back in (b, s, int_size)

        # Finish MLP with down_proj
        out = self.down_proj(h_gin_combined)  # (b, s, hidden_size)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaMLP_MultiHop(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = getattr(config, "mlp_bias", False)

        # Activation
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

        # Per-head linear transformations
        self.head_gate_projs = nn.ModuleList([
            nn.Linear(self.head_dim, self.intermediate_size, bias=self.mlp_bias)
            for _ in range(self.num_heads)
        ])
        self.head_up_projs = nn.ModuleList([
            nn.Linear(self.head_dim, self.intermediate_size, bias=self.mlp_bias)
            for _ in range(self.num_heads)
        ])
        self.head_down_projs = nn.ModuleList([
            nn.Linear(self.intermediate_size, self.head_dim, bias=self.mlp_bias)
            for _ in range(self.num_heads)
        ])

        # Learnable parameters for blending multiple hops
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_heads)
        ])
        self.alphas_1 = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_heads)
        ])
        self.alphas_2 = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.num_heads)
        ])
        self.alphas_3 = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_heads)
        ])

    def forward(self, x, adjacency):
        """
        x: (batch, seq_len, hidden_size)
        adjacency: (batch, num_heads, seq_len, seq_len)
        
        We'll compute:
        A² = A @ A
        A³ = A² @ A

        Then integrate them as:
        h_adj = (1+epsilon)*W_u x + alpha_1 * A(W_u x) + alpha_2 * A²(W_u x) + alpha_3 * A³(W_u x)
        """
        bsz, seq_len, _ = x.size()

        # Reshape to per-head
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, s, d)

        # Precompute multi-hop adjacencies
        # A: (b, h, s, s)
        A = adjacency
        A2 = torch.matmul(A, A)   # A²
        A3 = torch.matmul(A2, A)  # A³

        out_per_head = []
        for h in range(self.num_heads):
            x_h = x[:, h, :, :]  # (b, s, d)
            Wg_x = self.head_gate_projs[h](x_h)   # (b, s, int_size)
            Wu_x = self.head_up_projs[h](x_h)     # (b, s, int_size)
            Wg_x = self.act_fn(Wg_x)

            epsilon_h = self.epsilons[h]
            alpha_1 = self.alphas_1[h]
            alpha_2 = self.alphas_2[h]
            alpha_3 = self.alphas_3[h]

            # Single-hop aggregation: A * Wu_x
            A_Wu_x = torch.matmul(A[:, h], Wu_x)   # (b, s, int_size)
            # Two-hop aggregation: A² * Wu_x
            A2_Wu_x = torch.matmul(A2[:, h], Wu_x) # (b, s, int_size)
            # Three-hop aggregation: A³ * Wu_x
            A3_Wu_x = torch.matmul(A3[:, h], Wu_x) # (b, s, int_size)

            # Combine multiple hops
            # (1+epsilon)*Wu_x for self-features
            # alpha_1*(A Wu_x) for 1-hop neighbors
            # alpha_2*(A² Wu_x) for 2-hop neighbors
            # alpha_3*(A³ Wu_x) for 3-hop neighbors
            h_adj_h = (1.0 + epsilon_h)*Wu_x + alpha_1*A_Wu_x + alpha_2*A2_Wu_x + alpha_3*A3_Wu_x

            h_inter_h = Wg_x * h_adj_h  # gating

            out_h = self.head_down_projs[h](h_inter_h)  # (b, s, d)
            out_per_head.append(out_h.unsqueeze(1))

        out = torch.cat(out_per_head, dim=1)  # (b, h, s, d)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class LlamaMLP_HalfwayGIN_MultiAggregration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = getattr(config, "mlp_bias", False)

        self.act_fn = ACT2FN[config.hidden_act]

        # Standard MLP (split into two parts)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.mlp_bias)

        # GIN parameters per head
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_heads)
        ])
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_heads)
        ])

        # For richer aggregations, we'll introduce attention-based aggregator parameters
        # One set of Q/K projections per head
        self.attn_query_projs = nn.ModuleList([
            nn.Linear(self.intermediate_size // self.num_heads, self.intermediate_size // self.num_heads, bias=False)
            for _ in range(self.num_heads)
        ])
        self.attn_key_projs = nn.ModuleList([
            nn.Linear(self.intermediate_size // self.num_heads, self.intermediate_size // self.num_heads, bias=False)
            for _ in range(self.num_heads)
        ])

        # Aggregator MLP
        # We have 4 aggregators: original_feat, sum_agg, max_agg, attn_agg
        # Each is intermediate_size/num_heads in dimension, so combined is 4 * (int_size/num_heads)
        agg_input_dim = 4 * (self.intermediate_size // self.num_heads)
        self.aggregator_mlp = nn.Sequential(
            nn.Linear(agg_input_dim, self.intermediate_size // self.num_heads, bias=self.mlp_bias),
            self.act_fn,
            nn.Linear(self.intermediate_size // self.num_heads, self.intermediate_size // self.num_heads, bias=self.mlp_bias)
        )

    def forward(self, x, adjacency):
        """
        x: (batch, seq_len, hidden_size)
        adjacency: (batch, num_heads, seq_len, seq_len)

        Steps:
        1. Apply the first half of MLP (gate & up) in the full dimension.
        2. Reshape to per-head format and apply richer GIN aggregations.
        3. Recombine and apply down_proj to finish the MLP.
        """
        bsz, seq_len, _ = x.size()

        # First half of MLP
        gate = self.gate_proj(x)       # (b, s, int_size)
        up = self.up_proj(x)           # (b, s, int_size)
        h_inter = self.act_fn(gate) * up  # (b, s, int_size)

        # Reshape to per-head
        head_int_dim = self.intermediate_size // self.num_heads
        h_inter_heads = h_inter.view(bsz, seq_len, self.num_heads, head_int_dim).transpose(1, 2)
        # h_inter_heads: (b, h, s, head_int_dim)

        out_per_head = []
        for h in range(self.num_heads):
            h_h = h_inter_heads[:, h, :, :]        # (b, s, head_int_dim)
            A_h = adjacency[:, h, :, :]            # (b, s, s)
            epsilon_h = self.epsilons[h]
            alpha_h = self.alphas[h]

            # Original (1+ε)*h_h
            original_feat = (1.0 + epsilon_h)*h_h

            # Sum-based aggregator: alpha_h * (A_h @ h_h)
            sum_agg = alpha_h * torch.matmul(A_h, h_h)  # (b, s, head_int_dim)

            # Max-based aggregator:
            # To implement max, we consider all neighbors. If A_h is a probability distribution,
            # we just take a max over s dimension. For a true max over actual neighbors, 
            # consider thresholding or binarizing A_h.
            # Here, let's do a straightforward max over all tokens for demonstration.
            # Expand h_h to (b, s, s, head_int_dim) to apply a max across s dimension:
            h_h_expanded = h_h.unsqueeze(2).expand(bsz, seq_len, seq_len, head_int_dim)
            # max_agg over neighbors:
            max_agg, _ = torch.max(h_h_expanded, dim=2) # (b, s, head_int_dim)

            # Attention-based aggregator:
            # Q, K = projections of h_h
            Q = self.attn_query_projs[h](h_h)  # (b, s, head_int_dim)
            K = self.attn_key_projs[h](h_h)    # (b, s, head_int_dim)
            # Compute attn scores:
            attn_scores = torch.matmul(Q, K.transpose(2, 1)) / (head_int_dim**0.5)  # (b, s, s)
            # Combine with A_h structure: 
            # We'll use log trick: attn_weights = softmax(log(A_h + 1e-9) + attn_scores)
            combined_scores = attn_scores + torch.log(A_h + 1e-9)
            attn_weights = F.softmax(combined_scores, dim=-1)
            attn_agg = torch.matmul(attn_weights, h_h)  # (b, s, head_int_dim)

            # Combine all aggregators:
            # [original_feat, sum_agg, max_agg, attn_agg]
            combined = torch.cat([original_feat, sum_agg, max_agg, attn_agg], dim=-1) # (b, s, 4*head_int_dim)

            # Pass through aggregator MLP
            h_gin_head = self.aggregator_mlp(combined) # (b, s, head_int_dim)

            out_per_head.append(h_gin_head.unsqueeze(1))

        h_gin_combined = torch.cat(out_per_head, dim=1)  # (b, h, s, head_int_dim)
        h_gin_combined = h_gin_combined.transpose(1, 2).contiguous().view(bsz, seq_len, self.intermediate_size)

        # Finish with down_proj
        out = self.down_proj(h_gin_combined)  # (b, s, hidden_size)
        return out
