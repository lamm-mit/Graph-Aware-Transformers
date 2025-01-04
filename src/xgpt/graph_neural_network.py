# graph_neural_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
)
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import degree, softmax
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add, scatter_min, scatter_max
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from .gnn_config import GNNConfig


class CausalPNALayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggregators,
        scalers,
        deg,
        edge_dim=None,
        towers=1,
        pre_layers=1,
        post_layers=1,
        divide_input=False,
        act="relu",
        act_kwargs=None,
        activation_fn=None,
        train_norm=False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.aggregators = aggregators
        self.scalers = scalers
        self.deg = deg if deg is not None else torch.tensor([1, 1, 1, 1], dtype=torch.long)

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = nn.ModuleList()
        self.post_nns = nn.ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(nn.Sequential(*modules))

            in_channels_post = (len(aggregators) * len(scalers)) * self.F_in
            modules = [Linear(in_channels_post, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(nn.Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)
        self.activation_fn = activation_fn or nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.pre_nns)
        reset(self.post_nns)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            raise ValueError("Batch vector is required for causal masking.")

        # Apply causal mask
        edge_index, edge_attr = self.apply_causal_mask(edge_index, edge_attr, batch)

        # Proceed with standard MessagePassing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch)

        # Final transformations
        outs = [nn(out) for nn in self.post_nns]
        out = torch.cat(outs, dim=1)
        out = self.lin(out)
        out = self.activation_fn(out)
        return out

    def apply_causal_mask(self, edge_index, edge_attr, batch):
        source_nodes, target_nodes = edge_index
        source_batch = batch[source_nodes]
        target_batch = batch[target_nodes]

        # Keep edges within the same graph
        same_graph = source_batch == target_batch

        # Enforce causality within each graph
        causal_edges = (source_nodes <= target_nodes) & same_graph

        # Filter edges
        filtered_edge_index = edge_index[:, causal_edges]
        if edge_attr is not None:
            filtered_edge_attr = edge_attr[causal_edges]
        else:
            filtered_edge_attr = None

        return filtered_edge_index, filtered_edge_attr

    def message(self, x_i, x_j, edge_attr):
        if self.edge_dim is not None and edge_attr is not None:
            edge_feat = self.edge_encoder(edge_attr)
            h = torch.cat([x_i, x_j, edge_feat], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h) for nn in self.pre_nns]
        h = torch.stack(hs, dim=1)  # Shape: [num_edges, towers, F_in]

        return h

    def aggregate(self, inputs, index):
        # inputs: [num_edges, towers, F_in]
        # We need to apply aggregators and scalers here
        num_nodes = inputs.size(0)
        out = []

        for aggr in self.aggregators:
            if aggr == 'mean':
                agg = scatter_mean(inputs, index, dim=0)
            elif aggr == 'sum':
                agg = scatter_add(inputs, index, dim=0)
            elif aggr == 'min':
                agg = scatter_min(inputs, index, dim=0)[0]
            elif aggr == 'max':
                agg = scatter_max(inputs, index, dim=0)[0]
            else:
                raise ValueError(f"Unsupported aggregator: {aggr}")

            for scaler in self.scalers:
                if scaler == 'identity':
                    scaled = agg
                elif scaler == 'amplification':
                    deg = degree(index, num_nodes=num_nodes).unsqueeze(-1) + 1
                    scaled = agg * torch.log(deg)
                elif scaler == 'attenuation':
                    deg = degree(index, num_nodes=num_nodes).unsqueeze(-1) + 1
                    scaled = agg / torch.log(deg)
                else:
                    raise ValueError(f"Unsupported scaler: {scaler}")

                out.append(scaled)

        out = torch.cat(out, dim=-1)  # Concatenate along feature dimension
        return out  # Shape: [num_nodes, towers * len(aggregators) * len(scalers) * F_in]

    def update(self, aggr_out):
        # aggr_out: [num_nodes, towers * total_features]
        # Split the output per tower
        outs = []
        for i, nn_post in enumerate(self.post_nns):
            out = aggr_out[:, i * self.F_in : (i + 1) * self.F_in]
            out = nn_post(out)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        return out


class CausalGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, activation_fn=nn.ReLU(), bias=True):
        super(CausalGCNLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation_fn = activation_fn  # Custom activation function

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            raise ValueError("Batch vector is required for causal masking.")

        # Apply causal mask to enforce causality and optionally use supplied edge weights
        edge_index, edge_weight = self.apply_causal_mask(edge_index, edge_weight, batch)

        # Pass messages using the filtered causal edges and edge weights
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def apply_causal_mask(self, edge_index, edge_weight, batch):
        # Identify edges that respect causality within each graph
        source_nodes, target_nodes = edge_index
        source_batch = batch[source_nodes]
        target_batch = batch[target_nodes]

        # Keep edges within the same graph
        same_graph = source_batch == target_batch

        # Enforce causality within each graph
        causal_edges = (source_nodes <= target_nodes) & same_graph

        # Filter edges to include only causal connections
        filtered_edge_index = edge_index[:, causal_edges]

        # Filter or create edge weights for the causal edges
        if edge_weight is not None:
            filtered_edge_weight = edge_weight[causal_edges]
        else:
            # Default to ones if no edge weights are provided
            filtered_edge_weight = torch.ones(
                filtered_edge_index.size(1), dtype=torch.float32, device=edge_index.device
            )

        return filtered_edge_index, filtered_edge_weight

    def message(self, x_j, edge_weight):
        # Apply edge weight if provided
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * self.linear(x_j)
        else:
            return self.linear(x_j)

    def update(self, aggr_out):
        # Apply the configured activation function
        return self.activation_fn(aggr_out)


class CausalGINLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation_fn=nn.ReLU(),
        GIN_use_MLP=True,
        GIN_hidden_dim_multiplier=1,
        GIN_use_norm=True,
        norm_type='layer',
        edge_weight_scaling=False,
        bias=True,
    ):
        super(CausalGINLayer, self).__init__(aggr='add')  # Use "add" aggregation for injectivity

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.GIN_use_MLP = GIN_use_MLP
        self.GIN_use_norm = GIN_use_norm
        self.norm_type = norm_type.lower()

        if GIN_use_MLP:
            hidden_dim = int(in_channels * GIN_hidden_dim_multiplier)
            self.linear1 = nn.Linear(in_channels, hidden_dim, bias=bias)
            self.linear2 = nn.Linear(hidden_dim, out_channels, bias=bias)

            # Define a single normalization layer after the first linear layer
            if GIN_use_norm:
                if self.norm_type == 'layer':
                    self.norm = nn.LayerNorm(hidden_dim)
                    
                elif self.norm_type == 'graph':
                    from torch_geometric.nn import GraphNorm
                    self.norm = GraphNorm(hidden_dim)

                elif self.norm_type == 'instance':
                    self.norm = nn.InstanceNorm1d(hidden_dim)
                else:
                    raise ValueError(f"Unsupported norm type: {self.norm_type}")
            else:
                self.norm = nn.Identity()

            # If input and output channels differ, adjust residual connection
            if in_channels != out_channels:
                self.residual_linear = nn.Linear(in_channels, out_channels, bias=bias)
            else:
                self.residual_linear = nn.Identity()
        else:
            self.transform = nn.Linear(in_channels, out_channels, bias=bias)

        self.activation_fn = activation_fn
        self.epsilon = nn.Parameter(torch.zeros(1))  # Injective epsilon term
        self.edge_weight_scaling = edge_weight_scaling


    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            raise ValueError("Batch vector is required for causal masking.")

        # Apply causal mask to enforce causality and optionally use supplied edge weights
        edge_index, edge_weight = self.apply_causal_mask(edge_index, edge_weight, batch)

        # Propagate messages using the filtered causal edges and edge weights
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
 
        out =  self.epsilon * x + out



        if self.GIN_use_MLP:
            residual = out  # Save for skip connection

            # First linear layer
            out = self.linear1(out)
            if self.GIN_use_norm:
                out = self.norm(out)
            out = self.activation_fn(out)

            # Second linear layer
            out = self.linear2(out)
            # No normalization after the second linear layer

            # Adjust residual if dimensions differ
            residual = self.residual_linear(residual)

            
            out = out + residual


        else:
            out = self.transform(out)
            out = self.activation_fn(out)

        return out




    def apply_causal_mask(self, edge_index, edge_weight, batch):
        # Identify edges that respect causality within each graph
        source_nodes, target_nodes = edge_index
        source_batch = batch[source_nodes]
        target_batch = batch[target_nodes]

        # Keep edges within the same graph
        same_graph = source_batch == target_batch

        # Enforce causality within each graph
        causal_edges = (source_nodes <= target_nodes) & same_graph

        # Filter edges to include only causal connections
        filtered_edge_index = edge_index[:, causal_edges]

        # Filter or create edge weights for the causal edges
        if edge_weight is not None:
            filtered_edge_weight = edge_weight[causal_edges]
        else:
            # Default to None if no edge weights are provided
            filtered_edge_weight = None

        return filtered_edge_index, filtered_edge_weight

    def message(self, x_j, edge_weight):
        # Optionally scale messages using edge weights
        if self.edge_weight_scaling and edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

class CausalGNNRecombinationLayer(MessagePassing):
    def __init__(self, hidden_dim, activation_fn=nn.ReLU()):
        super(CausalGNNRecombinationLayer, self).__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.attention_fc = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        x: Tensor of shape (total_num_nodes, hidden_dim)
        edge_index: Tensor of shape (2, total_num_edges)
        edge_weight: Optional tensor of shape (total_num_edges,)
        batch: Tensor of shape (total_num_nodes,)
        """
        if batch is None:
            raise ValueError("Batch vector is required for causal masking.")

        # Apply causal mask to enforce causality
        edge_index, edge_weight = self.apply_causal_mask(edge_index, edge_weight, batch)

        # Start message passing
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=(x.size(0), x.size(0)))

    def apply_causal_mask(self, edge_index, edge_weight, batch):
        # Enforce causality by allowing edges from previous tokens only within the same graph
        source_nodes, target_nodes = edge_index
        source_batch = batch[source_nodes]
        target_batch = batch[target_nodes]

        # Keep edges within the same graph
        same_graph = source_batch == target_batch

        # Enforce causality within each graph
        causal_edges = (source_nodes < target_nodes) & same_graph

        # Filter edges to include only causal connections
        filtered_edge_index = edge_index[:, causal_edges]

        # Filter or create edge weights for the causal edges
        if edge_weight is not None:
            filtered_edge_weight = edge_weight[causal_edges]
        else:
            # Default to ones if no edge weights are provided
            filtered_edge_weight = torch.ones(
                filtered_edge_index.size(1), dtype=torch.float32, device=edge_index.device
            )

        return filtered_edge_index, filtered_edge_weight

    def message(self, x_i, x_j, edge_weight, index, ptr, size_i):
        """
        x_i: Target nodes' features (after receiving messages)
        x_j: Source nodes' features (sending messages)
        edge_weight: Edge weights for the edges
        index: Indices of target nodes for each message
        """
        # Compute attention coefficients (recombination weights)
        # Concatenate source and target node features
        combined = torch.cat([x_i, x_j], dim=-1)  # Shape: (num_edges, 2 * hidden_dim)

        # Compute attention scores
        attn_scores = self.attention_fc(combined).squeeze(-1)  # Shape: (num_edges,)
        attn_scores = self.leaky_relu(attn_scores)

        # Incorporate edge weights into attention scores
        attn_scores = attn_scores * edge_weight

        # Compute softmax over incoming edges for each node
        attn_weights = softmax(attn_scores, index, num_nodes=size_i)

        # Multiply source node features by attention weights
        weighted_messages = attn_weights.unsqueeze(-1) * x_j  # Shape: (num_edges, hidden_dim)

        return weighted_messages

    def update(self, aggr_out):
        # Apply activation function
        return self.activation_fn(aggr_out)


class CausalGraphNeuralNetwork(nn.Module):
    def __init__(self, config: GNNConfig, transformer_hidden_dim=None):
        super(CausalGraphNeuralNetwork, self).__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.activation = self.get_activation(config.activation)
        self.dropout = config.dropout
        self.use_layer_norm = config.use_layer_norm
        self.transformer_hidden_dim = transformer_hidden_dim
        
        # Convert degree back to tensor if it's a list
        if hasattr(config, 'degree') and isinstance(config.degree, list):
            self.config.degree = torch.tensor(config.degree, dtype=torch.float32)

        # Projection layers for input and output
        if transformer_hidden_dim and transformer_hidden_dim != self.hidden_dim:
            self.input_projection = nn.Linear(transformer_hidden_dim, self.hidden_dim)
            self.output_projection = nn.Linear(self.hidden_dim, transformer_hidden_dim)
        else:
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()

        # Edge encoder for PNA variants
        if config.gnn_type in ["causal_pna"]:
            self.edge_encoder = nn.Linear(1, self.hidden_dim)

        # Define GNN layers based on gnn_type
        self.convs = nn.ModuleList()
        if config.gnn_type == "causal_gcn":
            self._init_causal_gcn_layers()
        elif config.gnn_type == "causal_pna":
            self._init_causal_pna_layers()
        elif config.gnn_type == "causal_gin":
            self.GIN_use_MLP=config.GIN_use_MLP
            self.GIN_hidden_dim_multiplier=config.GIN_hidden_dim_multiplier
            self.GIN_use_norm=config.GIN_use_norm
            self.GIN_edge_weight_scaling=config.GIN_edge_weight_scaling
            self._init_causal_gin_layers()
        elif config.gnn_type == "causal_recombination":
            self._init_causal_recombination_layers()
        else:
            raise ValueError(f"Unsupported gnn_type: {config.gnn_type}")

    def _init_causal_pna_layers(self):
        """Initialize layers using CausalPNALayer."""
        for _ in range(self.num_layers):
            self.convs.append(
                CausalPNALayer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    aggregators=self.config.aggregators,
                    scalers=self.config.scalers,
                    deg=self.config.degree,
                    activation_fn=self.activation,
                    edge_dim=self.config.edge_dim,
                    towers=self.config.towers,
                    pre_layers=self.config.pre_layers,
                    post_layers=self.config.post_layers,
                    divide_input=self.config.divide_input,
                    act=self.config.act,
                    act_kwargs=self.config.act_kwargs,
                )
            )

    def _init_causal_gcn_layers(self):
        """Initialize layers using CausalGCNLayer with a custom activation function."""
        for _ in range(self.num_layers):
            self.convs.append(
                CausalGCNLayer(self.hidden_dim, self.hidden_dim, activation_fn=self.activation)
            )

    def _init_causal_gin_layers(self):
        """Initialize layers using CausalGINLayer."""
        for _ in range(self.num_layers):
            self.convs.append(
                CausalGINLayer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    activation_fn=self.activation,
                    GIN_use_MLP=self.GIN_use_MLP,
                    GIN_hidden_dim_multiplier=self.GIN_hidden_dim_multiplier,
                    GIN_use_norm=self.GIN_use_norm,
                    edge_weight_scaling=self.GIN_edge_weight_scaling,  # Use attention weights for scaling
                )
            )

    def _init_causal_recombination_layers(self):
        """Initialize layers using CausalGNNRecombinationLayer."""
        for _ in range(self.num_layers):
            self.convs.append(
                CausalGNNRecombinationLayer(
                    hidden_dim=self.hidden_dim, activation_fn=self.activation
                )
            )

    def get_activation(self, activation_name: str):
        """Return an activation function based on the specified activation name."""
        if activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "tanh":
            return nn.Tanh()
        elif activation_name.lower() == "gelu":
            return nn.GELU()
        elif activation_name.lower() == "prelu":
            return nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Project input to GNN dimension
        x = self.input_projection(x)

        # Encode edge weights for PNA variants
        if self.config.gnn_type in ["causal_pna"] and edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)
            edge_weight = self.edge_encoder(edge_weight)

        # Pass through GNN layers
        for i, conv in enumerate(self.convs):
            if self.config.gnn_type == "causal_gcn":
                x_new = conv(x, edge_index, edge_weight=edge_weight, batch=batch)
            elif self.config.gnn_type == "causal_pna":
                x_new = conv(x, edge_index, edge_attr=edge_weight, batch=batch)
            elif self.config.gnn_type == "causal_gin":
                x_new = conv(x, edge_index, edge_weight=edge_weight, batch=batch)
            elif self.config.gnn_type == "causal_recombination":
                x_new = conv(x, edge_index, edge_weight=edge_weight, batch=batch)
            else:
                raise ValueError(f"Unsupported gnn_type: {self.config.gnn_type}")

            x_new = self.activation(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Add residual connection within GNN layers if dimensions match
            if x_new.size(-1) == x.size(-1) and self.config.gnn_residual:
                x = x_new + x
            else:
                x = x_new

        # Project back to transformer dimension
        x = self.output_projection(x)

        return x
