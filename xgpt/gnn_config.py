# gnn_config.py
from transformers.models.llama.modeling_llama import *
from transformers import PretrainedConfig

#import warnings


class GNNConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: str = "gnn",
        num_layers: int = 3,
        hidden_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        combined_norm: bool = True,
        rms_norm_eps: float = 1e-5,
        aggregators: list = None,
        scalers: list = None,
        degree: list = None,
        norm_to_hidden_states: bool = False,
        remove_self_connections: bool = True,
        #use_attention_as_edge_weight: bool = False,
        continuous_transform_alpha: float = 10.0,
        zero_below_epsilon_threshold: bool = True,
        threshold: float = 0.5,
        epsilon_threshold: float = 0.5,
        #use_mean_attention: bool = True,
        add_rope: bool = False,
        enforce_causality: bool = True,
        gnn_type: str = 'causal_gnn',
        plot_for_debugging: bool = False,
        lambda_GNN: float = 0.5,
        lambda_GNN_initial: float = None,
        gnn_logic: str = 'before_MLP',  # Option to apply GNN 'before_MLP' or 'after_MLP', or 'parallel_GNN_MLP'
        adj_construction_method: str = 'sum', #'mean', 'learnable_aggregate' 'threshold_any'
        threshold_any_tau: float = 10., 
        top_k: int = 8,
        learnable_aggregate_activation: str = 'softmax', #'softmax' or 'sigmoid'
        adj_transform_hidden_dim: int = 128, #adj_transform_hidden_dim
        gnn_mode: str ='single',         # New parameter
        per_head_ff: bool = False,
        use_distance_scaling: bool = False,
        distance_weight_strength: float = 1.0,
        distance_scaling_method: str = 'power',
        gnn_residual: bool=True, 
        GIN_use_MLP: bool=False,
        GIN_use_norm: bool=False,
        GIN_hidden_dim_multiplier: float = 2, 
        GIN_edge_weight_scaling=True,
        tokenizer: Optional = None,
        use_original_hidden_states: bool = False,
        use_original_hidden_states_add_attention: bool = False,
        use_GNN_from_attention: str = 'none',
        N_GNN_from_attention_layers: int = 3,
        #GNN_from_attention_layers_o_proj_last_only: bool=False,
        use_GNN_from_attention_add_RoPE_at_every_layer: bool = False,

        attention_epsilon_strategy : str = "default",
        residual_epsilon_strategy: str = "default",
        attention_epsilon_uniform_value: float = 0.5,  # Default value if uniform strategy is used
        residual_epsilon_uniform_value: float = 0.1,  # Default value if uniform strategy is used
        attention_GIN_MLP_multiplier: int = 2,
        initial_sharpening_value: float=1.0,
        use_sharpening: bool = False,
        use_soft_masking: bool=  False,
        soft_masking_k: float = 10.,
        soft_masking_initial_threshold: float =0.01,
        sharpening_value_init: str = 'value', #or 'random'
        use_differential_attention: bool = False,
        use_differential_attention_group_norm: bool = False,
        
        use_graph_property_modulation: bool = False,
        use_graph_property_modulation_with_norm: bool = False,
        use_graph_property_modulation_with_norm_use_causal_clustering: bool = True,
        LlamaAttentionHierarchicalPerceiverAR_use_rope: bool = True,
        LlamaAttentionHierarchicalVariant_2_PerceiverAR_use_skip: bool = True,
        group_tokens_for_coarse_graining: bool = False,
        mix_weights_initial: float=0.5,
        use_projection: bool = True,
        attention_GIN_MLP_o_proj_at_end: bool = False,

        use_layer_norm_in_GIN_MLP: bool=False,
        use_no_norm_in_GIN_MLP: bool= False,

        use_hierarchical_attention: bool = False,
        #latent_size: int = 512,
        num_latent_layers: int=4,
        num_latents: int=32,
        #use_positional_embeddings_in_latent: bool = True,

        hierarchical_enc_dec_type:str = 'PerceiverAR',
        num_latents_list: list=[64, 32, 8],
        max_position_embeddings: int = 2048, #max length in PerceiverAR etc.
        use_fixed_number_of_tokens_per_latent: bool = False,

        MLP_type: str = 'standard_MLP',
        attention_GIN_MLP_GIN_MLP_mode: str ='shared', #or "per_head"

        attention_GIN_MLP_GIN_threshold_mode: str ='none',
        attention_GIN_MLP_GIN_learnable_threshold: bool = False,
        attention_GIN_MLP_GIN_threshold_value: float = 0.2,
        attention_GIN_MLP_GIN_binary_scale: float = 1.,
        attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length: float= 0.1,
        attention_GIN_MLP_attention_mix_mode: str ='A', 
        attention_GIN_MLP_separate_attention: bool=False,
        attention_GIN_MLP_scoring_hidden_dim: int = 512,
        attention_GIN_MLP_use_scoring_fnct: bool= True, # True, use MLP scoring, False= standart attention
        attention_GIN_MLP_GIN_use_softmax: bool = False, #no softmax standard 
        attention_GIN_MLP_GIN_softmax_temperature: float =1., 
        attention_GIN_MLP_GIN_MLP_pre_aggregate: bool=True, 
        attention_GIN_MLP_GIN_fuse_mode: str='epsilon',
        attention_GIN_MLP_GIN_use_ReLU_instead_of_softmax: bool = True,
        attention_GIN_MLP_GIN_sharp_softplus_beta: float = 10.,
        attention_GIN_MLP_GIN_mode: str = 'default', 
        
        
        GIN_after_attention: bool = True,
        GIN_after_attention_pre_GIN_norm: bool = True,
        GIN_after_attention_skip:bool = True,
        
                
        attention_GIN_MLP_use_second_order: bool = False,
        attention_GIN_MLP_second_order_factor: float = 0.1, 
    ):
        """
        Configuration class for the Graph Neural Network (GNN).

        Args:
            num_layers (int): Number of layers in the GNN.
            hidden_dim (int): Hidden dimension size for each GNN layer.
            activation (str): Activation function to use (e.g., 'relu', 'prelu').
            ...
        """
        self.model_type=model_type
        
        # GNN layer configuration
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
            
        self.tokenizer = tokenizer

        self.use_GNN_from_attention=use_GNN_from_attention
        self.N_GNN_from_attention_layers =N_GNN_from_attention_layers
        self.use_GNN_from_attention_add_RoPE_at_every_layer=use_GNN_from_attention_add_RoPE_at_every_layer
        self.LlamaAttentionHierarchicalPerceiverAR_use_rope = LlamaAttentionHierarchicalPerceiverAR_use_rope
        self.LlamaAttentionHierarchicalVariant_2_PerceiverAR_use_skip=LlamaAttentionHierarchicalVariant_2_PerceiverAR_use_skip

        self.mix_weights_initial=mix_weights_initial
        self.group_tokens_for_coarse_graining=group_tokens_for_coarse_graining
        self.use_fixed_number_of_tokens_per_latent=use_fixed_number_of_tokens_per_latent
        self.use_projection=use_projection
        
        self.attention_epsilon_strategy = attention_epsilon_strategy
        self.residual_epsilon_strategy = residual_epsilon_strategy
        self.attention_epsilon_uniform_value = attention_epsilon_uniform_value  # Default value if uniform strategy is used
        self.residual_epsilon_uniform_value = residual_epsilon_uniform_value # Default value if uniform strategy is used
        self.attention_GIN_MLP_multiplier=attention_GIN_MLP_multiplier #multiplier in FF MLP in GIN
        self.initial_sharpening_value=initial_sharpening_value
        self.use_sharpening=use_sharpening

        self.MLP_type=MLP_type
        self.attention_GIN_MLP_GIN_MLP_mode=attention_GIN_MLP_GIN_MLP_mode
        self.attention_GIN_MLP_GIN_learnable_threshold = attention_GIN_MLP_GIN_learnable_threshold
        self.attention_GIN_MLP_GIN_threshold_value=attention_GIN_MLP_GIN_threshold_value
        self.attention_GIN_MLP_GIN_threshold_mode=attention_GIN_MLP_GIN_threshold_mode
        self.attention_GIN_MLP_GIN_binary_scale=attention_GIN_MLP_GIN_binary_scale
        self.attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length=attention_GIN_MLP_GIN_top_k_fraction_of_sequence_length
        self.attention_GIN_MLP_attention_mix_mode=attention_GIN_MLP_attention_mix_mode
        self.attention_GIN_MLP_scoring_hidden_dim=attention_GIN_MLP_scoring_hidden_dim
        self.attention_GIN_MLP_GIN_use_softmax=attention_GIN_MLP_GIN_use_softmax
        self.attention_GIN_MLP_GIN_softmax_temperature=attention_GIN_MLP_GIN_softmax_temperature
        self.attention_GIN_MLP_GIN_MLP_pre_aggregate=attention_GIN_MLP_GIN_MLP_pre_aggregate
        self.attention_GIN_MLP_GIN_fuse_mode=attention_GIN_MLP_GIN_fuse_mode
        self.attention_GIN_MLP_GIN_use_ReLU_instead_of_softmax=attention_GIN_MLP_GIN_use_ReLU_instead_of_softmax
        self.attention_GIN_MLP_GIN_sharp_softplus_beta=attention_GIN_MLP_GIN_sharp_softplus_beta
        self.attention_GIN_MLP_GIN_mode=attention_GIN_MLP_GIN_mode
        self.GIN_after_attention = GIN_after_attention
        self.GIN_after_attention_pre_GIN_norm = GIN_after_attention_pre_GIN_norm
        self.GIN_after_attention_skip=GIN_after_attention_skip
        self.top_k=top_k
        self.use_differential_attention_group_norm=use_differential_attention_group_norm

        self.attention_GIN_MLP_use_scoring_fnct=attention_GIN_MLP_use_scoring_fnct

        self.sharpening_value_init=sharpening_value_init

        self.attention_GIN_MLP_use_second_order=attention_GIN_MLP_use_second_order
        self.attention_GIN_MLP_second_order_factor=attention_GIN_MLP_second_order_factor

        self.use_soft_masking = use_soft_masking
        self.soft_masking_k = soft_masking_k
        self.soft_masking_initial_threshold = soft_masking_initial_threshold

        self.gnn_residual = gnn_residual

        # Layer normalization settings
        self.use_layer_norm = use_layer_norm
        self.combined_norm = combined_norm
        self.rms_norm_eps = rms_norm_eps

        # GNN configuration options
        self.norm_to_hidden_states = norm_to_hidden_states
        self.remove_self_connections = remove_self_connections
        
        self.continuous_transform_alpha = continuous_transform_alpha
        
        self.zero_below_epsilon_threshold = zero_below_epsilon_threshold
        self.threshold = threshold
        self.epsilon_threshold=epsilon_threshold 
        #self.use_mean_attention = use_mean_attention
        self.add_rope = add_rope
        self.enforce_causality = enforce_causality
        self.gnn_type = gnn_type
        self.adj_construction_method = adj_construction_method
        self.threshold_any_tau=threshold_any_tau
        self.adj_transform_hidden_dim=adj_transform_hidden_dim
        self.learnable_aggregate_activation = learnable_aggregate_activation
        
        self.use_differential_attention=use_differential_attention
        
        self.use_graph_property_modulation=use_graph_property_modulation
        self.use_graph_property_modulation_with_norm=use_graph_property_modulation_with_norm
        self.use_graph_property_modulation_with_norm_use_causal_clustering=use_graph_property_modulation_with_norm_use_causal_clustering

        self.use_layer_norm_in_GIN_MLP=use_layer_norm_in_GIN_MLP
        self.use_no_norm_in_GIN_MLP=use_no_norm_in_GIN_MLP
        self.attention_GIN_MLP_o_proj_at_end=attention_GIN_MLP_o_proj_at_end
        self.attention_GIN_MLP_separate_attention=attention_GIN_MLP_separate_attention

        self.use_hierarchical_attention=use_hierarchical_attention
        self.num_latent_layers = num_latent_layers
        self.num_latents=num_latents
        
        self.num_latents_list=num_latents_list
        self.max_position_embeddings=max_position_embeddings


        self.gnn_mode = gnn_mode    
        self.per_head_ff = per_head_ff
        self.use_distance_scaling = use_distance_scaling
        self.distance_weight_strength = distance_weight_strength
        self.distance_scaling_method=distance_scaling_method

        self.use_original_hidden_states=use_original_hidden_states
        self.use_original_hidden_states_add_attention = use_original_hidden_states_add_attention

        self.hierarchical_enc_dec_type = hierarchical_enc_dec_type

        #for PNA
        if self.gnn_type == "causal_pna":
            self.aggregators = aggregators if aggregators is not None else ['mean', 'min', 'max', 'std']
            self.scalers = scalers if scalers is not None else ['identity', 'amplification', 'attenuation']
            self.degree = degree if degree is not None else [1.] * 10  # Default to a list of ones

        if self.gnn_type == "causal_gin":
            self.GIN_use_MLP=GIN_use_MLP
            self.GIN_hidden_dim_multiplier = GIN_hidden_dim_multiplier 
            self.GIN_use_norm=GIN_use_norm
            self.GIN_edge_weight_scaling=GIN_edge_weight_scaling

        self.lambda_GNN = lambda_GNN
        self.lambda_GNN_initial = lambda_GNN_initial
        
        self.plot_for_debugging = plot_for_debugging
        self.gnn_logic = gnn_logic  # GNN application logic in the transformer layer
        
        # Validation for `threshold` and `gnn_logic`
        assert isinstance(self.threshold, float) and self.threshold >= 0, \
            f"Threshold must be a non-negative float, got {self.threshold}"
        assert self.gnn_logic in ['before_MLP', 'after_MLP', 'parallel_GNN_MLP'], \
            f"Invalid gnn_logic value: {self.gnn_logic}. Expected 'before_MLP', 'parallel_GNN_MLP', or 'after_MLP'."
 
 
