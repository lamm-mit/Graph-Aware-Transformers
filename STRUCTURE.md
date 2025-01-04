# Code and File Documentation

We provide additional details about the code, structure, flow and algorithms featured in this repository. 

### custom_tokenizer.py
This file contains utilities for creating a custom tokenizer based on the `LlamaTokenizerFast` and BPE (Byte Pair Encoding). 

### gnn_config.py: Key config for all features included in this code 
Defines a configuration class for GNN integration:
- **GNNConfig**: Stores and manages GNN parameters, including layer counts and dimensions.

### llama_decoder_layer_with_gnn.py
Enhances Llama decoder layers with GNN functionality:
- **LlamaDecoderLayerWithGNN**: Key class, integrates GNNs into Llama decoder layers, offering methods for constructing adjacency matrices and applying GNNs.

Other classes:
- **AggregatedLearnableAdjacencyTransformer**: Computes learnable adjacency matrices for GNNs in transformers. Experimental. 
- **LlamaAttention_Original**: Implements original multi-headed attention mechanisms. Included again for reference. 
- **LlamaSimpleMLP**: Implements a simple MLP layer for integration with transformers.
- **ShallowLlamaMLP**: Defines a shallow variant of MLP.

Note: **LlamaMLP** is defined in the original Llama class, not repeated here. 

### Attention_GNN.py
Implements attention mechanisms with GNN functionality, replacing standard Llama attention with GNN-Attention variants (LlamaAttentionGIN, LlamaAttentionPNA, LlamaAttentionPNA_LM)
- **AdjRMSNorm**: Applies RMS normalization to adjacency matrices.
- **GINLayer**: Implements a Graph Isomorphism Network (GIN) layer.
- **LlamaAttentionGIN**: Replaces multi-headed attention with GIN-based aggregation.
- **LlamaAttentionPNA**: Uses Principal Neighborhood Aggregation (PNA) for attention.
- **LlamaAttentionPNA_LM**: Adapts PNA-based attention for language modeling (variant with per-token aggregation that LlamaAttentionPNA versus processing each hidden dimension index separately as in LlamaAttentionPNA).

### graph_neural_network.py
Implements GNN layers for causal modeling and message-passing, for use in fine-tuning:
- **CausalGINLayer**: A GIN layer with causal masking and messaging capabilities.

Additionalm experimental features: 
- **CausalGCNLayer**: A GCN (Graph Convolutional Network) layer adapted for causal modeling.
- **CausalPNALayer**: A PNA layer tailored for causal relationships.
- **CausalGNNRecombinationLayer**: Combines multiple GNN strategies for enhanced performance.
- **CausalGraphNeuralNetwork**: Combines multiple GNN layers and provides forward propagation with activation functions.

## Other experimental Features (CG-Attention, and others) 

### CG_Attention.py

Defines classes for various attention mechanisms, including cross-attention and latent attention for GNNs:
- **CrossAttention**: Implements a basic cross-attention mechanism.
- **LatentTransformerLayer**: A transformer layer designed for processing latent representations.
- **PerceiverAR_Fixed_Token_Per_Latent**: Encodes and decodes fixed tokens per latent representation.
- **CG_Attention**: Creates a configurable attention mechanism for GNN integration.
- **CrossAttentionPlus**: Enhances cross-attention with additional parameters.
- **LatentTransformerLayerPlus**: Extends latent transformer layers with more advanced configurations.
- **PerceiverAR_Fixed_Token_Per_Latent_Scaling**: Adds scaling capabilities to the PerceiverAR class.
- **CG_Attention_Interpolate**: Interpolates between local and supplied attention.

#### CG_Attention

The `CG_Attention` class handles the encode-attend-decode process for latent representations using fixed token-per-latent relationships. It uses ```PerceiverAR_Fixed_Token_Per_Latent```.

##### PerceiverAR_Fixed_Token_Per_Latent
- Encodes input tokens into fixed latents and applies causal attention masks.
- Provides methods:
  - `encode`: Processes input tokens into latent representations.
  - `decode`: Transforms processed latents back into token representations.
- Ensures causal masking to maintain proper sequence-based processing.

### PerceiverAR_Fixed_Token_Per_Latent_Scaling
- Extends `PerceiverAR_Fixed_Token_Per_Latent` with scaling functionality.
- Dynamically adjusts latent features or attention weights using interpolation parameters.
- Enables blending of multiple attention mechanisms.

In the following we provide a more detailed look at these two experimental classes. 

##### Workflow CG_Attention:

1. **Initialization (`__init__`)**:
   - Configures the attention mechanism using `_create_layer_config`.
   - Defines layers like `CrossAttention` and `LatentTransformerLayer`.

2. **Layer Configuration (`_create_layer_config`)**:
   - Sets up:
     - `CrossAttention`: Handles multi-head attention with causal masking.
     - `LatentTransformerLayer`: Wraps attention with feedforward layers.

3. **Forward Method (`forward`)**:
   - **Input**:
     - Latent representations (`latents`).
     - Optional attention parameters like causal masks.
   - **Process**:
     - Encodes input latents using `PerceiverAR_Fixed_Token_Per_Latent.encode`.
     - Applies the attention mechanism with causal masking.
     - Decodes processed latents using `PerceiverAR_Fixed_Token_Per_Latent.decode`.
   - **Output**: Processed latents after the encode-attend-decode process.

### Flowchart

```plaintext
Input Tokens
   ↓
Encode with PerceiverAR_Fixed_Token_Per_Latent.encode
   ↓
Apply Causal Mask from PerceiverAR_Fixed_Token_Per_Latent
   ↓
Latents → Attention → Latents
   ↓
Decode with PerceiverAR_Fixed_Token_Per_Latent.decode
   ↓
Output Processed Tokens
```

#### CG_Attention_Scaling.py Workflow

The `CG_Attention_Interpolate` class extends functionality to blend local and external attention using scaled latents. It uses `PerceiverAR_Fixed_Token_Per_Latent_Scaling`.

##### PerceiverAR_Fixed_Token_Per_Latent_Scaling
- Extends `PerceiverAR_Fixed_Token_Per_Latent` with scaling functionality.
- Dynamically adjusts latent features or attention weights using interpolation parameters.
- Provides methods:
  - `encode`: Encodes input tokens into latent representations with scaling adjustments.
  - `decode`: Decodes scaled and interpolated latents back into token representations.
- Enables blending of local and external attention mechanisms.

##### Workflow:

1. **Initialization (`__init__`)**:
   - Configures interpolation parameters.
   - Calls `_create_layer_config` to define attention blending.

2. **Layer Configuration (`_create_layer_config`)**:
   - Sets up:
     - `CrossAttentionPlus`: Extends multi-head attention to support blending of local and external attention.
     - `LatentTransformerLayerPlus`: Processes blended attention with feedforward transformations.

3. **Forward Method (`forward`)**:
   - **Input**:
     - Latent representations (`latents`).
     - Local attention and external attention.
   - **Process**:
     - Encodes input tokens using `PerceiverAR_Fixed_Token_Per_Latent_Scaling.encode`.
     - Applies scaling adjustments to attention weights or latent features.
     - Blends local and external attention using interpolation masks.
     - Decodes interpolated latents using `PerceiverAR_Fixed_Token_Per_Latent_Scaling.decode`.
   - **Output**: Processed latents with interpolated attention.

### Flowchart CG_Attention_Scaling

```plaintext
Input Tokens  
   ↓
Encode with PerceiverAR_Fixed_Token_Per_Latent_Scaling.encode
   ↓
Scale Attention/Latents (PerceiverAR_Fixed_Token_Per_Latent_Scaling)
   ↓
Blend Local and External Attention
   ↓
Latents → Interpolated Attention → Latents
   ↓
Decode with PerceiverAR_Fixed_Token_Per_Latent_Scaling.decode
   ↓
Output Processed Tokens
```

### MLP_GNN.py
This file implements several classes that integrate the Transformer FF MLP (Multi-Layer Perceptron) with GNN processing:
- **LlamaMLP_HalfwayGIN**: Likely defines a halfway integration of GNN with MLP. Handles initialization and forward computation.
- **LlamaMLP_MultiHop**: Extends GNN functionality to support multi-hop message passing with MLP.
- **LlamaMLP_HalfwayGIN_MultiAggregation**: Combines halfway GNN with multiple aggregation strategies.



## Additional helper classes

### utils.py
Contains utility functions for working with GNN-integrated Llama models, including configuration, initialization, and processing functions.  

### llama_model_with_gnn.py
Extends the Llama model to integrate GNN layers:
- **FlashAttentionKwargs**: Defines arguments for configuring Flash Attention.
- **LlamaModelWithGNN**: Extends Llama models to include GNN layers and custom configurations.

### llama_for_causal_lm_with_gnn.py
Wraps the causal language model to include GNN-enhanced capabilities:
- **LlamaForCausalLMWithGNN**: Adds GNN functionality to the standard `LlamaForCausalLM` class.
