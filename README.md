# Graph-Aware Isomorphic Attention for Adaptive Dynamics in Transformers

We present an approach to enhancing Transformer architectures by integrating graph-aware relational reasoning into their attention mechanisms. Building on the inherent connection between attention and graph theory, we reformulate the Transformer’s attention mechanism as a graph operation and propose Graph-Aware Isomorphic Attention. This method leverages advanced graph modeling strategies, including Graph Isomorphism Networks (GIN) and Principal Neighborhood Aggregation (PNA), to enrich the representation of relational structures. Our approach improves the model’s ability to capture complex dependencies and generalize across tasks, as evidenced by a reduced generalization gap and improved learning performance. 

Additionally, we expand the concept of graph-aware attention to introduce Sparse GIN-Attention, a fine-tuning approach that employs sparse GINs. By interpreting attention matrices as sparse adjacency graphs, this technique enhances the adaptability of pre-trained foundational models with minimal computational overhead, endowing them with graph-aware capabilities. Across our experiments, our results demonstrate that graph-aware attention mechanisms outperform traditional attention in both training efficiency and validation performance. Furthermore, sparse GIN fine-tuning achieves improved training dynamics and better generalization compared to conventional methods like LoRA. These insights not only bridge graph theory and Transformer architectures but also uncover latent graph-like structures within traditional attention mechanisms, offering a new lens through which Transformers can be understood and optimized. 

By evolving Transformers as hierarchical GIN models, we reveal their implicit capacity for graph-level relational reasoning. This perspective suggests profound implications for foundational model development, enabling the design of architectures that dynamically adapt to both local and global dependencies. Applications in bioinformatics, materials science, language modeling, and beyond could benefit from this synthesis of relational and sequential data modeling, setting the stage for interpretable and generalizable modeling strategies.

![image](https://github.com/user-attachments/assets/02c9b587-73f0-4293-84f8-574bc2e9018c)

Figure 1: Encoder-only transformer architecture (panel A), adapted here by using a GNN-based self-attention mechanism with a graph neural network. As another variant (panel B) suitable for fine-tuning a pre-trained model akin to a LoRA model, we introduce Sparse-GIN, an option where we retain the adjacency matrix predicted by the pretrained model but instead use it to construct a sparse adjacency matrix.

![image](https://github.com/user-attachments/assets/5c15d37d-c693-453d-822a-97a36d4c9b8b)

Figure 2: Visualization of adjacency matrices and interpretation of corresponding causal graphs. Panel A: Visual representation of an adjacency matrix for one specific layer and one head, extracted from a pretrained model. Panel B, left shows a large-scale adjacency matrix, where interaction strengths are color-coded, with annotations highlighting specific points of interest. Panel B, right displays the corresponding causal graph, illustrating directional relationships between nodes based on the adjacency matrix.  These visualizations provide insights into the structural and causal relationships encoded in the adjacency matrices.

## Installation

#### Install PyTorch first

It is recommended to first install PyTorch separately so that ```torch_scatter``` is installed correctly.

Note, for PyTorch installation, check https://pytorch.org/get-started/locally/ for details:

```bash
pip3 install torch torchvision torchaudio
```

#### Install directly from GitHub via pip

```bash
conda create -n xgpt_env python=3.11 -y
conda activate xgpt_env

pip install git+https://github.com/lamm-mit/Graph-Aware-Transformers.git#egg=xgpt
```

#### Clone repository and install as editable library

```bash
conda create -n xgpt_env python=3.11 -y
conda activate xgpt_env

git clone https://github.com/lamm-mit/Graph-Aware-Transformers.git
cd Graph-Aware-Transformers

pip install -e .
```

Additional details of the code and algorithms, including experimental features such as coarse-grained (CG) latent representations, can be found [here](STRUCTURE.md). 


#### Import the library for use in Python
```python
from xgpt import *
```

Detailed examples on how to set up/train models are included below. 

## Create a GIN-Transformer Model from Scratch

Here we show how to create a GIN-Transformer model from scratch. We use the ```meta-llama/Meta-Llama-3-8B-Instruct``` model as source for basic model hyperparameters (not weights, however). 

- Step 1: Load dataset (necessary to train custom tokenizer)
- Step 2: Train tokenizer
- Step 3: Setup GIN-Transformer model
- Step 4: Train model

### Load dataset and train tokenizer

#### Load dataset
```python
from datasets import load_dataset

dataset = load_dataset("lamm-mit/protein_secondary_structure_from_PDB")
max_length=256

dataset = dataset.filter(lambda example: example['Sequence_length'] < max_length)
dataset = dataset['train'] 
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access the new splits
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Verify the sizes of the new datasets
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Apply chat template
def format_data(example):
    '''
    example["text"] = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["answer"]}],
        tokenize=False, add_generation_prompt=False
    )
    '''
    example["text"] =f"<|begin_of_text|><|sequence|>{example['Sequence']}<|/sequence|><|{example['Primary_SS_Type']}|><|{example['Secondary_SS_Type']}|><|eot_id|>"
    
    return example

train_dataset = train_dataset.map(format_data,remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(format_data, remove_columns=test_dataset.column_names)
```

#### Train tokenizer

```python
from xgpt import *

# Train the tokenizer
texts = train_dataset['text']
tokenizer = train_tokenizer_from_scratch(texts, vocab_size=20, special_tokens = [
    "<|pad|>",
    "<|eot_id|>", 
    "<|begin_of_text|>",
    "<|unk|>",
    "<|mask|>",
    "<|sequence|>",
    "<|/sequence|>",
    # Single-letter amino acid codes
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    # Additional special words
    "<|AH|>", "<|BS|>", "<|UNSTRUCTURED|>", "<|BEND|>", "<|PHIHELIX|>", "<|310HELIX|>", "<|BETABRIDGE|>", "<|T|>",
]
)

# Save the trained tokenizer
tokenizer.save_pretrained("./custom_tokenizer")

# Test with various scenarios
test_cases = [
    "<|begin_of_text|><|sequence|>A A A I<|/sequence|>",  # Simple space-separated
    "<|begin_of_text|><|sequence|>AAAIAIIAJ<|/sequence|>",  # Simple space-separated
    "Hello World!",  # With punctuation
    "Test   Multiple   Spaces",  # Multiple spaces
    "NoSpaces",  # No spaces
    "123.456",  # Numbers
    "user@email.com",  # Special characters
    "Mixed12345Chars!@#",  # Mixed content
]

print("Testing tokenizer:")
for test in test_cases:
    encoded = tokenizer.encode(test, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {repr(test)}")
    print(f"Encoded : {encoded}")
    print(f"Decoded : {repr(decoded)}")
    
# Print vocabulary info
print(f"\nVocabulary size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.special_tokens_map}")

# tokenizer.padding_side,    tokenizer.pad_token
```
You can also push the tokenizer to the hub:
```python
tokenizer.push_to_hub ('lamm-mit/custom_GIN_Attention_tokenizer')
```

#### Create GIN model
```python
#Load Graph-Aware Transformer library
from xgpt import *

from transformers import set_seed
set_seed(42)

# Load Pretrained LLaMA Configuration on which model will be based
pretrained_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

# Adapt these parameters to whatever your choices are, e.g. change number of heads, head_dim, number of layers, etc. 
transformer_config = LlamaConfig.from_pretrained(pretrained_model_name)
transformer_config.num_attention_heads=8
transformer_config.num_key_value_heads=transformer_config.num_attention_heads
transformer_config.head_dim=70
transformer_config.hidden_size=transformer_config.head_dim*transformer_config.num_attention_heads 
transformer_config.intermediate_size=512 #ALT: 4*transformer_config.hidden_size
transformer_config.num_hidden_layers=6
transformer_config.torch_dtype='bfloat16'
transformer_config.vocab_size=tokenizer.vocab_size
transformer_config._attn_implementation='eager' 

gnn_config = GNNConfig(
    num_layers=1,        
    activation="relu", #"prelu"
    dropout=0.1,
    lambda_GNN=1,
    norm_to_hidden_states=False,
    use_layer_norm=False,  
    combined_norm=False,
    rms_norm_eps=1e-5,
    hidden_dim=transformer_config.hidden_size,
    learnable_aggregate_activation ='softmax', #
    gnn_mode='none',
    
    ### Set type of GNN-Attention you want to create
    #use_GNN_from_attention='LlamaAttentionPNA',
    use_GNN_from_attention='LlamaAttentionGIN',    

    attention_GIN_MLP_GIN_use_softmax=True,
    attention_GIN_MLP_use_scoring_fnct=False, #standard attn
    attention_GIN_MLP_multiplier = 0.5, #1, 2, 4, ...  
    
    use_sharpening=True, sharpening_value_init='value', initial_sharpening_value=1.0,

    attention_GIN_MLP_o_proj_at_end=False, 

    use_differential_attention = False,

    ### Set transformer FF MLP type - here you can change the transformer FF type if needed
    MLP_type='standard_MLP', #'linear_MLP' 'no_MLP' 'shallow_MLP'
)

model_with_gnn  = load_model_with_pretrained_transformer( gnn_config, transformer_config, 
                                torch_dtype='bfloat16',
                                pretrained_model_name = None, attn_implementation='eager',# 'flash_attention_2' #'eager'
                                )

# Move to appropriate device (if necessary)
model_with_gnn.to("cuda" if torch.cuda.is_available() else "cpu")

count_trainable_parameters(model_with_gnn)
```
#### Train model

Once we loaded the training data and created the model, we train the model, like so: 

```python
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback

sample_steps    = 100
max_seq_length  = 300

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, prompts, max_tokens, temperature, sample_steps, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.sample_steps = sample_steps
        self.test_dataset = test_dataset
        self.perplexity_scores = []
        self.test_scores = []
        self.trainable_scale_history = []
        self.loss_fct = CrossEntropyLoss(reduction='none')  # Changed to 'none' for per-token loss

    def on_step_end(self, args, state, control,
                    log_trainable_scale_values=True,
                    **kwargs):
        if state.global_step % self.sample_steps == 0:
            print(f"\n[Sample Generation at Step {state.global_step}]")
            for item in self.prompts:
                res=perform_inference(self.model, self.tokenizer, 
                                  prompt=item, 
                                  max_tokens=self.max_tokens, 
                                  temperature=self.temperature)[0]
                print ("QUESTION: ", item)
                print ("RESPONSE: ", res)
                
            try:
                # Log trainable_scale values
                if log_trainable_scale_values:
                    layer_scales = []
                    total_scale = 0
                    num_layers = len(self.model.model.layers)
    
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        trainable_scale_value = layer.trainable_scale.item()
                        layer_scales.append(trainable_scale_value)
                        total_scale += trainable_scale_value
    
                    average_trainable_scale = total_scale / num_layers
                    self.trainable_scale_history.append((state.global_step, layer_scales, average_trainable_scale))
                    print(f"Average trainable_scale at step {state.global_step}: {average_trainable_scale}")

            except Exception as e:
                raise OSError(f"Error: {str(e)}")

sample_generation_callback = SampleGenerationCallback(
    model=model_with_gnn,
    tokenizer=tokenizer,
    prompts=[
             test_dataset['text'][0][:-60],
             test_dataset['text'][10][:-60]
             #...
            ],

    max_tokens=128,
    temperature=0.1,
    sample_steps=sample_steps,
    test_dataset=test_dataset,    
)

# Training arguments and initialization remain the same
training_args = SFTConfig(
    output_dir="./results_output",
    eval_strategy="steps",
    eval_steps=sample_steps,
    learning_rate=1e-4, #1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=9,
    weight_decay=0.01,
    logging_dir="./logs_output",
    lr_scheduler_type="constant", #'cosine'
    max_seq_length=max_seq_length,
    logging_steps=sample_steps,
    warmup_steps=250,
    dataset_text_field="text",
    packing=False,
    max_grad_norm=1,
    report_to='none',
    save_strategy='no', #'epoch',
    do_eval=True,
)

trainer = SFTTrainer(
    model=model_with_gnn,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[sample_generation_callback],
)

# Train
trainer.train()
```

You can save/push the model like so:
```python
model_with_gnn.push_to_hub ('lamm-mit/GIN-Transformer-Model')
tokenizer.push_to_hub ()
```
![image](https://github.com/user-attachments/assets/42c3c673-58e6-4595-b4f6-4e94641d7431)

Figure 3: Construction of the GIN-Attention mechanism. The flowchart shows how input embeddings in the hidden states in each layer in the  transformer via self-attention are used to construct the attention matrix. The output is processed further before aggregation and GIN-MLP application. 

## Create a Sparse-GIN Fine Tuning Model

Here we show how to fine-tune a pre-trained Transformer model using the Sparse-GIN fine-tuning method. We use the ```meta-llama/Llama-3.2-3B-Instruct``` model as pre-trained model. 

- Step 1: Load dataset  
- Step 2: Create Sparse-GIN on top of pre-trained Llama model 
- Step 3: Train model

#### Load dataset

```python
from datasets import load_dataset

dataset = load_dataset("mlabonne/orca-math-word-problems-80k")
dataset = dataset['train']

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access the new splits
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
 
# Apply chat template
def format_data(example):
    '''
    example["text"] = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}, {"role": "assistant", "content": example["answer"]}],
        tokenize=False, add_generation_prompt=False
    )
    '''
    example["text"] =f"### User: {example['question']}<|eot_id|>### Assistant: {example['answer']}<|eot_id|>" 
    
    return example
 
columns_to_remove = [col for col in train_dataset.column_names if col != "text"]

train_dataset = train_dataset.map(format_data, remove_columns=columns_to_remove)
test_dataset = test_dataset.map(format_data, remove_columns=columns_to_remove)

# Verify the sizes of the new datasets
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
```

#### Create Sparse-GIN model on top of pre-trained LLM

```python
#Load Graph-Aware Transformer library
from xgpt import *

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"

transformer_config = LlamaConfig.from_pretrained(pretrained_model_name)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

tokenizer.padding_side='right'
tokenizer.pad_token = "<|finetune_right_pad_id|>"# tokenizer.eos_token
tokenizer.pad_token,tokenizer.pad_token_id

# Define Sparse-GIN Configuration  
 
gnn_config = GNNConfig(
    num_layers=1,        
    activation="prelu", #"relu" 
    dropout=0.1,
    lambda_GNN_initial = 0.,
    lambda_GNN=0.5,
    norm_to_hidden_states=False,
    use_layer_norm=True, 
    combined_norm=False,
    rms_norm_eps=1e-5,
    hidden_dim=155,

    ### GIN type/approach
    gnn_type='causal_gin', 
    gnn_mode='single', #one GIN, not separate per head
    GIN_use_MLP = True, 
    GIN_hidden_dim_multiplier = 1, # MLP hidden dimension in the GIN

    ### Parameters for adjacency processing
    adj_construction_method='sum', #sum all per-head adj matrices, clamped at 1.0    
    continuous_transform_alpha = 10.0, threshold = 0.1,   
    epsilon_threshold = 0.6, zero_below_epsilon_threshold = True, # All edges below threshold are set to zero
    remove_self_connections = False, 
    GIN_use_norm = False, 
    GIN_edge_weight_scaling = True, # Scale graph edges based on adjacency matrix derived from attention weights

    gnn_residual = False, 
    
    plot_for_debugging=False,

    gnn_logic='before_MLP', #'after_MLP' 'parallel_GNN_MLP',
)

transformer_config._attn_implementation='eager' 

model_with_gnn  = load_model_with_pretrained_transformer ( gnn_config, transformer_config, 
                               pretrained_model_name = pretrained_model_name,
                               attn_implementation='eager',
                               )
count_trainable_parameters(model_with_gnn)

# Move to appropriate device (if necessary)
model_with_gnn.to("cuda" if torch.cuda.is_available() else "cpu")
 
transformer_config = LlamaConfig.from_pretrained(pretrained_model_name)

freeze_except_select(model_with_gnn, unfreeze_keywords=['gnn', 
                                                        'trainable_scale',
                                                        'gnn_norm',
                                                        'combined_norm'
                                                       ], 
                                                       verbose=True)

count_trainable_parameters(model_with_gnn)
```

#### Training
```python
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback

sample_steps    = 1000
max_seq_length  = 1024

# Training arguments and initialization remain the same
training_args = SFTConfig(
    output_dir="./output_dir/",
    eval_strategy="epoch", #"steps",
    eval_steps=sample_steps,
    learning_rate=2e-4, 
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logging_dir/",
    lr_scheduler_type="constant", #"cosine"
    max_seq_length=max_seq_length,
    logging_steps=sample_steps,
    warmup_steps=50,
    dataset_text_field="text",
    packing=False,
    max_grad_norm=0.5,
    report_to='none',
    save_strategy='no',
    do_eval=True,
)

trainer = SFTTrainer(
    model=model_with_gnn,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    #callbacks=[sample_generation_callback],
)

# Train
trainer.train()
```
![image](https://github.com/user-attachments/assets/ba3ed75f-949f-4638-bcd8-f2474e6e3df4)

Figure 4: Trainable scale parameter over all layers in the model, plotted over all epochs, for the Sparse-GIN model. The trainable scale parameter delineates the relative importance of the sparse GIN as it is added to the original signal. The plot illustrates how the scale parameter evolves over both the layer index and the epoch fraction. Early in training, higher layers exhibit stronger scaling values, indicating a higher reliance on sparse GIN adjustments. As training progresses, the scaling values stabilize, suggesting convergence in the relative importance of the sparse GIN contributions across layers. The color gradient reflects the magnitude of the scale parameter, with warmer colors (red) indicating higher values and cooler colors (blue) indicating lower values. This visualization provides insights into the adaptive behavior of the trainable scale parameter over the course of training.

#### Model weights and other datasets

A trained model can be found at the LAMM Hugging Face organization. 

1) ORCA math word problems: [lamm-mit/Llama-3.2-3B-Instruct-Sparse-GIN-orca-math-word-problems](https://huggingface.co/lamm-mit/Llama-3.2-3B-Instruct-Sparse-GIN-orca-math-word-problems).
2) Bio-inspired materials and technology: [lamm-mit/Llama-3.2-3B-Instruct-Sparse-GIN-bio](https://huggingface.co/lamm-mit/Llama-3.2-3B-Instruct-Sparse-GIN-bio)

Task:

```Explain the concept of hierarchical structures in materials science, especially relating proteins and music for use as optical device. Think step by step.```

Output:

```raw
Hierarchical structures are complex arrangements of materials at multiple scales, from the atomic to the macroscopic level.In materials science, these structures are often found in biological materials, such as proteins, which exhibit a hierarchical organization of their molecular building blocks. This hierarchical structure allows for the creation of materials with unique properties and functions.

One example of hierarchical structures in materials science is the use of proteins as optical devices. Proteins can be engineered to have specific optical properties, such as refractive index and birefringence, which can be used to create optical devices with unique functions. For example, proteins can be used to create optical fibers with high refractive index and low birefringence, which can be used for high-speed data transmission.

The hierarchical structure of proteins allows for the creation of materials with unique properties and functions. At the atomic level, proteins are composed of amino acids, which are arranged in a specific sequence to form a three-dimensional structure. This structure is then organized into larger units, such as domains and fibrils, which are arranged in a hierarchical manner.

The hierarchical structure of proteins also allows for the creation of materials with specific optical properties. For example, the refractive index of a protein can be controlled by adjusting the sequence of amino acids and the arrangement of the protein's molecular building blocks. Similarly, the birefringence of a protein can be controlled by adjusting the orientation of the protein's molecular building blocks.

The use of proteins as optical devices is a promising area of research, with potential applications in fields such as telecommunications, optics, and biomedicine. By engineering proteins with specific optical properties, researchers can create materials with unique functions and properties, such as high-speed data transmission and optical sensing.

In conclusion, hierarchical structures are a key feature of materials science, particularly in the field of protein-based materials. The hierarchical organization of proteins allows for the creation of materials with unique properties and functions, such as optical devices with high refractive index and low birefringence. The use of proteins as optical devices is a promising area of research, with potential applications in fields such as telecommunications, optics, and biomedicine.<|eot_id|>
```

## Reference

```bibtex
@article{Buehler2025GraphAwareGPT,
  title={Graph-Aware Isomorphic Attention for Adaptive Dynamics in Transformers},
  author={Markus J. Buehler},
  journal={arXiv},
  year={2025},
  publisher={}
  url={}
}
```
