# utils.py

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from .llama_for_causal_lm_with_gnn import LlamaForCausalLMWithGNN
from .gnn_config import GNNConfig
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import pandas as pd
from tqdm.notebook import tqdm

from transformers import TrainerCallback
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import CrossEntropyLoss
import numpy as np

def count_trainable_parameters(model):
    """
    Print the number of trainable parameters in a given model.
    
    Args:
        model (torch.nn.Module): The model for which to count trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

def freeze_except_select(model, unfreeze_keywords=["gnn"], 
                        # exclude_keywords=[],
                         verbose=False):
    """
    Freeze all model weights except for the GNN layers and trainable_scale parameters.
    
    Args:
        model (torch.nn.Module): The model in which to freeze weights.
        layer_keyword (str): Keyword to identify which layers to unfreeze (default is "gnn").
        verbose (bool): If True, print the names of unfrozen parameters.
    """
    unfrozen_params = []  # List to keep track of unfrozen parameters
    
    for name, param in model.named_parameters():
        # Freeze all parameters by default
        param.requires_grad = False

        for layer_keyword in unfreeze_keywords:# and layer_keyword not in exclude_keywords:
            # Unfreeze parameters in the specified layers or trainable_scale
            if layer_keyword in name:
            #if layer_keyword in name or "trainable_scale" in name:
                param.requires_grad = True
                unfrozen_params.append(name)  # Keep track of unfrozen parameters
                
                if verbose:
                    print(f"Unfrozen parameter: {name}")
    
    print("Freezing complete. Only layers containing '{}' are trainable.".format(unfreeze_keywords))
    return unfrozen_params

def freeze_except_original_no_trainable_scale(model, unfreeze_keywords=["gnn"], verbose=False):
    """
    Freeze all model weights except for the original model layers, excluding GNN and trainable_scale parameters.
    
    Args:
        model (torch.nn.Module): The model in which to freeze weights.
        gnn_keyword (str): Keyword to identify GNN layers that should remain frozen.
        verbose (bool): If True, print the names of unfrozen parameters.
    """
    unfrozen_params = []  # List to keep track of unfrozen parameters

    
    for name, param in model.named_parameters():
        # Freeze all parameters by default
        param.requires_grad = False

        for keyword in unfreeze_keywords:
            # Unfreeze parameters that are part of the original model and not 'gnn' or 'trainable_scale'
            if keyword not in name and "trainable_scale" not in name:
                param.requires_grad = True
                unfrozen_params.append(name)  # Keep track of unfrozen parameters
                
                if verbose:
                    print(f"Unfrozen parameter: {name}")
    
    print("Freezing complete. Only original model layers are trainable, excluding 'gnn' and 'trainable_scale'.")
    return unfrozen_params

    
def unfreeze_all(model):
    """
    Freeze all model weights except for the GNN layers.
    
    Args:
        model (torch.nn.Module): The model in which to freeze weights.
    """
    for name, param in model.named_parameters():
        # Unfreeze all parameters  
        param.requires_grad = True
 

    print("Unfreezing complete. Entire model trainable.")
  

def load_model_with_pretrained_transformer(gnn_config, transformer_config, 
                      pretrained_model_name = None,device='cuda',
                      torch_dtype='bfloat16',
                      attn_implementation="flash_attention_2",
                    ):
    
    # Define Pretrained Model Name
    
    transformer_config.use_cache=False
    transformer_config._attn_implementation=attn_implementation #'eager'

    transformer_config.gnn_config= gnn_config 
    
    # Initialize the Enhanced LLaMA Model with GNN
    transformer_config._attn_implementation=attn_implementation
    transformer_config.torch_dtype=torch_dtype
    
    model_with_gnn = LlamaForCausalLMWithGNN(config=transformer_config,).to(device)

    if pretrained_model_name != None:
        print ("Load pretrained xeights to transformer part of the model, GNN still needs to be trained.")
        
        # Load Pretrained LLaMA Weights
        pretrained_llama = LlamaForCausalLM.from_pretrained(pretrained_model_name, 
                                                            torch_dtype=torch_dtype,
                                                            attn_implementation=attn_implementation,
                    ).to(device)
    
        # Copy embedding and final layer weights
        model_with_gnn.model.embed_tokens.weight.data.copy_(pretrained_llama.model.embed_tokens.weight.data)
        model_with_gnn.model.norm.weight.data.copy_(pretrained_llama.model.norm.weight.data)
        model_with_gnn.lm_head.weight.data.copy_(pretrained_llama.lm_head.weight.data)
    
        # Align and copy transformer layer weights
        for layer_idx in range(len(model_with_gnn.model.layers)):
            pretrained_layer = pretrained_llama.model.layers[layer_idx]
            custom_layer = model_with_gnn.model.layers[layer_idx] 
    
            # Copy self-attention projection weights
            custom_layer.self_attn.q_proj.weight.data.copy_(pretrained_layer.self_attn.q_proj.weight.data)
            custom_layer.self_attn.k_proj.weight.data.copy_(pretrained_layer.self_attn.k_proj.weight.data)
            custom_layer.self_attn.v_proj.weight.data.copy_(pretrained_layer.self_attn.v_proj.weight.data)
            custom_layer.self_attn.o_proj.weight.data.copy_(pretrained_layer.self_attn.o_proj.weight.data)
    
            # Copy MLP weights
            custom_layer.mlp.gate_proj.weight.data.copy_(pretrained_layer.mlp.gate_proj.weight.data)
            custom_layer.mlp.up_proj.weight.data.copy_(pretrained_layer.mlp.up_proj.weight.data)
            custom_layer.mlp.down_proj.weight.data.copy_(pretrained_layer.mlp.down_proj.weight.data)
    
            # Copy layer normalization weights
            custom_layer.input_layernorm.weight.data.copy_(pretrained_layer.input_layernorm.weight.data)
            custom_layer.post_attention_layernorm.weight.data.copy_(pretrained_layer.post_attention_layernorm.weight.data)
    
        # Verification: Check weight differences after loading
        mismatches = []
        for custom_key, pretrained_key in zip(model_with_gnn.state_dict().keys(), pretrained_llama.state_dict().keys()):
            if custom_key.startswith('model.layers') and 'llama_decoder_layer' in custom_key:
                adjusted_key = custom_key.replace('llama_decoder_layer.', '')
                if adjusted_key in pretrained_llama.state_dict():
                    custom_weight = model_with_gnn.state_dict()[custom_key]
                    pretrained_weight = pretrained_llama.state_dict()[adjusted_key]
                    if not torch.equal(custom_weight, pretrained_weight):
                        mismatches.append((custom_key, adjusted_key))
    
        if mismatches:
            for custom_key, pretrained_key in mismatches:
                print(f"Parameter name mismatch: Custom model parameter '{custom_key}' does not match pretrained parameter '{pretrained_key}'")
        else:
            print("All parameters match between the models.")

    return model_with_gnn 
 
import matplotlib.cm as cm
import numpy as np

def extract_gnn_parameters_with_labels(model_with_gnn, parameters_to_extract=None):
    """
    Extract specified parameters from the provided model with GNN layers. Organize labels by transformer layer and GIN layer.
    
    Args:
        model_with_gnn: The model containing GNN layers.
        parameters_to_extract (list): List of parameter names to extract. Defaults to common parameters.
    
    Returns:
        dict: Extracted parameters as a dictionary.
        list: Labels indicating transformer and GIN layer.
    """
    # Default parameters to extract if not provided
    if parameters_to_extract is None:
        parameters_to_extract = [
            "attention_epsilon",
            "residual_epsilon",
            "sharpening_parameters",
            "soft_masking_tau"
        ]
    
    # Initialize parameters dictionary
    params = {key: [] for key in parameters_to_extract}
    labels = []

    layers = model_with_gnn.model.layers

    for transformer_idx, layer in enumerate(layers):
        for param_name in parameters_to_extract:
            if hasattr(layer.self_attn, param_name):
                param_list = getattr(layer.self_attn, param_name)
                for gin_idx, p in enumerate(param_list):
                    params[param_name].append(p.item())
                    if param_name == parameters_to_extract[0]:  # Add labels only once
                        labels.append(f"Transformer {transformer_idx}, GIN {gin_idx}")

    return params, labels

def plot_gnn_parameters_vertical_with_layer_connections_and_ylabels(model_with_gnn, parameters_to_plot=None):
    """
    Extract and plot selected values from the model with GNN layers in a vertical layout.
    Connect lines for each transformer layer with thicker lines and connect different layers with thinner lines.
    Use a more sophisticated color palette for distinction. Add y-labels instead of titles.

    Args:
        model_with_gnn: The model containing GNN layers.
        parameters_to_plot (list): List of parameter names to extract and plot.
    Example usage
        parameters_to_extract = ["attention_epsilon", "residual_epsilon", "sharpening_parameters"]
        plot_gnn_parameters_vertical_with_layer_connections_and_ylabels(model_with_gnn, parameters_to_extract)
    """
    # Extract parameters and labels
    params, labels = extract_gnn_parameters_with_labels(model_with_gnn, parameters_to_plot)

    # Default parameters to plot if not provided
    if parameters_to_plot is None:
        parameters_to_plot = list(params.keys())  # Plot everything extracted

    # Assign distinct colors using a sophisticated palette
    num_transformer_layers = len(set(label.split(",")[0] for label in labels))  # Unique transformer layers
    color_map = cm.get_cmap("viridis", num_transformer_layers)  # Sophisticated color palette
    transformer_colors = {
        f"Transformer {i}": color_map(i / (num_transformer_layers - 1)) for i in range(num_transformer_layers)
    }
    
    # Create vertical plots for the selected parameters
    num_plots = len(parameters_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 3.5 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable if only one plot is created

    for ax, param_name in zip(axes, parameters_to_plot):
        values = params[param_name]
        
        # Connect lines for each transformer layer with thicker lines
        for transformer_idx in range(num_transformer_layers):
            transformer_indices = [
                i for i, label in enumerate(labels)
                if f"Transformer {transformer_idx}" in label
            ]
            transformer_values = [values[i] for i in transformer_indices]
            
            # Connect values between different layers with thinner lines
            ax.plot(
                range(len(values)),
                values,
                linestyle="--",
                linewidth=1.0,  # Thinner line for between-layer connections
                color="gray",
                alpha=0.5,
            )
            
            ax.plot(
                transformer_indices,
                transformer_values,
                marker="o",
                linewidth=2.5,  # Thicker line for within-layer connections
                label=f"Transformer {transformer_idx}",
                color=transformer_colors[f"Transformer {transformer_idx}"],
                alpha=1.0,
            )
        
        # Add y-label for each plot
        ax.set_ylabel(param_name.replace("_", " ").capitalize())
        ax.grid()

    # Add x-axis labels to the bottom plot only
    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=90)

    plt.tight_layout()

    # Save as SVG with timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./gnn_parameters_selected_vertical_with_ylabels_{timestamp}.svg"
    #plt.savefig(filename, format="svg")
    plt.show()
    print(f"Plot saved to {filename}")

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_loss_over_epochs(trainer, num_epochs=5):
    # Extract training and validation losses from trainer.state.log_history
    log_history = trainer.state.log_history

    # Separate out training and validation loss with step values
    train_loss = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry]
    val_loss = [(entry['step'], entry['eval_loss']) for entry in log_history if 'eval_loss' in entry]

    # Print train_loss and val_loss lists to match your format
    print("Train Loss:", train_loss)
    print("Validation Loss:", val_loss)

    # Extract steps and losses separately for plotting
    train_steps, train_losses = zip(*train_loss)
    val_steps, val_losses = zip(*val_loss)

    # Normalize steps to epochs
    max_step = max(train_steps[-1], val_steps[-1])
    normalized_train_epochs = np.array(train_steps) / max_step * num_epochs
    normalized_val_epochs = np.array(val_steps) / max_step * num_epochs

    # Plot training and validation loss
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 6))
    plt.plot(normalized_train_epochs, train_losses, marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.plot(normalized_val_epochs, val_losses, marker='o', linestyle='-', color='orange', label='Validation Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend(loc='upper right', title="Loss Type")

    # Save plot as SVG with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"train_val_loss_over_epochs_{timestamp}.svg"
    plt.savefig(filename, format="svg")
    plt.show()

    # Find the minimum value and its corresponding step
    min_step_train, min_value_train = min(train_loss, key=lambda x: x[1])
    min_step_test, min_value_test = min(val_loss, key=lambda x: x[1])
    
    print ("Min val loss: ", min_value_test, "at step: ", min_step_test)
    print ("Min train loss: ", min_value_train, "at step: ", min_step_train)

    return normalized_train_epochs, train_loss, normalized_val_epochs, val_loss

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_average_loss_per_epoch(data, num_epochs=5):
    """
    Plots average training and validation loss per epoch.

    Args:
    - data: Tuple of two lists (training_loss, validation_loss), where each list contains
            tuples of (step, loss).
    - num_epochs: Total number of epochs for normalization.
    """
    # Unpack training and validation data
    train_loss, val_loss = data

    # Extract steps and losses separately
    train_steps, train_losses = zip(*train_loss)
    val_steps, val_losses = zip(*val_loss)

    # Normalize steps to epochs
    max_step = max(train_steps[-1], val_steps[-1])

    # Calculate average loss per epoch
    train_avg_loss = []
    val_avg_loss = []
    epoch_range = np.arange(1, num_epochs + 1)

    for epoch in epoch_range:
        train_epoch_losses = [loss for step, loss in zip(train_steps, train_losses) if int(step / max_step * num_epochs) + 1 == epoch]
        val_epoch_losses = [loss for step, loss in zip(val_steps, val_losses) if int(step / max_step * num_epochs) + 1 == epoch]
        
        train_avg_loss.append(np.mean(train_epoch_losses) if train_epoch_losses else None)
        val_avg_loss.append(np.mean(val_epoch_losses) if val_epoch_losses else None)

    # Plot average training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_range, train_avg_loss, marker='o', linestyle='-', color='blue', label='Avg Training Loss')
    plt.plot(epoch_range, val_avg_loss, marker='o', linestyle='-', color='orange', label='Avg Validation Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Average Training and Validation Loss per Epoch")
    plt.legend(loc='upper right', title="Loss Type")

    # Save plot as SVG with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"avg_train_val_loss_per_epoch_{timestamp}.svg"
    plt.savefig(filename, format="svg")
    plt.show()

    # Print minimum loss details
    min_train_loss_epoch = epoch_range[np.nanargmin(train_avg_loss)]
    min_val_loss_epoch = epoch_range[np.nanargmin(val_avg_loss)]
    
    print("Minimum Training Loss:", np.nanmin(train_avg_loss), "at Epoch:", min_train_loss_epoch)
    print("Minimum Validation Loss:", np.nanmin(val_avg_loss), "at Epoch:", min_val_loss_epoch)

    return train_avg_loss, val_avg_loss

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_average_loss_per_epoch(trainer, num_epochs=5):
    # Extract training and validation losses from trainer.state.log_history
    log_history = trainer.state.log_history

    # Separate out training and validation loss with step values
    train_loss = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry]
    val_loss = [(entry['step'], entry['eval_loss']) for entry in log_history if 'eval_loss' in entry]

    # Print train_loss and val_loss lists to match your format
    print("Train Loss:", train_loss)
    print("Validation Loss:", val_loss)

    # Extract steps and losses separately for calculation
    train_steps, train_losses = zip(*train_loss)
    val_steps, val_losses = zip(*val_loss)

    # Normalize steps to epochs
    max_step = max(train_steps[-1], val_steps[-1])

    # Calculate average loss per epoch
    train_avg_loss = []
    val_avg_loss = []
    epoch_range = np.arange(1, num_epochs + 1)

    for epoch in epoch_range:
        train_epoch_losses = [loss for step, loss in zip(train_steps, train_losses) if int(step / max_step * num_epochs) + 1 == epoch]
        val_epoch_losses = [loss for step, loss in zip(val_steps, val_losses) if int(step / max_step * num_epochs) + 1 == epoch]
        
        train_avg_loss.append(np.mean(train_epoch_losses) if train_epoch_losses else None)
        val_avg_loss.append(np.mean(val_epoch_losses) if val_epoch_losses else None)

    # Plot average training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_range, train_avg_loss, marker='o', linestyle='-', color='blue', label='Avg Training Loss')
    plt.plot(epoch_range, val_avg_loss, marker='o', linestyle='-', color='orange', label='Avg Validation Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Average Training and Validation Loss per Epoch")
    plt.legend(loc='upper right', title="Loss Type")

    # Save plot as SVG with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"avg_train_val_loss_per_epoch_{timestamp}.svg"
    plt.savefig(filename, format="svg")
    plt.show()

    # Print minimum loss details
    min_train_loss_epoch = epoch_range[np.nanargmin(train_avg_loss)]
    min_val_loss_epoch = epoch_range[np.nanargmin(val_avg_loss)]
    
    print("Minimum Training Loss:", np.nanmin(train_avg_loss), "at Epoch:", min_train_loss_epoch)
    print("Minimum Validation Loss:", np.nanmin(val_avg_loss), "at Epoch:", min_val_loss_epoch)

    return train_avg_loss, val_avg_loss

# Example Usage
#plot_average_loss_per_epoch(trainer, num_epochs=5)