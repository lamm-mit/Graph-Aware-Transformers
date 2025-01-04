# custom_tokenizer.py

from pathlib import Path
from typing import Union
from transformers import LlamaTokenizerFast
from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import matplotlib.pyplot as plt
from datetime import datetime

def train_tokenizer_from_scratch(texts, vocab_size=128,
                                 save_dir: Union[str, Path] = "./custom_tokenizer",
                                 special_tokens = [
                                    "<|pad|>",
                                    "<|eot_id|>", 
                                    "<|begin_of_text|>",
                                    "<|unk|>",
                                    "<|mask|>",
                                    "<|sequence|>",
                                    "<|/sequence|>",
                                ],
                                ):
    '''
    Example use:

    # Train the tokenizer
    texts = train_dataset['text']
    tokenizer = train_tokenizer_from_scratch(texts, vocab_size=128)

    # Save the trained tokenizer
    tokenizer.save_pretrained("./custom_tokenizer")

    # Test with various scenarios
    test_cases = [
        "<|begin_of_text|><|sequence|>A A A I<|/sequence|>",  # Simple space-separated
        "<|begin_of_text|><|sequence|>AAAI<|/sequence|>",  # Simple space-separated
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

    # When you create your model
    transformer_config.vocab_size = tokenizer.vocab_size
    '''    

    # Initialize a new tokenizer
    tokenizer = Tokenizer(BPE())
    
    # Initialize the trainer with byte-level alphabet
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        min_frequency=2,
        continuing_subword_prefix="",
        end_of_word_suffix=""
    )
    
    # Use ByteLevel pre-tokenizer but customize its behavior
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    
    # Set the decoder to handle spaces properly
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False, use_regex=True)
    
    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    formatted_special_tokens = [
                (token, tokenizer.token_to_id(token)) for token in special_tokens
            ]

    # Set up TemplateProcessing
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens= formatted_special_tokens
        )
    
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = save_dir / "tokenizer.json"
    try:
        tokenizer.save(str(tokenizer_path))

    except Exception as e:
        raise OSError(f"Failed to save tokenizer: {str(e)}")
    
    # Convert to LlamaTokenizerFast with custom settings
    llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file="./custom_tokenizer/tokenizer.json",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        mask_token="<|mask|>",
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=True,
    )
    
    # Add special tokens to ensure they're properly registered
    llama_tokenizer.add_special_tokens({
        'pad_token': "<|pad|>",
        'bos_token': "<|begin_of_text|>",
        'eos_token': "<|eot_id|>",
        'unk_token': "<|unk|>",
        'mask_token': "<|mask|>",
        'additional_special_tokens': ["<|sequence|>", "<|/sequence|>"]
    })

    llama_tokenizer.padding_side ='right'
    llama_tokenizer.pad_token = "<|pad|>" 
    return llama_tokenizer

def plot_token_length_histogram(dataset, tokenizer):
    
    # Calculate token lengths for each example in the dataset
    token_lengths = [len(tokenizer.encode(example["text"])) for example in dataset]

    # Plot histogram of token lengths
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=30, edgecolor='black')
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Token Length Distribution in Dataset")

    # Save plot as SVG with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"token_length_histogram_{timestamp}.svg"

    plt.savefig(filename, format="svg")

    plt.show()
