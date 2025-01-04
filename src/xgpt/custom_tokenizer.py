# custom_tokenizer.py

import os
from pathlib import Path
from typing import List, Optional, Union
from transformers import LlamaTokenizerFast
from tokenizers import Tokenizer, processors, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

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
        "こんにちは",  # Non-Latin characters
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

    # Test out-of-vocabulary handling
    print("\nTesting OOV handling:")
    oov_test = "XYZ123!@#未知の単語"  # Mix of unknown chars
    encoded = tokenizer.encode(oov_test, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {repr(oov_test)}")
    print(f"Encoded : {encoded}")
    print(f"Decoded : {repr(decoded)}")

    tokenizer.padding_side,    tokenizer.pad_token

    # When you create your model
    transformer_config.vocab_size=tokenizer.vocab_size

    '''    
    # Initialize a new tokenizer
    tokenizer = Tokenizer(BPE())
    
    # Define special tokens
    
    
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
    
    # Convert the list to the desired format with function calls
    
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
        #pair=None,  # Use this if processing pairs
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


import os
from pathlib import Path
from typing import List, Optional, Union, Iterator
from transformers import LlamaTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace

from pathlib import Path
from typing import List, Optional, Union, Iterator
from transformers import LlamaTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.pre_tokenizers import Metaspace, PreTokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder

def train_tokenizer_from_scratch_NFKC(
    texts: Iterator[str],
    vocab_size: int = 512,
    save_dir: Union[str, Path] = "./custom_tokenizer",
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None
) -> LlamaTokenizerFast:
    """
    Train a custom tokenizer using BPE with NFKC normalization.
    
    Args:
        texts: Iterator of text samples for training
        vocab_size: Size of the vocabulary to learn
        save_dir: Directory to save the tokenizer files
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens. If None, defaults will be used
        
    Returns:
        LlamaTokenizerFast: Trained tokenizer ready for use
        
    Raises:
        ValueError: If vocab_size is less than the number of special tokens
    """
    if special_tokens is None:
        special_tokens = [
            "<|pad|>",
            "<|eot_id|>",
            "<|begin_of_text|>",
            "<|unk|>",
            "<|mask|>",
            "<|sequence|>",
            "<|/sequence|>",
        ]
    
    # Validate vocab size
    if vocab_size <= len(special_tokens):
        raise ValueError(
            f"vocab_size ({vocab_size}) must be greater than the number of special tokens ({len(special_tokens)})"
        )
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    
    # Set up normalizer
    tokenizer.normalizer = Sequence([NFKC()])
    
    # Fixed: Correctly set up pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁")
    
    # Configure and run trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=min_frequency
    )
    
    try:
        tokenizer.train_from_iterator(texts, trainer=trainer)
    except Exception as e:
        raise RuntimeError(f"Failed to train tokenizer: {str(e)}") from e
    
    # Save the base tokenizer
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    # Initialize and configure LlamaTokenizerFast
    llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<|begin_of_txt|>",
        eos_token="<|eot_id|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        mask_token="<|mask|>",
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
    )
    
    # Add sequence tokens and configure padding
    sequence_tokens = ["<|sequence|>", "<|/sequence|>"]
    if sequence_tokens[0] not in llama_tokenizer.get_vocab():
        llama_tokenizer.add_special_tokens({
            'additional_special_tokens': sequence_tokens
        })
    
    llama_tokenizer.padding_side = 'right'
    
    # Save the complete tokenizer configuration
    llama_tokenizer.save_pretrained(save_dir)
    
    return llama_tokenizer

# Example use-case:
# texts = ["Some sample text", "Another line of text"]
# tokenizer = train_tokenizer_from_scratch_NFKC(texts, vocab_size=128)
# encoded = tokenizer("Test <|sequence|> data <|/sequence|>", return_tensors="pt")
# print(encoded)


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
