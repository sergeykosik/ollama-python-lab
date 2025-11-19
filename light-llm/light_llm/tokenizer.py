"""
Tokenizer for Light LLM.

This module provides tokenization functionality optimized for code,
using Byte-Pair Encoding (BPE) with special handling for code structures.
"""

import json
import os
from typing import List, Optional, Union

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class CodeTokenizer:
    """
    Tokenizer optimized for code with BPE encoding.

    This tokenizer handles:
    - Code-specific tokens (indentation, operators, brackets)
    - Multiple programming languages
    - Special tokens (BOS, EOS, PAD, UNK)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        bos_token: str = "<|begin|>",
        eos_token: str = "<|end|>",
        pad_token: str = "<|pad|>",
        unk_token: str = "<|unk|>",
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of the vocabulary
            bos_token: Beginning-of-sequence token
            eos_token: End-of-sequence token
            pad_token: Padding token
            unk_token: Unknown token
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "The 'tokenizers' library is required. "
                "Install it with: pip install tokenizers"
            )

        self.vocab_size = vocab_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Special tokens
        self.special_tokens = [pad_token, bos_token, eos_token, unk_token]
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 3

        # Initialize tokenizer
        self.tokenizer = None
        self._create_tokenizer()

    def _create_tokenizer(self):
        """Create a new BPE tokenizer."""
        self.tokenizer = HFTokenizer(BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    def train(self, files: List[str], vocab_size: Optional[int] = None):
        """
        Train the tokenizer on a corpus of code files.

        Args:
            files: List of file paths to train on
            vocab_size: Vocabulary size (overrides default if provided)
        """
        if vocab_size is None:
            vocab_size = self.vocab_size

        # Define special tokens
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        # Train tokenizer
        self.tokenizer.train(files, trainer)

        # Add post-processing for special tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum length (for padding/truncation)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        # Encode text
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids = encoded.ids

        # Handle truncation
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # Ensure EOS token at the end if special tokens were added
            if add_special_tokens:
                token_ids[-1] = self.eos_token_id

        # Handle padding
        if padding and max_length is not None and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode (can be 1D or 2D list)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text (or list of texts for batch)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        # Handle batch decoding
        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]

        # Decode single sequence
        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """
        Encode a batch of texts.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Encode all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=False,
                truncation=truncation,
            )
            all_token_ids.append(token_ids)

        # Find max length if padding is needed
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_token_ids)

            # Pad all sequences
            padded_ids = []
            attention_masks = []
            for token_ids in all_token_ids:
                padding_length = max_length - len(token_ids)
                padded = token_ids + [self.pad_token_id] * padding_length
                mask = [1] * len(token_ids) + [0] * padding_length
                padded_ids.append(padded)
                attention_masks.append(mask)

            return {
                "input_ids": padded_ids,
                "attention_mask": attention_masks,
            }
        else:
            return {
                "input_ids": all_token_ids,
                "attention_mask": [[1] * len(ids) for ids in all_token_ids],
            }

    def save(self, path: str):
        """
        Save tokenizer to disk.

        Args:
            path: Directory path to save tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        os.makedirs(path, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        """
        Load tokenizer from disk.

        Args:
            path: Directory path containing tokenizer files

        Returns:
            Loaded tokenizer instance
        """
        # Load config
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            pad_token=config["pad_token"],
            unk_token=config["unk_token"],
        )

        # Load trained tokenizer
        tokenizer_path = os.path.join(path, "tokenizer.json")
        tokenizer.tokenizer = HFTokenizer.from_file(tokenizer_path)

        # Update token IDs from config
        tokenizer.bos_token_id = config["bos_token_id"]
        tokenizer.eos_token_id = config["eos_token_id"]
        tokenizer.pad_token_id = config["pad_token_id"]
        tokenizer.unk_token_id = config["unk_token_id"]

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    def __call__(self, text: Union[str, List[str]], **kwargs):
        """
        Convenience method for encoding.

        Args:
            text: Text or list of texts to encode
            **kwargs: Additional arguments for encoding

        Returns:
            Encoded output
        """
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        else:
            return self.batch_encode(text, **kwargs)


def create_simple_tokenizer(vocab_size: int = 32000) -> CodeTokenizer:
    """
    Create a simple pre-configured tokenizer.

    Args:
        vocab_size: Size of vocabulary

    Returns:
        CodeTokenizer instance
    """
    return CodeTokenizer(vocab_size=vocab_size)
