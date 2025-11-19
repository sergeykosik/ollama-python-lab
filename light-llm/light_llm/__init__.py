"""
Light LLM: A lightweight transformer-based language model for code generation.

This package provides a complete implementation of a transformer model
optimized for training and inference on consumer-grade hardware.
"""

from .config import LightLLMConfig, get_config, LIGHT_LLM_CONFIGS
from .model import LightLLM, TransformerBlock, MLP
from .attention import MultiHeadAttention, RotaryPositionEmbedding
from .tokenizer import CodeTokenizer, create_simple_tokenizer

__version__ = "0.1.0"

__all__ = [
    "LightLLMConfig",
    "get_config",
    "LIGHT_LLM_CONFIGS",
    "LightLLM",
    "TransformerBlock",
    "MLP",
    "MultiHeadAttention",
    "RotaryPositionEmbedding",
    "CodeTokenizer",
    "create_simple_tokenizer",
]
