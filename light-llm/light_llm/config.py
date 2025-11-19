"""
Configuration module for Light LLM.

This module defines the model configuration with all hyperparameters
optimized for consumer-grade hardware.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LightLLMConfig:
    """
    Configuration class for Light LLM model.

    This configuration is optimized for training on consumer GPUs (8-16GB VRAM)
    and focuses on code generation tasks.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        hidden_size: Dimensionality of the embeddings and hidden states
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        intermediate_size: Dimensionality of the feed-forward layer
        max_position_embeddings: Maximum sequence length the model can handle
        dropout: Dropout probability for regularization
        layer_norm_eps: Epsilon for layer normalization
        use_cache: Whether to use KV-cache during inference
        pad_token_id: ID of the padding token
        bos_token_id: ID of the beginning-of-sequence token
        eos_token_id: ID of the end-of-sequence token
        tie_word_embeddings: Whether to tie input and output embeddings
        attention_dropout: Dropout probability for attention weights
        initializer_range: Standard deviation for weight initialization
        use_gradient_checkpointing: Whether to use gradient checkpointing (saves memory)
        rope_theta: Base frequency for RoPE (Rotary Position Embeddings)
        rope_scaling: Scaling factor for RoPE
    """

    # Vocabulary and embedding configuration
    vocab_size: int = 32000
    hidden_size: int = 768

    # Architecture configuration
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072  # Typically 4 * hidden_size
    max_position_embeddings: int = 2048

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Model features
    use_cache: bool = True
    tie_word_embeddings: bool = True

    # Training configuration
    use_gradient_checkpointing: bool = False
    initializer_range: float = 0.02

    # Position encoding (RoPE)
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None

    # Activation function
    hidden_act: str = "gelu"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) should be "
                f"greater than hidden_size ({self.hidden_size})"
            )

    @property
    def head_dim(self) -> int:
        """Calculate the dimension of each attention head."""
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LightLLMConfig":
        """Create a config instance from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "initializer_range": self.initializer_range,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
        }


# Predefined configurations for different model sizes
LIGHT_LLM_CONFIGS = {
    "tiny": LightLLMConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
    ),
    "small": LightLLMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
    ),
    "medium": LightLLMConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
    ),
    "large": LightLLMConfig(
        vocab_size=32000,
        hidden_size=1280,
        num_layers=24,
        num_heads=20,
        intermediate_size=5120,
        max_position_embeddings=2048,
    ),
}


def get_config(size: str = "small") -> LightLLMConfig:
    """
    Get a predefined configuration by size.

    Args:
        size: One of "tiny", "small", "medium", "large"

    Returns:
        LightLLMConfig instance

    Example:
        >>> config = get_config("small")
        >>> print(config.hidden_size)
        768
    """
    if size not in LIGHT_LLM_CONFIGS:
        raise ValueError(
            f"Unknown config size: {size}. "
            f"Available sizes: {list(LIGHT_LLM_CONFIGS.keys())}"
        )
    return LIGHT_LLM_CONFIGS[size]
