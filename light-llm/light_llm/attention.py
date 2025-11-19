"""
Attention mechanisms for Light LLM.

This module implements multi-head self-attention with:
- Rotary Position Embeddings (RoPE)
- Causal masking for autoregressive generation
- KV-cache support for efficient inference
- Optional Flash Attention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in RoFormer.

    RoPE encodes positional information by rotating query and key vectors,
    which provides better extrapolation to longer sequences.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            dim: Dimension of each attention head
            max_position_embeddings: Maximum sequence length
            base: Base for the exponential (theta parameter)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache for cos and sin values
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin values for the given sequence length.

        Args:
            x: Input tensor (not used, just for device/dtype inference)
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # If sequence length exceeds cache, rebuild it
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Unsqueeze to match dimensions [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with:
    - Causal masking for autoregressive generation
    - Rotary Position Embeddings (RoPE)
    - KV-cache support for efficient inference
    """

    def __init__(self, config):
        """
        Initialize Multi-Head Attention.

        Args:
            config: LightLLMConfig instance
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(self.attention_dropout)

        # Rotary Position Embedding
        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            past_key_value: Cached (key, value) from previous forward pass
            use_cache: Whether to return key-value cache

        Returns:
            Tuple of (output, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get RoPE embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)

        # Apply rotary position embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV-cache for efficient inference
        if past_key_value is not None:
            # Concatenate with cached keys and values
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Save current key-value if caching is enabled
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Final output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a causal attention mask for autoregressive generation.

    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        dtype: Data type of the mask

    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None) -> torch.Tensor:
    """
    Expand attention mask from [batch_size, seq_len] to [batch_size, 1, target_len, seq_len].

    Args:
        mask: Input mask [batch_size, seq_len]
        dtype: Target dtype
        target_len: Target sequence length (if different from source)

    Returns:
        Expanded mask
    """
    batch_size, seq_len = mask.size()
    target_len = target_len if target_len is not None else seq_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, seq_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), float("-inf"))
