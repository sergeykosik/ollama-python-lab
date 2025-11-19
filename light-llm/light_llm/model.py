"""
Main model architecture for Light LLM.

This module implements the complete transformer-based language model
optimized for code generation on consumer hardware.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LightLLMConfig
from .attention import MultiHeadAttention, _make_causal_mask


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) with GELU activation.
    """

    def __init__(self, config: LightLLMConfig):
        """
        Initialize MLP.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Activation function
        if config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "relu":
            self.act_fn = F.relu
        elif config.hidden_act == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU-style gating
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    - Multi-head self-attention
    - Feed-forward network (MLP)
    - Layer normalization (pre-norm)
    - Residual connections
    """

    def __init__(self, config: LightLLMConfig):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Layer normalization (pre-norm)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention and MLP
        self.self_attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of transformer block.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            past_key_value: Cached key-value from previous forward pass
            use_cache: Whether to return key-value cache

        Returns:
            Tuple of (output, past_key_value)
        """
        # Self-attention with residual connection (pre-norm)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        attn_output = self.dropout(attn_output)
        hidden_states = residual + attn_output

        # MLP with residual connection (pre-norm)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.dropout(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states, past_key_value


class LightLLM(nn.Module):
    """
    Light LLM: A lightweight transformer-based language model
    optimized for code generation on consumer hardware.
    """

    def __init__(self, config: LightLLMConfig):
        """
        Initialize Light LLM.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying (share embeddings between input and output)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights using normal distribution.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        """Return input embeddings."""
        return self.token_embedding

    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.token_embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass of Light LLM.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: List of cached key-values for each layer
            use_cache: Whether to return key-value cache
            labels: Labels for computing language modeling loss [batch_size, seq_len]

        Returns:
            Tuple of (logits, loss, past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # Create causal mask
        causal_mask = _make_causal_mask(seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to [batch_size, 1, 1, seq_len]
            expanded_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            # Invert mask (1 -> 0, 0 -> -inf)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min
            # Combine with causal mask
            causal_mask = causal_mask + expanded_mask

        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Initialize cache for current forward pass
        present_key_values = [] if use_cache else None

        # Pass through transformer blocks
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=use_cache)
                    return custom_forward

                hidden_states, present_key_value = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    causal_mask,
                    past_key_value,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            if use_cache:
                present_key_values.append(present_key_value)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return logits, loss, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p
            do_sample: Whether to use sampling (vs greedy decoding)
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs [batch_size, generated_seq_len]
        """
        self.eval()

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track which sequences have finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        past_key_values = None

        for _ in range(max_length):
            # Forward pass
            logits, _, past_key_values = self(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get logits for next token
            next_token_logits = logits[:, -1, :]

            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Update unfinished sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # Check for EOS token
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: Whether to exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
