"""
Inference engine for Light LLM.

This module provides efficient text generation with:
- Multiple sampling strategies (greedy, top-k, top-p, temperature)
- KV-cache for efficient generation
- Batch inference support
- Streaming generation
"""

from typing import Optional, List, Union, Iterator
import torch
import torch.nn.functional as F


class TextGenerator:
    """
    Text generator for Light LLM with various sampling strategies.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize text generator.

        Args:
            model: Trained Light LLM model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop_tokens: List of tokens to stop generation

        Returns:
            Generated text (or list of texts if num_return_sequences > 1)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long, device=self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode
        generated_texts = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)

            # Handle stop tokens
            if stop_tokens is not None:
                for stop_token in stop_tokens:
                    if stop_token in text:
                        text = text[:text.index(stop_token)]

            generated_texts.append(text)

        return generated_texts[0] if num_return_sequences == 1 else generated_texts

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming output (yields tokens as they're generated).

        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            stop_tokens: List of tokens to stop generation

        Yields:
            Generated text chunks
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        past_key_values = None
        generated_text = prompt

        for _ in range(max_length):
            # Forward pass
            logits, _, past_key_values = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Decode token
            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)

            # Check for stop tokens
            should_stop = False
            if stop_tokens is not None:
                for stop_token in stop_tokens:
                    if stop_token in generated_text + token_text:
                        should_stop = True
                        break

            if should_stop:
                break

            generated_text += token_text
            yield token_text

            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    @torch.no_grad()
    def complete_code(
        self,
        code: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Complete code snippet.

        Args:
            code: Partial code to complete
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (lower for code)
            top_p: Nucleus sampling threshold
            stop_tokens: Tokens to stop generation (e.g., ['\n\n', 'def ', 'class '])

        Returns:
            Completed code
        """
        if stop_tokens is None:
            # Default stop tokens for code completion
            stop_tokens = ['\n\n\n', '```']

        return self.generate(
            prompt=code,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=top_p,
            do_sample=True,
            stop_tokens=stop_tokens,
        )

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts: List of input prompts
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold

        Returns:
            List of generated texts
        """
        # Encode all prompts
        encoded = self.tokenizer.batch_encode(prompts, padding=True)
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode
        generated_texts = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text under the model.

        Args:
            text: Input text

        Returns:
            Perplexity value
        """
        # Encode text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=self.device)
        labels = torch.tensor([token_ids[1:]], dtype=torch.long, device=self.device)

        # Forward pass
        _, loss, _ = self.model(input_ids=input_ids, labels=labels)

        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        return perplexity


def load_model_for_inference(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    from .model import LightLLM
    from .config import LightLLMConfig

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model from config
    config = LightLLMConfig.from_dict(checkpoint['config'])
    model = LightLLM(config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float = 1.2) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size]
        token_ids: Previously generated token IDs [batch_size, seq_len]
        penalty: Penalty factor (> 1.0 discourages repetition)

    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits

    batch_size = logits.size(0)
    for i in range(batch_size):
        unique_tokens = torch.unique(token_ids[i])
        logits[i, unique_tokens] = logits[i, unique_tokens] / penalty

    return logits


def beam_search(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    beam_width: int = 4,
    device: str = "cuda",
) -> str:
    """
    Generate text using beam search.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum length to generate
        beam_width: Number of beams to keep
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Initialize beams: (score, sequence)
    beams = [(0.0, input_ids)]

    for _ in range(max_length):
        new_beams = []

        for score, sequence in beams:
            # Forward pass
            with torch.no_grad():
                logits, _, _ = model(input_ids=sequence, use_cache=False)
                next_token_logits = logits[:, -1, :]

            # Get top-k tokens
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            # Create new beams
            for log_prob, token_id in zip(top_log_probs[0], top_indices[0]):
                new_score = score + log_prob.item()
                new_sequence = torch.cat([sequence, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                new_beams.append((new_score, new_sequence))

        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # Check if all beams have EOS token
        if all(sequence[0, -1].item() == tokenizer.eos_token_id for _, sequence in beams):
            break

    # Return best beam
    best_sequence = beams[0][1]
    generated_text = tokenizer.decode(best_sequence[0].tolist(), skip_special_tokens=True)

    return generated_text
