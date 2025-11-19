"""
Simple example demonstrating Light LLM usage.

This script shows how to:
1. Create and configure a model
2. Train on a small dataset
3. Generate text
"""

import torch
from light_llm import LightLLM, get_config
from light_llm.tokenizer import CodeTokenizer
from light_llm.trainer import Trainer
from light_llm.inference import TextGenerator
from data.dataset import SimpleTextDataset
from torch.utils.data import DataLoader


def create_toy_dataset():
    """Create a small toy dataset for demonstration."""
    # Simple Python code examples
    examples = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height",
        "for i in range(10):\n    print(i)",
        "if x > 0:\n    print('positive')\nelse:\n    print('negative')",
    ] * 10  # Repeat to have more examples

    return examples


def main():
    print("=" * 80)
    print("Light LLM - Simple Example")
    print("=" * 80)

    # 1. Create a tiny model for quick testing
    print("\n1. Creating model...")
    config = get_config("tiny")
    config.vocab_size = 1000  # Small vocab for toy example
    config.max_position_embeddings = 128

    model = LightLLM(config)
    print(f"Model created with {model.get_num_params():,} parameters")

    # 2. Create a simple tokenizer (in practice, you'd train this properly)
    print("\n2. Creating tokenizer...")
    # For this example, we'll use character-level tokenization
    # In practice, you should train a proper BPE tokenizer

    # Simple character tokenizer for demonstration
    class SimpleCharTokenizer:
        def __init__(self):
            self.vocab = {chr(i): i for i in range(32, 127)}
            self.vocab['<|pad|>'] = 0
            self.vocab['<|begin|>'] = 1
            self.vocab['<|end|>'] = 2
            self.vocab['<|unk|>'] = 3
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = 3
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        def encode(self, text, add_special_tokens=True, **kwargs):
            tokens = [self.bos_token_id] if add_special_tokens else []
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_token_id))
            if add_special_tokens:
                tokens.append(self.eos_token_id)
            return tokens

        def decode(self, tokens, skip_special_tokens=True):
            text = ""
            for token in tokens:
                if skip_special_tokens and token in [0, 1, 2, 3]:
                    continue
                text += self.reverse_vocab.get(token, "")
            return text

        def __len__(self):
            return len(self.vocab)

    tokenizer = SimpleCharTokenizer()

    # Update model config with actual vocab size
    model.config.vocab_size = len(tokenizer)
    model = LightLLM(model.config)

    print(f"Tokenizer created with vocabulary size: {len(tokenizer)}")

    # 3. Create dataset
    print("\n3. Creating dataset...")
    examples = create_toy_dataset()
    dataset = SimpleTextDataset(examples, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset created with {len(dataset)} examples")

    # 4. Train for a few steps (very short training for demo)
    print("\n4. Training for a few steps...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        learning_rate=1e-3,
        max_steps=50,  # Very short training
        gradient_accumulation_steps=1,
        mixed_precision=False,  # Disable for toy example
        device=device,
        output_dir="./example_checkpoints",
        logging_steps=10,
        eval_steps=1000,  # Skip evaluation for this example
        save_steps=1000,  # Don't save checkpoints for this example
    )

    trainer.train()

    # 5. Generate text
    print("\n5. Generating text...")
    generator = TextGenerator(model, tokenizer, device=device)

    prompts = [
        "def ",
        "class ",
        "for ",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        try:
            generated = generator.generate(
                prompt=prompt,
                max_length=20,
                temperature=0.8,
                do_sample=True,
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Generation failed: {e}")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nNote: This is a toy example with minimal training.")
    print("For real use cases, you should:")
    print("  1. Train a proper BPE tokenizer on your code corpus")
    print("  2. Use a larger dataset")
    print("  3. Train for many more steps")
    print("  4. Use a larger model if you have the resources")


if __name__ == "__main__":
    main()
