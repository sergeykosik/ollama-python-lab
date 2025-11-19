"""
Text generation script for Light LLM.

This script provides inference capabilities with:
- Text generation from prompts
- Code completion
- Batch generation
- Streaming output

Usage:
    python generate.py --checkpoint ./checkpoints/best_model.pt --prompt "def fibonacci(n):"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from light_llm.inference import TextGenerator, load_model_for_inference
from light_llm.tokenizer import CodeTokenizer
from light_llm.utils import print_model_info, get_gpu_memory_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with Light LLM")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer",
        help="Path to tokenizer directory",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Keep only top k tokens with highest probability",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Keep top tokens with cumulative probability >= top_p",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalty for repeating tokens",
    )
    parser.add_argument(
        "--stop_tokens",
        type=str,
        nargs="+",
        default=None,
        help="Tokens to stop generation",
    )

    # Mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "complete", "interactive", "batch", "streaming"],
        help="Generation mode",
    )
    parser.add_argument(
        "--code_mode",
        action="store_true",
        help="Optimize for code generation (lower temperature, specific stop tokens)",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for generated text",
    )
    parser.add_argument(
        "--show_model_info",
        action="store_true",
        help="Show model information",
    )

    return parser.parse_args()


def interactive_mode(generator: TextGenerator, args):
    """
    Interactive generation mode.

    Args:
        generator: TextGenerator instance
        args: Command line arguments
    """
    print("=" * 80)
    print("Light LLM - Interactive Mode")
    print("=" * 80)
    print("Enter your prompt (or 'quit' to exit, 'clear' to clear history)")
    print("=" * 80 + "\n")

    while True:
        try:
            prompt = input("\n> ")

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if prompt.lower() == "clear":
                print("\n" * 50)  # Clear screen
                continue

            if not prompt.strip():
                continue

            print("\nGenerating...\n")

            # Generate
            generated = generator.generate(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
                repetition_penalty=args.repetition_penalty,
                stop_tokens=args.stop_tokens,
            )

            # Print results
            if isinstance(generated, list):
                for i, text in enumerate(generated):
                    print(f"\n--- Generated {i+1} ---")
                    print(text)
            else:
                print(generated)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")


def streaming_mode(generator: TextGenerator, prompt: str, args):
    """
    Streaming generation mode.

    Args:
        generator: TextGenerator instance
        prompt: Input prompt
        args: Command line arguments
    """
    print("Prompt:", prompt)
    print("\nGenerating (streaming):\n")
    print(prompt, end="", flush=True)

    for token in generator.generate_streaming(
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=args.stop_tokens,
    ):
        print(token, end="", flush=True)

    print("\n")


def batch_mode(generator: TextGenerator, prompts: list, args):
    """
    Batch generation mode.

    Args:
        generator: TextGenerator instance
        prompts: List of prompts
        args: Command line arguments
    """
    print(f"Generating for {len(prompts)} prompts...\n")

    generated = generator.batch_generate(
        prompts=prompts,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Print results
    for i, (prompt, text) in enumerate(zip(prompts, generated)):
        print(f"\n{'=' * 80}")
        print(f"Prompt {i+1}: {prompt}")
        print(f"{'=' * 80}")
        print(text)
        print()

    # Save to file if specified
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for prompt, text in zip(prompts, generated):
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {text}\n")
                f.write("-" * 80 + "\n\n")
        print(f"Results saved to {args.output_file}")


def main():
    """Main inference function."""
    args = parse_args()

    # Apply code mode settings
    if args.code_mode:
        args.temperature = 0.7
        if args.stop_tokens is None:
            args.stop_tokens = ["\n\n\n", "```", "def ", "class "]
        print("Code mode enabled (optimized for code generation)")

    print("=" * 80)
    print("Light LLM Inference")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}...")
    tokenizer = CodeTokenizer.load(args.tokenizer_path)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_for_inference(args.checkpoint, device=args.device)
    print("Model loaded successfully")

    # Show model info if requested
    if args.show_model_info:
        print_model_info(model)

    # Create generator
    generator = TextGenerator(model, tokenizer, device=args.device)

    # Run generation based on mode
    if args.mode == "interactive":
        interactive_mode(generator, args)

    elif args.mode == "streaming":
        if args.prompt is None:
            print("Error: --prompt required for streaming mode")
            return
        streaming_mode(generator, args.prompt, args)

    elif args.mode == "batch":
        # Load prompts from file
        if args.prompt_file is None:
            print("Error: --prompt_file required for batch mode")
            return

        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        batch_mode(generator, prompts, args)

    elif args.mode in ["generate", "complete"]:
        # Single generation
        if args.prompt is None:
            print("Error: --prompt required for generate/complete mode")
            return

        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...\n")

        if args.mode == "complete":
            # Code completion mode
            generated = generator.complete_code(
                code=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_tokens=args.stop_tokens,
            )
        else:
            # Standard generation
            generated = generator.generate(
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
                repetition_penalty=args.repetition_penalty,
                stop_tokens=args.stop_tokens,
            )

        # Print results
        print("=" * 80)
        print("Generated Text:")
        print("=" * 80)
        if isinstance(generated, list):
            for i, text in enumerate(generated):
                print(f"\n--- Sequence {i+1} ---")
                print(text)
        else:
            print(generated)
        print("\n" + "=" * 80)

        # Save to file if specified
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                if isinstance(generated, list):
                    for i, text in enumerate(generated):
                        f.write(f"--- Sequence {i+1} ---\n")
                        f.write(text + "\n\n")
                else:
                    f.write(generated)
            print(f"\nGenerated text saved to {args.output_file}")

    # Print memory usage
    if args.device == "cuda":
        print("\nGPU Memory Usage:")
        mem_info = get_gpu_memory_info()
        for key, value in mem_info.items():
            print(f"  {key.capitalize()}: {value:.2f} GB")


if __name__ == "__main__":
    main()
