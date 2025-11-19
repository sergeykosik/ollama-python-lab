"""
Training script for Light LLM.

This script provides a complete training pipeline with:
- Model initialization
- Data loading
- Training with mixed precision
- Checkpointing
- Logging

Usage:
    python train.py --data_dir ./data/code --model_size small --batch_size 4
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from light_llm.config import get_config, LightLLMConfig
from light_llm.model import LightLLM
from light_llm.tokenizer import CodeTokenizer
from light_llm.trainer import Trainer
from light_llm.utils import (
    set_seed,
    print_model_info,
    get_gpu_memory_info,
    estimate_training_time,
)
from data.dataset import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Light LLM")

    # Model arguments
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size preset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing code files",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to trained tokenizer (if None, will train new one)",
    )
    parser.add_argument(
        "--file_extensions",
        type=str,
        nargs="+",
        default=[".py", ".js", ".java", ".cpp"],
        help="File extensions to include",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Optimization arguments
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def train_tokenizer(data_dir: str, vocab_size: int, file_extensions: list) -> CodeTokenizer:
    """
    Train a new tokenizer on the data.

    Args:
        data_dir: Directory containing code files
        vocab_size: Vocabulary size
        file_extensions: File extensions to include

    Returns:
        Trained tokenizer
    """
    from data.dataset import collect_code_files
    import tempfile

    print("Training new tokenizer...")

    # Collect code files into a single file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        corpus_file = f.name

    collect_code_files(data_dir, corpus_file, file_extensions=file_extensions)

    # Train tokenizer
    tokenizer = CodeTokenizer(vocab_size=vocab_size)
    tokenizer.train([corpus_file], vocab_size=vocab_size)

    # Save tokenizer
    tokenizer_dir = Path("./tokenizer")
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_dir))

    # Clean up
    os.remove(corpus_file)

    print(f"Tokenizer trained and saved to {tokenizer_dir}")
    return tokenizer


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    print("=" * 80)
    print("Light LLM Training")
    print("=" * 80)

    # Load or train tokenizer
    if args.tokenizer_path is not None:
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = CodeTokenizer.load(args.tokenizer_path)
    else:
        tokenizer = train_tokenizer(args.data_dir, args.vocab_size, args.file_extensions)

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create model configuration
    config = get_config(args.model_size)
    config.vocab_size = len(tokenizer)
    config.max_position_embeddings = args.max_length
    config.use_gradient_checkpointing = args.gradient_checkpointing

    # Create model
    print("\nInitializing model...")
    model = LightLLM(config)

    # Print model info
    print_model_info(model)

    # Create dataloaders
    print("\nLoading data...")
    train_dataloader, val_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        file_extensions=args.file_extensions,
    )

    print(f"Training examples: {len(train_dataloader.dataset)}")
    print(f"Validation examples: {len(val_dataloader.dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")

    # Estimate training time
    print("\nEstimating training time (assuming ~0.5s per batch)...")
    time_estimate = estimate_training_time(
        num_examples=len(train_dataloader.dataset),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seconds_per_batch=0.5,
    )
    print(f"Estimated time: {time_estimate['hours']:.2f} hours ({time_estimate['days']:.2f} days)")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        device=args.device,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    try:
        trainer.train(num_epochs=args.num_epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(is_best=False)
        print("Checkpoint saved")

    print("\nTraining completed!")

    # Print final memory usage
    if args.device == "cuda":
        print("\nFinal GPU Memory Usage:")
        mem_info = get_gpu_memory_info()
        for key, value in mem_info.items():
            print(f"  {key.capitalize()}: {value:.2f} GB")


if __name__ == "__main__":
    main()
