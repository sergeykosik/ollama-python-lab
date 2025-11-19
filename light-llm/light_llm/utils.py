"""
Utility functions for Light LLM.

This module provides helper functions for:
- Model parameter counting
- Memory profiling
- Checkpoint management
- Logging
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get the size of a model in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.

    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }


def print_model_info(model: nn.Module):
    """
    Print detailed model information.

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size_mb(model)

    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")

    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_info['allocated']:.2f} GB")
        print(f"  Reserved: {mem_info['reserved']:.2f} GB")
        print(f"  Free: {mem_info['free']:.2f} GB")
        print(f"  Total: {mem_info['total']:.2f} GB")

    print("=" * 60)


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        path: Path to save file
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def setup_logging(log_dir: str, name: str = "training") -> Path:
    """
    Set up logging directory.

    Args:
        log_dir: Directory for logs
        name: Name of the log subdirectory

    Returns:
        Path to log directory
    """
    log_path = Path(log_dir) / name
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob("checkpoint-*.pt"))
    if not checkpoints:
        return None

    # Sort by step number and return latest
    checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
    return str(checkpoints[-1])


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float,
) -> Dict[str, float]:
    """
    Estimate training time.

    Args:
        num_examples: Number of training examples
        batch_size: Batch size
        num_epochs: Number of epochs
        seconds_per_batch: Average seconds per batch

    Returns:
        Dictionary with time estimates
    """
    batches_per_epoch = num_examples // batch_size
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch

    return {
        "batches_per_epoch": batches_per_epoch,
        "total_batches": total_batches,
        "seconds": total_seconds,
        "minutes": total_seconds / 60,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400,
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cleanup_memory():
    """Clean up GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gradient_checkpointing_enable(model: nn.Module):
    """
    Enable gradient checkpointing for a model.

    Args:
        model: PyTorch model
    """
    if hasattr(model, 'gradient_checkpointing'):
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Model does not support gradient checkpointing")


def gradient_checkpointing_disable(model: nn.Module):
    """
    Disable gradient checkpointing for a model.

    Args:
        model: PyTorch model
    """
    if hasattr(model, 'gradient_checkpointing'):
        model.gradient_checkpointing = False
        print("Gradient checkpointing disabled")


def print_gpu_utilization():
    """Print current GPU utilization."""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f} GB")
    else:
        print("CUDA not available")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        """
        Initialize meter.

        Args:
            name: Name of the metric
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def save_generation_samples(
    model,
    tokenizer,
    prompts: list,
    output_file: str,
    **generation_kwargs,
):
    """
    Generate samples and save to file.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompts: List of prompts
        output_file: Output file path
        **generation_kwargs: Additional generation arguments
    """
    from .inference import TextGenerator

    generator = TextGenerator(model, tokenizer)
    results = []

    print(f"Generating samples for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/{len(prompts)}...")
        generated = generator.generate(prompt, **generation_kwargs)
        results.append({
            "prompt": prompt,
            "generated": generated,
        })

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated samples saved to {output_file}")


def load_pretrained_embeddings(
    model: nn.Module,
    embedding_path: str,
    embedding_dim: int,
):
    """
    Load pretrained embeddings into model.

    Args:
        model: Model with embedding layer
        embedding_path: Path to embeddings file
        embedding_dim: Dimension of embeddings
    """
    print(f"Loading pretrained embeddings from {embedding_path}")
    # Implementation depends on embedding format
    # This is a placeholder for custom embedding loading
    pass
