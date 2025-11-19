"""
Training pipeline for Light LLM.

This module provides a comprehensive training loop with:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Logging and monitoring
"""

import os
import math
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """
    Trainer for Light LLM with optimizations for consumer hardware.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./checkpoints",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        save_total_limit: int = 3,
        use_tensorboard: bool = True,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for learning rate
            max_steps: Maximum number of training steps
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mixed_precision: Whether to use mixed precision training
            device: Device to train on
            output_dir: Directory to save checkpoints
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            save_total_limit: Maximum number of checkpoints to keep
            use_tensorboard: Whether to use TensorBoard logging
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        # Logging and checkpointing
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Initialize learning rate scheduler
        total_steps = max_steps if max_steps is not None else len(train_dataloader) * 100
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=min(warmup_steps / total_steps, 0.1),
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.mixed_precision and device == "cuda" else None

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = self.output_dir / "logs"
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)

    def train(self, num_epochs: Optional[int] = None):
        """
        Run training loop.

        Args:
            num_epochs: Number of epochs to train (if None, trains until max_steps)
        """
        self.model.train()

        if num_epochs is None:
            num_epochs = 1000  # Large number

        print(f"Starting training on {self.device}")
        print(f"Number of parameters: {self.model.get_num_params():,}")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_loss = self._train_epoch()

            print(f"Epoch {epoch + 1} completed - Average Loss: {epoch_loss:.4f}")

            # Check if we've reached max_steps
            if self.max_steps is not None and self.global_step >= self.max_steps:
                print(f"Reached maximum steps ({self.max_steps})")
                break

        # Save final checkpoint
        self.save_checkpoint(is_best=False)

        if self.writer is not None:
            self.writer.close()

        print("Training completed!")

    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'step': self.global_step})

            # Logging
            if self.global_step % self.logging_steps == 0:
                self._log_metrics({'train/loss': loss, 'train/lr': self.optimizer.param_groups[0]['lr']})

            # Evaluation
            if self.val_dataloader is not None and self.global_step % self.eval_steps == 0:
                val_loss = self.evaluate()
                self._log_metrics({'val/loss': val_loss})

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

                self.model.train()

            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(is_best=False)

            # Check max steps
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass with mixed precision
        if self.mixed_precision and self.scaler is not None:
            with autocast():
                _, loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

        else:
            # Standard training without mixed precision
            _, loss, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Update weights
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

        self.global_step += 1

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        print("\nRunning evaluation...")
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.mixed_precision:
                with autocast():
                    _, loss, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                _, loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.to_dict(),
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pt"
            print(f"Saving best model to {checkpoint_path}")
        else:
            checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")

        torch.save(checkpoint, checkpoint_path)

        # Clean up old checkpoints
        if not is_best:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(self.output_dir.glob("checkpoint-*.pt"))
        if len(checkpoints) > self.save_total_limit:
            for checkpoint in checkpoints[:-self.save_total_limit]:
                checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")

    def _log_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
        """
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.global_step)
