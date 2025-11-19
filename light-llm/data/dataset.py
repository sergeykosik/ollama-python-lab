"""
Dataset classes for training Light LLM on code.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def collect_code_files(
    data_dir: str,
    output_file: str,
    file_extensions: List[str] = [".py", ".js", ".java", ".cpp"],
    max_files: Optional[int] = None,
) -> int:
    """
    Collect code files into a single text file.

    Args:
        data_dir: Directory containing code files
        output_file: Output file path
        file_extensions: List of file extensions to include
        max_files: Maximum number of files to collect (None for all)

    Returns:
        Number of files collected
    """
    data_path = Path(data_dir)
    files = []

    # Collect all matching files
    for ext in file_extensions:
        files.extend(data_path.rglob(f"*{ext}"))

    # Limit files if specified
    if max_files is not None and len(files) > max_files:
        files = random.sample(files, max_files)

    # Write to output file
    num_files = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as code_file:
                    content = code_file.read()
                    # Add file separator
                    f.write(f"# File: {file_path.name}\n")
                    f.write(content)
                    f.write("\n\n")
                    num_files += 1
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return num_files


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for training.

    This dataset tokenizes text examples and creates input_ids, attention_mask, and labels.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text examples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)

        # Pad to max_length
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            tokens = tokens + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # For language modeling, labels are the same as input_ids
        # We'll mask out padding tokens in the loss calculation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class CodeDataset(Dataset):
    """
    Dataset for loading code files for training.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        file_extensions: List[str] = [".py", ".js", ".java", ".cpp"],
        max_files: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing code files
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            file_extensions: List of file extensions to include
            max_files: Maximum number of files to load (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load code files
        data_path = Path(data_dir)
        files = []

        for ext in file_extensions:
            files.extend(data_path.rglob(f"*{ext}"))

        # Limit files if specified
        if max_files is not None and len(files) > max_files:
            files = random.sample(files, max_files)

        # Read files
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        self.examples.append(content)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

        print(f"Loaded {len(self.examples)} code files")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(tokens)

        # Pad to max_length
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            tokens = tokens + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    file_extensions: List[str] = [".py", ".js", ".java", ".cpp"],
    max_files: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing code files
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        train_ratio: Ratio of data for training
        file_extensions: List of file extensions to include
        max_files: Maximum number of files to load (None for all)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = CodeDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        file_extensions=file_extensions,
        max_files=max_files,
    )

    # Split into train and validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
