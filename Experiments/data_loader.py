# data_loader.py
"""
Provides dummy data loaders
"""

import torch
import torch.utils.data
import math
from typing import List, Tuple

def get_train_queue(batch_size: int = 128) -> torch.utils.data.DataLoader:
    """Creates a dummy DataLoader for training."""
    print(f"      [Dummy] Creating dummy train data loader (Batch size: {batch_size})...")
    num_samples = 512 # Simulate smaller dataset for faster dummy runs
    # Generate tensors on CPU first, move to device in training loop
    dummy_train_images = torch.rand(num_samples, 3, 32, 32, dtype=torch.float32)
    dummy_train_labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(dummy_train_images, dummy_train_labels)
    # Use standard DataLoader for batching and shuffling
    # Add num_workers=... and pin_memory=... for speed with real data/GPU
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_validation_queue(batch_size: int = 100) -> torch.utils.data.DataLoader:
    """Creates a dummy DataLoader for validation."""
    print(f"      [Dummy] Creating dummy validation data loader (Batch size: {batch_size})...")
    num_samples = 256 # Simulate smaller dataset
    dummy_valid_images = torch.rand(num_samples, 3, 32, 32, dtype=torch.float32)
    dummy_valid_labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(dummy_valid_images, dummy_valid_labels)
    # No shuffling for validation
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)