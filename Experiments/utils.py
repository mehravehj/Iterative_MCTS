# utils.py
"""
Contains general utility functions like optimizer creation and accuracy calculation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Tuple

def create_optimizers(net: nn.Module, lr: float, m: float, wd: float) -> Tuple[optim.Optimizer, None]:
    """Creates SGD optimizer. Placeholder for scheduler."""
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m, weight_decay=wd)
    # Scheduler implementation can be added here if needed
    scheduler = None
    # print("      [INFO] LR scheduler not implemented/returned.") # Verbose option
    return optimizer, scheduler

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Calculates batch accuracy.

    Args:
        outputs: Raw logits output from the model (Batch, NumClasses).
        targets: Ground truth labels (Batch,).

    Returns:
        Tuple containing:
            - Correct prediction count (tensor scalar).
            - Total number of samples in batch (int).
    """
    with torch.no_grad(): # Ensure no gradients calculated here
        batch_total = targets.size(0)
        if batch_total == 0:
            # Return tensor on same device as outputs if possible, else CPU
            return torch.tensor(0, device=outputs.device if outputs is not None else 'cpu'), 0
        _, predicted = torch.max(outputs.data, 1) # Get predicted class index
        correct_count = (predicted == targets).sum() # Sum boolean tensor for count
    # Return the count as a tensor and total as int
    return correct_count, batch_total