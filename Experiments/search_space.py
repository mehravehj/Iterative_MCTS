# search_space.py
"""
Defines the search space (pooling configurations) and uniform sampling.
"""

import itertools
import random
from typing import Sequence, Optional, Tuple, TypeVar

# Type alias for architecture representation (e.g., pooling path tuples)
Architecture = Tuple
ArchitectureType = TypeVar('ArchitectureType')

def create_search_space(num_layers: int, num_scales: int) -> Tuple[Tuple[Architecture, ...], int]:
    """
    Generates all possible pooling configurations (architectures) based on the
    number of layers and desired downsampling scales.
    """
    num_pooling = num_scales - 1
    num_available_layers = num_layers - 1
    paths =[]
    if num_pooling < 0 or num_pooling > num_available_layers:
        print(f"Warning: Cannot choose {num_pooling} pooling layers from {num_available_layers} slots.")
        return tuple(), 0

    # print(f"Generating combinations for {num_available_layers} choose {num_pooling}...")
    for positions in itertools.combinations(range(num_available_layers), num_pooling):
        p = [0] * num_available_layers
        for i in positions: p[i] = 1
        paths.append(tuple([0] + p)) # Prepend 0

    paths = tuple(paths)
    number_paths = len(paths)
    print(f'Generated {number_paths} architecture paths.')
    return paths, number_paths

def sample_architecture_uniformly(
    all_architectures: Sequence[ArchitectureType]
) -> Optional[ArchitectureType]:
    """Samples one architecture uniformly at random from a sequence."""
    if not all_architectures:
        print("Warning: Cannot sample from empty architecture list.")
        return None
    return random.choice(all_architectures)