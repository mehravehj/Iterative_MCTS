# tree_utils.py
"""
Defines the Node class for the search tree and functions for building,
traversing (UCT+Boltzmann), and updating the tree (EMA rewards).
"""

import numpy as np
from scipy.stats import rankdata
import math
import random
from typing import Optional, List, Tuple, Any, TypeVar, Sequence

# === Node Class Definition ===
class Node:
    """Represents a node in the rank-based search tree."""
    def __init__(self, left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[float] = None, node_id: Any = None):
        self.left = left; self.right = right; self.value = value; self.node_id = node_id
        self.visit_count: int = 0; self.reward: Optional[float] = None # Stores EMA
    def is_leaf(self) -> bool: return self.node_id is not None
    def get_average_reward(self) -> Optional[float]: return self.reward # Returns EMA
    def __repr__(self, level: int = 0, prefix: str = "Root: ") -> str:
        indent = "  " * level
        ema_reward = self.get_average_reward()
        ema_reward_str = f"EMA_R={ema_reward:.4f}" if ema_reward is not None else "EMA_R=N/A"
        node_info = f"(V={self.visit_count}, {ema_reward_str})"
        if self.is_leaf():
            node_id_str = str(self.node_id); max_len=30
            if isinstance(self.node_id, tuple) and len(self.node_id) > max_len: node_id_str = str(self.node_id[:max_len//2] + ('...', ) + self.node_id[-max_len//2:])
            return f"{indent}{prefix}Leaf(ID={node_id_str}, Val={self.value}) {node_info}\n"
        else:
            s = f"{indent}{prefix}Internal {node_info}\n"
            if self.left: s += self.left.__repr__(level + 1, "L-- ")
            else: s += f"{indent}  L-- None\n"
            if self.right: s += self.right.__repr__(level + 1, "R-- ")
            else: s += f"{indent}  R-- None\n"
            return s

# === Tree Building Functions ===
def build_balanced_rank_tree(rank_sorted_items: List[Tuple[Any, float, float]]) -> Optional[Node]:
    """Recursively builds tree node by splitting rank-sorted items."""
    n = len(rank_sorted_items)
    if n == 1: item_id, item_value, item_rank = rank_sorted_items[0]; return Node(value=item_value, node_id=item_id)
    if n == 0: return None
    mid_index = n // 2; left_group = rank_sorted_items[:mid_index]; right_group = rank_sorted_items[mid_index:]
    node = Node(); node.left = build_balanced_rank_tree(left_group); node.right = build_balanced_rank_tree(right_group)
    return node

def create_rank_split_tree_with_tracking(data: List[Tuple[Any, float]]) -> Optional[Node]:
    """Builds the balanced tree from (id, value) data."""
    if not data: print("Tree build error: Input data empty."); return None
    if not isinstance(data, list) or not isinstance(data[0], (list, tuple)) or len(data[0]) != 2: print("Tree build error: Input data format invalid."); return None
    try: values = [float(item[1]) for item in data]
    except Exception as e: print(f"Tree build error: Invalid value: {e}"); return None
    try: ranks = rankdata(values, method='min')
    except Exception as e: print(f"Tree build error: Ranking failed: {e}"); return None
    items_with_rank = [(data[i][0], data[i][1], rank) for i, rank in enumerate(ranks)]
    rank_sorted_items = sorted(items_with_rank, key=lambda x: (x[2], x[1]))
    root = build_balanced_rank_tree(rank_sorted_items)
    return root

# === UCT+Boltzmann Traversal Function ===
def traverse_uct_boltzmann_and_get_path(start_node: Node, temperature: float, exploration_constant_C: float) -> Tuple[Optional[Node], List[Node]]:
    """ Traverses tree using UCT+Boltzmann, returns (leaf_node, path). """
    if not start_node: print("T Error: Start node is null."); return None, []
    if temperature <= 0: print("T Error: Temperature must be positive."); return None, []
    if exploration_constant_C < 0: print("T Error: C cannot be negative."); return None, []
    current_node = start_node; path = [current_node]
    while not current_node.is_leaf():
        parent_node = current_node; left_child = parent_node.left; right_child = parent_node.right; chosen_child = None
        if left_child and not right_child: chosen_child = left_child
        elif right_child and not left_child: chosen_child = right_child
        elif not left_child and not right_child: print("T Error: Internal node has no children."); return None, []
        else:
            N_parent = parent_node.visit_count; N_left = left_child.visit_count; N_right = right_child.visit_count
            if N_left == 0 and N_right == 0: chosen_child = np.random.choice([left_child, right_child])
            elif N_left == 0: chosen_child = left_child
            elif N_right == 0: chosen_child = right_child
            else:
                Q_left = left_child.get_average_reward(); Q_right = right_child.get_average_reward(); uct_left, uct_right = None, None
                if N_parent == 0: uct_left = Q_left if Q_left is not None else -np.inf; uct_right = Q_right if Q_right is not None else -np.inf
                elif Q_left is None or Q_right is None: chosen_child = np.random.choice([left_child, right_child]); print("T Warning: Visited child EMA_R=None. Random choice.")
                else:
                    log_N_parent = np.log(max(1, N_parent)); exp_term_left = exploration_constant_C * np.sqrt(log_N_parent / N_left); exp_term_right = exploration_constant_C * np.sqrt(log_N_parent / N_right)
                    uct_left = Q_left + exp_term_left; uct_right = Q_right + exp_term_right
                if chosen_child is None and uct_left is not None and uct_right is not None:
                    max_uct = max(uct_left, uct_right); exp_left = np.exp(np.clip((uct_left - max_uct) / temperature, -20, 20)); exp_right = np.exp(np.clip((uct_right - max_uct) / temperature, -20, 20))
                    sum_exp = exp_left + exp_right; prob_left = 0.5 if sum_exp == 0 else exp_left / sum_exp; prob_right = 1.0 - prob_left
                    chosen_child = np.random.choice([left_child, right_child], p=[prob_left, prob_right])
                elif chosen_child is None: chosen_child = np.random.choice([left_child, right_child]); print("T Warning: Fallback random choice in UCT.")
        if chosen_child: current_node = chosen_child; path.append(current_node)
        else: print("T Error: Could not determine next node."); return None, []
    return current_node, path

# === EMA Reward Update Function ===
def update_tree_rewards_ema(path: List[Node], validation_accuracy: float, decay: float):
    """Updates visit count and EMA reward for each node in the path."""
    if not path: return
    if not (0.0 <= decay < 1.0): print(f"W: Invalid EMA decay {decay}. Clamping."); decay = max(0.0, min(0.99, decay))
    current_acc = float(validation_accuracy)
    for node in path:
        node.visit_count += 1
        if node.reward is None: node.reward = current_acc
        else: node.reward = decay * node.reward + (1.0 - decay) * current_acc