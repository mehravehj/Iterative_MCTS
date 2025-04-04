# training_phases.py
"""
Defines functions for the different phases of the NAS process:
- Phase 1: Uniform Warmup Training
- Phase 2: Full Architecture Evaluation
- Phase 4: Tree-Based Training (Outer loop and Inner step)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from typing import Sequence, Dict, Optional, List, Tuple

# --- Import from other modules ---
from search_space import sample_architecture_uniformly, Architecture # Import type alias
from tree_utils import Node, traverse_uct_boltzmann_and_get_path, update_tree_rewards_ema
from utils import calculate_accuracy

# === Phase 1 Function ===
def uniform_warmup_training(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            all_architectures: Sequence[Architecture],
                            n_epochs: int, device: torch.device):
    """Phase 1: Train model with uniformly sampled architectures."""
    if n_epochs <= 0: print("\n--- Skipping Phase 1: Uniform Warmup (0 Epochs) ---"); return
    print(f"\n--- Starting Phase 1: Uniform Warmup ({n_epochs} Epochs) ---")
    model.train().to(device); criterion.to(device)
    total_steps = 0
    for epoch in range(n_epochs):
        print(f"  Warmup Epoch {epoch+1}/{n_epochs}")
        for batch_idx, batch_data in enumerate(train_loader):
            arch_config = sample_architecture_uniformly(all_architectures)
            if not arch_config: print("    Skipping step: Failed to sample arch."); continue
            try:
                model.set_path(arch_config)
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets); loss.backward(); optimizer.step()
                total_steps += 1
                if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader)-1:
                     print(f"    Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            except Exception as e: print(f"    Error during Warmup step (Arch: {arch_config}): {e}")
    print(f"--- Finished Phase 1 ({total_steps} steps) ---")

# === Phase 2 Function ===
def evaluate_all_architectures(model: nn.Module, all_architectures: Sequence[Architecture],
                               validation_loader: torch.utils.data.DataLoader,
                               m_batches: int, device: torch.device) -> Dict[Architecture, float]:
    """Phase 2: Evaluate all architectures for m batches."""
    if m_batches <= 0: print("\n--- Skipping Phase 2: Full Validation (0 batches) ---"); return {arch: 0.0 for arch in all_architectures}
    print(f"\n--- Starting Phase 2: Full Validation ({m_batches} batches/arch) ---")
    accuracy_map = {}
    model.eval().to(device)
    num_archs_total = len(all_architectures)
    if num_archs_total == 0: return {} # Handle empty search space

    with torch.no_grad():
        for arch_idx, arch_config in enumerate(all_architectures):
            try: model.set_path(arch_config)
            except Exception as e: print(f"  Eval Error: setting path {arch_config}: {e}. Skip."); accuracy_map[arch_config] = 0.0; continue
            total_correct = 0.0; total_samples = 0.0; batches_done = 0; validation_iterator = iter(validation_loader)
            for _ in range(m_batches):
                 try:
                     inputs, targets = next(validation_iterator); inputs, targets = inputs.to(device), targets.to(device); outputs = model(inputs)
                     correct, total = calculate_accuracy(outputs, targets); total_correct += correct.item(); total_samples += total; batches_done += 1
                 except StopIteration: break
                 except Exception as e: print(f"    Eval Error: valid batch arch {arch_config}: {e}")
            avg_acc = (total_correct / total_samples) if total_samples > 0 else 0.0; accuracy_map[arch_config] = avg_acc
            if (arch_idx + 1) % max(1, num_archs_total // 5) == 0: print(f"  Evaluated {arch_idx+1}/{num_archs_total} Archs...") # Progress more often
    print(f"--- Finished Phase 2 (Evaluated {len(accuracy_map)} architectures) ---")
    return accuracy_map

# === Phase 4 Inner Function ===
def tree_based_single_step(tree_root: Node, model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                           train_batch: tuple, valid_batch: tuple,
                           temperature: float, exploration_constant_C: float,
                           device: torch.device
                           ) -> Tuple[bool, Optional[List[Node]], Optional[float], Optional[float], Optional[Architecture]]:
    """ Performs one step of tree-based phase: sample, train, validate. """
    # 1. Sample Architecture Path
    leaf_node, path_nodes = traverse_uct_boltzmann_and_get_path(tree_root, temperature, exploration_constant_C )
    if not leaf_node or not path_nodes: return False, None, None, None, None # Traversal failed
    sampled_arch_id = leaf_node.node_id
    # 2. Configure Model
    try: model.set_path(sampled_arch_id)
    except Exception as e: print(f"    Step Error: setting path {sampled_arch_id}: {e}"); return False, path_nodes, None, None, sampled_arch_id
    # 3. Train Step
    model.train(); train_loss_item = None
    try:
        train_inputs, train_targets = train_batch; train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        optimizer.zero_grad(); train_outputs = model(train_inputs); train_loss = criterion(train_outputs, train_targets); train_loss.backward(); optimizer.step(); train_loss_item = train_loss.item()
    except Exception as e: print(f"    Step Error: during training: {e}"); return False, path_nodes, None, train_loss_item, sampled_arch_id
    # 4. Validation Step
    model.eval(); validation_accuracy_signal = None
    try:
        validation_inputs, validation_targets = valid_batch; validation_inputs, validation_targets = validation_inputs.to(device), validation_targets.to(device)
        with torch.no_grad():
            validation_outputs = model(validation_inputs); batch_correct, batch_total = calculate_accuracy(validation_outputs, validation_targets)
            validation_accuracy_signal = (float(batch_correct.item()) / float(batch_total)) if batch_total > 0 else 0.0
    except Exception as e: print(f"    Step Warning: during validation: {e}"); return True, path_nodes, None, train_loss_item, sampled_arch_id # Train OK, Valid Failed
    return True, path_nodes, validation_accuracy_signal, train_loss_item, sampled_arch_id

# === Phase 4 Outer Loop Function ===
def run_phase4_tree_based_training(tree_root: Node, model: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                                   train_loader: torch.utils.data.DataLoader,
                                   validation_loader: torch.utils.data.DataLoader,
                                   k_epochs: int, temperature_start: float,
                                   exploration_c: float, device: torch.device,
                                   ema_decay: float, # Added EMA decay param
                                   anneal_temp: bool = False
                                   ) -> Tuple[Node, float]: # Return updated tree and temp
    """ Runs Phase 4: K epochs of tree-based sampling, training, and EMA reward updating. """
    if k_epochs <= 0: print("\n--- Skipping Phase 4: Tree-Based Training (0 Epochs) ---"); return tree_root, temperature_start
    print(f"\n--- Running Phase 4: Tree-Based Training ({k_epochs} Epochs) ---"); temperature = temperature_start; model.to(device); criterion.to(device)
    try: steps_per_epoch = len(train_loader)
    except TypeError: steps_per_epoch = -1
    total_steps = k_epochs * steps_per_epoch if steps_per_epoch > 0 else k_epochs * 50 # Guess steps if unknown len
    if steps_per_epoch > 0: print(f"(Approx. {steps_per_epoch} steps/epoch, Total: {total_steps} steps)")
    else: print(f"(Running for {k_epochs} epochs, loader length unknown)")

    global_step = 0
    for epoch in range(k_epochs):
        print(f"  Epoch {epoch + 1}/{k_epochs}"); train_iter = iter(train_loader); valid_iter = iter(validation_loader); epoch_steps = 0
        for train_batch in train_iter: # Iterate through training data
            try: valid_batch = next(valid_iter)
            except StopIteration: valid_iter = iter(validation_loader); # Reset validation iterator
            try: valid_batch = next(valid_iter)
            except StopIteration: print("    Error: Validation loader empty after reset! Stopping epoch."); break # Stop epoch

            # Perform one step
            success, path_nodes, reward_signal, train_loss, sampled_arch_id = \
                tree_based_single_step(tree_root, model, optimizer, criterion,
                                       train_batch, valid_batch,
                                       temperature, exploration_c, device)

            # Update tree rewards based on step outcome
            if path_nodes is not None and reward_signal is not None:
                 update_tree_rewards_ema(path_nodes, reward_signal, ema_decay) # Use EMA update
                 # Log periodically
                 if (global_step + 1) % 100 == 0:
                     id_str = str(sampled_arch_id); max_len=20
                     if len(id_str) > max_len: id_str = id_str[:max_len//2] + "..." + id_str[-max_len//2:]
                     print(f"    Step {global_step+1}: Sampled {id_str}, R: {reward_signal:.4f}, L: {train_loss:.4f if train_loss else 'N/A'}")
            elif success: # Step succeeded but validation failed
                 if (global_step + 1) % 100 == 0: print(f"    Step {global_step+1}: Sampled {str(sampled_arch_id)[:20]}... OK, but validation failed. No tree update.")
            else: # Step failed earlier
                 if (global_step + 1) % 100 == 0: print(f"    Step {global_step+1}: Step failed for {str(sampled_arch_id)[:20]}... No tree update.")

            global_step += 1; epoch_steps += 1
            # Optional: Temperature annealing
            if anneal_temp: temperature = max(0.05, temperature * 0.9998)
            # Optional: LR scheduler step (step-based)

        print(f"  Epoch {epoch + 1} completed ({epoch_steps} steps). Current Temp: {temperature:.3f}")
        # Optional: LR scheduler step (epoch-based)

    print(f"--- Finished Phase 4 ({k_epochs} epochs, {global_step} total steps) ---")
    return tree_root, temperature