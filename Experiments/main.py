# main.py

# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
import argparse # Import argparse
from typing import List, Tuple, Dict, Optional, Any # Keep necessary typing

# --- Import from project files ---
from model_def import create_model
from search_space import create_search_space, Architecture # Import type alias too
from tree_utils import create_rank_split_tree_with_tracking, Node # Import Node if needed for type hints or analysis
from training_phases import uniform_warmup_training, evaluate_all_architectures, run_phase4_tree_based_training
from utils import create_optimizers # Optimizer/Accuracy utils are here now
from data_loader import get_train_queue, get_validation_queue # Data functions

# === Main Execution ===
if __name__ == "__main__":

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Multi-Phase NAS with Tree Search")

    # Phase Lengths
    parser.add_argument('--n_warmup', type=int, default=1, help='Epochs for Phase 1 Uniform Warmup (n)')
    parser.add_argument('--m_batches', type=int, default=5, help='Num validation batches per arch in Phase 2 (m)')
    parser.add_argument('--k_epochs', type=int, default=2, help='Epochs for Phase 4 Tree-Based Training (k)')
    parser.add_argument('--h_iterations', type=int, default=2, help='Number of outer iterations (Phase 2-3-4 cycle) (h)')

    # Tree/Sampling Params
    parser.add_argument('--rank_method', type=str, default='neg_acc', choices=['neg_acc', 'zero', 'sum', 'random'], help='Method to get rank value for tree building (neg_acc uses Phase 2 results)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Initial Boltzmann temperature for tree traversal (T)')
    parser.add_argument('--exploration_c', type=float, default=1.0, help='UCT exploration constant (C)')
    parser.add_argument('--ema_decay', type=float, default=0.9, help='Decay factor for EMA reward update (alpha)')
    parser.add_argument('--anneal_temp', action='store_true', help='Enable temperature annealing in Phase 4')

    # Optimizer Params
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for SGD')

    # Data/Setup Params
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (None for no seed)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use (cuda, cpu, auto)')

    # Search Space / Model Structure
    parser.add_argument('--num_layers', type=int, default=5, help='Number of stages/layers (defines path length)')
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales (defines num_pooling = scales-1)')
    parser.add_argument('--channels', type=str, default='16,32,64,64,128', help='Comma-separated list of channels per stage (length must match num_layers)')

    # Use parse_known_args() for Colab/Jupyter compatibility
    args, unknown = parser.parse_known_args()
    if unknown: print(f"Ignoring unrecognized arguments: {unknown}")

    # --- Process Parsed Arguments ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        channels_list = [int(c.strip()) for c in args.channels.split(',')]
        print(f"Parsed Channels: {channels_list}")
    except ValueError:
        raise ValueError("Invalid format for --channels. Use comma-separated integers (e.g., '16,32,64').")

    if len(channels_list) != args.num_layers:
         raise ValueError(f"--channels length ({len(channels_list)}) must match --num_layers ({args.num_layers})")

    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # --- Setup ---
    start_time = datetime.datetime.now()
    print("="*20 + f" NAS Start Time: {start_time} " + "="*20)
    print("\n" + "="*10 + " Initial Setup " + "="*10)

    all_architectures, num_archs = create_search_space(args.num_layers, args.num_scales)
    if num_archs == 0: raise ValueError("Search space generation failed or resulted in zero architectures!")

    # !! Using Dummy implementations defined above !!
    model = create_model(args.num_layers, channels_list) # Use parsed channels_list
    model = model.to(device)
    # Use parsed hyperparams for optimizer
    optimizer, scheduler = create_optimizers(model, args.lr, args.momentum, args.weight_decay) # Scheduler currently None
    # Use parsed batch_size for data loaders
    train_loader = get_train_queue(batch_size=args.batch_size)
    validation_loader = get_validation_queue(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Setup Complete.")
    print("-"*40)

    # --- Run Phase 1: Uniform Warmup ---
    uniform_warmup_training(model, optimizer, criterion, train_loader, all_architectures, args.n_warmup, device) # Use args.n_warmup
    print("-"*40)

    # --- Run Outer Loop (Phase 5) ---
    tree_root: Optional[Node] = None # Type hint for clarity
    current_temp = args.temperature # Use args.temperature

    for h_iter in range(args.h_iterations): # Use args.h_iterations
        print(f"\n{'='*20} Outer Iteration {h_iter + 1}/{args.h_iterations} {'='*20}")

        # --- Run Phase 2: Full Validation ---
        current_accuracy_map = evaluate_all_architectures(model, all_architectures, validation_loader, args.m_batches, device) # Use args.m_batches

        # --- Run Phase 3: Rebuild Tree ---
        print("\n--- Starting Phase 3: Rebuilding Tree ---")
        tree_input_data = []
        valid_archs_in_map = [arch for arch in all_architectures if arch in current_accuracy_map]
        if not valid_archs_in_map: print("Warning: Accuracy map empty or contains no valid architectures after Phase 2.")

        # Determine ranking value based on method and availability of accuracy map
        rank_method_this_iter = args.rank_method # Use arg
        # Force using accuracy after first iteration if method was neg_acc
        if h_iter > 0 and args.rank_method == 'neg_acc':
            if not valid_archs_in_map:
                 print("  Warning: Reverting to zero ranking for rebuild as accuracy map was empty/invalid.")
                 rank_method_this_iter = 'zero'
            else:
                 rank_method_this_iter = 'neg_acc'
        elif h_iter == 0 and args.rank_method == 'neg_acc':
             # Use zero for very first build if accuracy isn't ready
             rank_method_this_iter = 'zero'
             print("  Building initial tree: zero (neg_acc selected but no accuracy map yet)")

        # Prepare data based on chosen method for this iteration
        if rank_method_this_iter == 'neg_acc' and valid_archs_in_map:
            tree_input_data = [(arch, -current_accuracy_map[arch]) for arch in valid_archs_in_map]
            print(f"  Building tree based on negative accuracy (Items: {len(tree_input_data)})")
        elif rank_method_this_iter == 'sum':
            tree_input_data = [(path, float(sum(path))) for path in all_architectures]; print("  Building tree: sum")
        elif rank_method_this_iter == 'random':
            tree_input_data = [(path, np.random.rand()) for path in all_architectures]; print("  Building tree: random")
        else: # Default 'zero'
            tree_input_data = [(path, 0.0) for path in all_architectures]; print("  Building tree: zero")

        if not tree_input_data: print("Error: No data to build tree. Stopping."); break
        tree_root = create_rank_split_tree_with_tracking(tree_input_data) # Build/Rebuild
        if not tree_root: print("Error: Failed to rebuild tree. Stopping."); break
        print("--- Finished Phase 3 (Tree Rebuilt) ---")
        # Optional: Print the newly built tree
        # print("\n--- Tree Structure after Phase 3 Rebuild ---")
        # print(tree_root)

        # --- Run Phase 4: Tree-Based Training ---
        tree_root, final_temp = run_phase4_tree_based_training(
            tree_root=tree_root, model=model, optimizer=optimizer, criterion=criterion,
            train_loader=train_loader, validation_loader=validation_loader,
            k_epochs=args.k_epochs, # Use args.k_epochs
            temperature_start=current_temp, # Use current temp
            exploration_c=args.exploration_c, # Use args.exploration_c
            device=device,
            ema_decay=args.ema_decay, # Use args.ema_decay
            anneal_temp=args.anneal_temp # Use args.anneal_temp
        )
        current_temp = final_temp # Update temperature if annealing was used
        print("-"*40)

    print(f"\n{'='*20} NAS Process Finished ({args.h_iterations} Iterations) {'='*20}")
    end_time = datetime.datetime.now()
    print(f"End time: {end_time}")
    print(f"Total duration: {end_time - start_time}")

    # --- Final Analysis ---
    if tree_root:
        print("\n--- Final Tree State (Summary) ---")
        all_nodes = []; queue = [tree_root]
        while queue: node = queue.pop(0); all_nodes.append(node);
        if node.left: queue.append(node.left);
        if node.right: queue.append(node.right);
        leaf_stats = [{'id': n.node_id, 'visits': n.visit_count, 'ema_reward': n.get_average_reward() if n.get_average_reward() is not None else -1} for n in all_nodes if n.is_leaf()]
        sorted_leaves = sorted(leaf_stats, key=lambda x: x['ema_reward'], reverse=True)
        print("\n--- Final Analysis: Top 5 Leaves by EMA Reward ---")
        for i, stats in enumerate(sorted_leaves[:5]):
             avg_r_str = f"{stats['ema_reward']:.4f}" if stats['ema_reward'] != -1 else "N/A"
             id_str = str(stats['id']); max_len=40
             if len(id_str) > max_len: id_str = id_str[:max_len//2] + "..." + id_str[-max_len//2:]
             print(f"  {i+1}. ID: {id_str}, EMA_R: {avg_r_str}, Visits: {stats['visits']}")
    else:
        print("\nNAS loop did not produce a final tree.")