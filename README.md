# Iterative MCTS for NAS

This repository contains the release of PyTorch code to replicate main results on CIFAR-10, figures and tabels presented in the paper: "Iterative Monte Carlo Tree Search for Neural Architecture Search"

The repository structure is as follows:
  * `requirements.txt`, contains all required libraries
  * `figures/`, contains all figures represented in the paper


`Experiments/Pooling Experiments/` contains: 
  * contains code for Tory examples experiments in [Pooling benchmark](https://proceedings.mlr.press/v224/roshtkhari23a/roshtkhari23a.pdf)

## Getting Started
### Install
To run the code Python 3.8+ is needed. To install requiremetns, please run:

   ```bash
   $ pip install -r requirements.txt
   ```

Install PyTorch: Before installing the requirements, install the correct PyTorch version for your system (specific CUDA version).

### Dataset Preperation
For the code, we used a toy exaple for dataset To run for CIFAR10, first the training dataset needs to be split 50/50 to provide training/validation (replacing data_loader with data_loader_CIFAR) 

### Experiments
Use main.py to run experiments.
The arg are:

'--n_warmup': number itrations for uniform sampling to build initial tree $\mathcal{T}_{init}$

'--k_epochs' : $K$ in Alg. 1, iteration of each MCTS

'--h_iterations' : $M$ in Alg. 1, number of MCTS iterations

'--temperature' : $T$ Boltzmann temperature in eq. 2

'--exploration_c' : $C$ UCTS exploration constant in eq. 1 

'--ema_decay' : $\beta$ weighting factor in eq. 3

'--m_batches' : number of btaches for validation (B in table 3 right)


  
