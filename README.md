# Iterative MCTS for NAS

This repository contains the release of PyTorch code to replicate main results on CIFAR-10, figures and tabels presented in the paper: "Iterative Monte Carlo Tree Search for Neural Architecture Search"

The repository structure is as follows:
  * `requirements.txt`, contains all required libraries
  * `figures/`, contains all figures represented in the paper


`Experiments/Pooling Experiments/` contains: 
  * contains code for CIFAR10 experiments in [Pooling benchmark](https://proceedings.mlr.press/v224/roshtkhari23a/roshtkhari23a.pdf)

## Getting Started
### Install
For our experimetns we used Python3.9 as it is compatible with FFCV library. To install requiremetns, please use one of the two ways:

   ```bash
   $ pip install -r requirements.txt
   ```
Or:
   ```bash
$ conda create -n ffcv2 python==3.9
$ conda activate ffcv2
$ pip install -r requirements.txt
   ```

### Dataset Preperation
For our experimetns, we converted data to [FFCV](https://ffcv.io/) format. For NAS methods presented in this paper, training datasets are split 50/50:

   * For cifar datasets experiments (assuming in respective dirctory such as `cifar/resnet18/`), to download and convert them for CIFAR100 run:

   ```bash
   $ ./write_cifar.sh
   ```
To split CIFAR datsets before conversion, run:

   ```bash
   $ ./write_cifar_50.sh
   ```

### Experiments
Use main.py to run experiments. See Readme file for more details.


  
