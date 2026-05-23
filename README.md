# EDAIL: Adversarial Imitation Learning via Exploration-Driven Data Augmentation



The Official PyTorch implementation of [**EDAIL: Adversarial Imitation Learning via Exploration-Driven Data Augmentation**].

## Installation

### Preliminaries

- [pytorch](https://pytorch.org/get-started/locally/)
- [mujoco-py](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)

### Environment Setup

1. This code base requires `Python 3.8` or higher. All package requirements are in
   `requirements.txt`. To install from scratch using Anaconda, use the following
   commands.
   ```   
   conda create -n [your_env_name] python=3.8
   conda activate [your_env_name]
   ./utils/setup.sh
   ```

2. Setup [Weights and Biases](https://wandb.ai/site) by first logging in with `wandb login <YOUR_API_KEY>` and then editing `config.yaml` with your W&B username and project name.

### Expert Demonstration Setup

1. Download expert demonstration datasets to `./expert_datasets`. Most of the expert demonstration datasets we used are provided by [goal_prox_il](https://github.com/clvrai/goal_prox_il). We provide a script for downloading and post-processing the expert datasets.

   ```
   ./utils/expert_data.sh
   ```

## How to reproduce experiments

To replicate the experiments conducted in our paper, follow these steps:

1. **Select Configuration Files:** The wandb sweep configuration files for all tasks can be found in the `configs` directory:
   - `edail.yaml`
   - `gail.yaml`
   - `wail.yaml`
   - `bc.yaml`
   - `diffusion-policy.yaml`
2. **Run Experiments:** After selecting the desired configuration file, execute the following command:
   ```
   python run_experiments.py  <Path_to_Configuration_File.yaml>
   ```

## Acknowledgement

### Code

- The base code was adapted from [drail](https://github.com/NVlabs/DRAIL).
- The Grid world environment was obtained from [maximecb](https://github.com/maximecb/gym-minigrid)
- The Fetch and Hand Rotate environments were customized based on [OpenAI](https://github.com/openai/robogym) implementations.
- The Ant environment was customized by [goal_prox_il](https://github.com/clvrai/goal_prox_il) and originated from [Farama-Foundation](https://github.com/Farama-Foundation/Gymnasium).
- The SAC code was obtained from [denisyarats](https://github.com/denisyarats/pytorch_sac)
- The Maze2D environment is based on [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/rail-berkeley/d4rl).

### Expert Dataset

- The expert demonstrations of the Maze, FetchPick, FetchPush, HandRotate, and AntReach tasks were obtained from [goal_prox_il](https://github.com/clvrai/goal_prox_il).
