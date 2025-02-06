# Inverse Reinforcement Learning with Diffusion Models &nbsp;&nbsp;


Code for the thesis "Inverse Reinforcement Learning (IRL) with Diffusion Models". This repository contains the implementation of diffusion models for solving IRL tasks by learning reward functions from expert demonstrations.

## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

Further operations, which are system dependent, might be required in order to install mujoco.

Download the used dataset of expert trajectories from [here](https://drive.google.com/drive/folders/18tfzPTmFu_pH_UoISrH2IEMyrk5uVjQ9?usp=sharing)

## Usage

Train a diffusion model with:
```
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```

The default hyperparameters are listed in [`config/maze2d.py`](config/maze2d.py).
You can override any of them with runtime flags, eg `--batch_size 64`.

Plan using the diffusion model with:
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```

Learn the IRL Reward with BDGIRL method:
```
python scripts/train_inv_reward.py
```

Learn the IRL Reward with stop gradient method:
```
python scripts/train_inv_reward.py --no_grad_diff_steps 190
```

Learn the IRL Reward with fast sampling method:
```
python scripts/train_inv_reward.py --no_grad_diff_steps 190 --fast_sampling_batch_size 8 --inv_batch_size 2
```

## Reference
```
@thesis{fantoni2024irl_diffusion,
  title={Inverse Reinforcement Learning with Diffusion Models},
  author={Giovanni Fantoni},
  year={2024},
  school={University College London}
}
```


## Acknowledgements
The original codebase is from Janner et al. [Planning with Diffusion](https://github.com/jannerm/diffuser).
The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
