import json
import os

import torch
import numpy as np
from gymnasium.spaces import Box

from imitation.data.types import Trajectory
from imitation.algorithms.bc import BC

import diffuser.utils as utils
import diffuser.datasets as datasets

from diffuser.utils.trajectory import load_exp_trajectories, generate_trajectory_generic_policy


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')
if args.exp_traj_folder == None:
    args.exp_traj_folder = args.dataset
args.savepath = f'logs/{args.dataset}/metrics/bc_hm'
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

env = datasets.load_environment(args.dataset)
env_observation_space = Box(low=-np.inf, high=np.inf, shape=(1,4), dtype=np.float32)
env_action_space = Box(low = -1.0, high= 1.0, shape=(1,2), dtype=np.float32)

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

exp_trajectories = load_exp_trajectories(folder=f'exp_trajectories/{args.exp_traj_folder}')
print(len(exp_trajectories))

##--- Transform expert trajectories to be used with imitation ---##

im_expert_trajectories = []

for traj in exp_trajectories:
    traj = torch.squeeze(traj)
    # acts = traj[:-1, :2].numpy()
    # obs = traj[:, 2:].numpy()
    obs = traj[:, :4].numpy()
    acts = traj[:-1, 4:].numpy()
    im_traj = Trajectory(obs, acts, infos=None, terminal=True)
    im_expert_trajectories.append(im_traj)


##--- Behaviour Cloning ---##

bc = BC(observation_space = env_observation_space, 
        action_space = env_action_space, 
        rng = np.random.default_rng(seed=42), 
        demonstrations=im_expert_trajectories,)

bc.train(n_epochs=10000)

with open(f'logs/{args.dataset}/metrics/starting_positions.json', 'r') as file:
    starting_points = json.load(file)

for starting_point, idx in zip(starting_points, range(len(starting_points))):
    rollout, _ = generate_trajectory_generic_policy(env, bc.policy, args, 
                                                    starting_location=tuple(starting_point), 
                                                    n_timesteps=args.n_timesteps, 
                                                    verbose=True)

    rollout = np.array(rollout)[None]

    np.save(f'{args.savepath}/rollout_{idx}.npy', rollout)
    renderer.composite(f'{args.savepath}/rollout_{idx}.pdf', rollout, ncol=1)
    renderer.composite(f'{args.savepath}/rollout_{idx}.jpg', rollout, ncol=1)