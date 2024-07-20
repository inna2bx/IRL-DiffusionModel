import json
import numpy as np
from os.path import join
import os
import pdb
import torch

import matplotlib.pyplot as plt

#from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction


def load_exp_trajectories():
    trajectory_files = os.listdir('exp_trajectories')
    exp_trajectories = []
    for file in trajectory_files:
        exp_trajectory = torch.load(f'exp_trajectories/{file}')
        exp_trajectories.append(exp_trajectory)
    
    return exp_trajectories



class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('inv')


# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)


diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

#---------------------------- inv value function -----------------------------#
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

inv_value_function = InvValueFunction(
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,)

inv_guide_config = utils.Config(args.guide, 
                                model=inv_value_function, 
                                verbose=False)
inv_guide = inv_guide_config()

inv_policy_config = utils.Config(
    args.policy,
    guide=inv_guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

inv_policy = inv_policy_config()

#---------------------------------- main loop ----------------------------------#

#generating exp Trajectories
exp_trajectories = load_exp_trajectories()



optimiser = torch.optim.Adam(inv_guide.parameters(), lr=2e-4)

losses = []
K = 10

for epoch in range(200):
    print(f'epoch: {epoch}')
    epoch_loss = 0
    for exp_trajectory in exp_trajectories:
        exp_trajectory_len = exp_trajectory.shape[1]
        for t in range(exp_trajectory_len):
            if t % K == 0:
                optimiser.zero_grad()
                clipped_exp_trajectory = exp_trajectory[:,t:t+K, :]
                first_observation = clipped_exp_trajectory[0, 0, :4].reshape((4)).detach().numpy()

                conditions = {0: first_observation}
                _, _, sampled_trajectory = inv_policy(conditions, batch_size=args.batch_size, verbose=args.verbose, return_tensor = True)
                clipped_sampled_trajectory = sampled_trajectory[:,:K,:]

                loss = torch.sum(torch.square(clipped_exp_trajectory - clipped_sampled_trajectory)) / (K*6)

                loss.backward()
                optimiser.step()



                epoch_loss += loss.detach()
    print(epoch_loss)
    losses.append(epoch_loss)

    if epoch % 10 == 0:
        plt.plot(losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')


plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss.png')


observation = env.reset_to_location((1, 1))

if args.conditional:
    print('Resetting target')
    env.set_target()
rollout = [observation.copy()]

total_reward = 0

for t in range(env.max_episode_steps):
    conditions = {0: observation}
    
    action, samples = inv_policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    if terminal:
        break

    observation = next_observation

print(join(args.savepath, 'rollout.png'))
renderer.composite(join(args.savepath, 'rollout_inv.png'), np.array(rollout)[None], ncol=1)

## save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)