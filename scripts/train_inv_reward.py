import json
import numpy as np
from os.path import join
import os
import pdb
import torch

import matplotlib.pyplot as plt

import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import load_exp_trajectories, generate_trajectory

def plot_loss(loss):
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(join(args.savepath, 'loss_inv_reward_training.pdf'),
                        bbox_inches='tight',
                        dpi=400,
                        transparent=True)


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('inv')

TRAJ_STEP_SIZE = 10

N_EPOCHS = 500

GAMMA_LOSS = 0.6

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
exp_trajectories = load_exp_trajectories(n_trajectories=2)



optimiser = torch.optim.Adam(inv_guide.parameters(), lr=2e-4)
loss_weight = torch.tensor([GAMMA_LOSS]).repeat(args.horizon)
exponents = torch.arange(args.horizon)
loss_weight = torch.pow(loss_weight, exponents)

losses = []


for epoch in range(N_EPOCHS):
    print(f'epoch: {epoch}')
    epoch_loss = 0
    for exp_trajectory in exp_trajectories:
        exp_trajectory_len = exp_trajectory.shape[1]
        for t in range(exp_trajectory_len):
            if t % TRAJ_STEP_SIZE == 0:
                optimiser.zero_grad()

                index_end = t+args.horizon
                if index_end < exp_trajectory_len:
                    clipped_exp_trajectory = exp_trajectory[:,t:index_end, :]
                else:
                    clipped_exp_trajectory = exp_trajectory[:,t:, :]

                clipped_exp_trajectory_len = clipped_exp_trajectory.shape[1]
                
                first_observation = clipped_exp_trajectory[0, 0, :4].reshape((4)).detach().numpy()

                conditions = {0: first_observation}
                _, _, sampled_trajectory = inv_policy(conditions, batch_size=args.batch_size, return_tensor = True)
                clipped_sampled_trajectory = sampled_trajectory[:,:clipped_exp_trajectory_len,:]
                clipped_loss_weight = loss_weight[:clipped_exp_trajectory_len]

                clipped_loss_weight = clipped_loss_weight[None,:,None]

                loss = torch.sum(clipped_loss_weight*torch.square(clipped_exp_trajectory - clipped_sampled_trajectory)) / (torch.sum(clipped_loss_weight) * 6)

                loss.backward()
                optimiser.step()
                epoch_loss += loss.detach()
    
    print(epoch_loss)
    losses.append(epoch_loss)

    if epoch % 10 == 0:
        plot_loss(losses)
        torch.save(inv_guide.state_dict(), 
                   join(args.savepath, 'model_weights.pth'))


plot_loss(losses)
torch.save(inv_guide.state_dict(), join(args.savepath, 'model_weights.pth'))


rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(1,1))

renderer.composite(join(args.savepath, 'rollout_inv_1_1.pdf'), np.array(rollout)[None], ncol=1)

rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(3,1))

renderer.composite(join(args.savepath, 'rollout_inv_3_1.pdf'), np.array(rollout)[None], ncol=1)

rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(2,3))

renderer.composite(join(args.savepath, 'rollout_inv_2_3.pdf'), np.array(rollout)[None], ncol=1)