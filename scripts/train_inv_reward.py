import json
import numpy as np
from os.path import join
import os
import torch
import time

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

GAMMA_LOSS = 0.7

PROFILING = False

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, device=args.device
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

if not os.path.exists(join(args.savepath, 'rollouts')):
        os.makedirs(join(args.savepath, 'rollouts'))

if PROFILING:
    epoch_times = []
    iteration_times = []
    sampling_times = []
    backprop_times = []
    if not os.path.exists(join(args.savepath, 'times')):
        os.makedirs(join(args.savepath, 'times'))


for epoch in range(N_EPOCHS):
    if PROFILING:
        epoch_time_start = time.time()
    
    print(f'epoch: {epoch}')
    epoch_loss = 0
    for exp_trajectory in exp_trajectories:
        exp_trajectory_len = exp_trajectory.shape[1]
        for t in range(exp_trajectory_len):
            if t % TRAJ_STEP_SIZE == 0:
                if PROFILING:
                    iteration_time_start = time.time()

                optimiser.zero_grad()

                index_end = t+args.horizon
                if index_end < exp_trajectory_len:
                    clipped_exp_trajectory = exp_trajectory[:,t:index_end, :]
                else:
                    clipped_exp_trajectory = exp_trajectory[:,t:, :]

                clipped_exp_trajectory_len = clipped_exp_trajectory.shape[1]
                
                first_observation = clipped_exp_trajectory[0, 0, :4].reshape((4)).detach().numpy()

                conditions = {0: first_observation}

                if PROFILING:
                    sampling_time_start = time.time()

                _, _, sampled_trajectory = inv_policy(conditions, batch_size=args.batch_size, no_grad_diff_steps=0, return_tensor = True)
                
                if PROFILING:
                    sampling_time_end = time.time()
                    sampling_times.append(sampling_time_end - sampling_time_start)
                
                clipped_sampled_trajectory = sampled_trajectory[:,:clipped_exp_trajectory_len,:]
                clipped_loss_weight = loss_weight[:clipped_exp_trajectory_len]

                clipped_loss_weight = clipped_loss_weight[None,:,None]

                loss = torch.sum(clipped_loss_weight*torch.square(clipped_exp_trajectory - clipped_sampled_trajectory)) / (torch.sum(clipped_loss_weight) * 6)

                if PROFILING:
                    backprop_time_start = time.time()
                
                loss.backward()
                optimiser.step()
                
                if PROFILING:
                    backprop_time_end = time.time()
                    backprop_times.append(backprop_time_end - backprop_time_start)
                
                epoch_loss += loss.detach()

                if PROFILING:
                    iteration_time_end = time.time()
                    iteration_times.append(iteration_time_end - iteration_time_start)
    
    print(epoch_loss)
    losses.append(epoch_loss)

    if PROFILING:
        epoch_time_end = time.time()
        epoch_times.append(epoch_time_end - epoch_time_start)

    if epoch % 10 == 0:
        plot_loss(losses)
        torch.save(inv_guide.state_dict(), 
                   join(args.savepath, 'model_weights.pth'))


plot_loss(losses)
np.save(join(args.savepath, 'losses.npy'), np.array(losses, dtype=object), allow_pickle=True)
torch.save(inv_guide.state_dict(), join(args.savepath, 'model_weights.pth'))


if PROFILING:
    epoch_times = np.array(epoch_times)
    iteration_times = np.array(iteration_times)
    sampling_times = np.array(sampling_times)
    backprop_times = np.array(backprop_times)

    epoch_times_mean = np.round(np.mean(epoch_times), 2)
    epoch_times_std = np.round(np.std(epoch_times), 2)
    iteration_times_mean = np.round(np.mean(iteration_times), 2)
    iteration_times_std = np.round(np.std(iteration_times), 2)
    sampling_times_mean = np.round(np.mean(sampling_times), 2)
    sampling_times_std = np.round(np.std(sampling_times), 2)
    backprop_times_mean = np.round(np.mean(backprop_times), 2)
    backprop_times_std = np.round(np.std(backprop_times), 2)

    report_str = f'Epochs time: {epoch_times_mean} +- {epoch_times_std}\n'
    report_str += f'Iteration time: {iteration_times_mean} +- {iteration_times_std}\n'
    report_str += f'Sampling time: {sampling_times_mean} +- {sampling_times_std}\n'
    report_str += f'Backprop time: {backprop_times_mean} +- {backprop_times_std}'

    np.save(join(args.savepath, 'times/epoch_times.npy'), epoch_times)
    np.save(join(args.savepath, 'times/iteration_times.npy'), iteration_times)
    np.save(join(args.savepath, 'times/sampling_times.npy'), sampling_times)
    np.save(join(args.savepath, 'times/backprop_times.npy'), backprop_times)

    print(report_str)
    with open(join(args.savepath, 'times_report.txt'), 'w') as f:
        f.write(report_str)

rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(1,1))
np.save(join(args.savepath, 'rollouts/rollout_inv_1_1.npy'), np.array(rollout)[None])
renderer.composite(join(args.savepath, 'rollouts/rollout_inv_1_1.pdf'), np.array(rollout)[None], ncol=1)

rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(3,1))
np.save(join(args.savepath, 'rollouts/rollout_inv_3_1.npy'), np.array(rollout)[None])
renderer.composite(join(args.savepath, 'rollouts/rollout_inv_3_1.pdf'), np.array(rollout)[None], ncol=1)

rollout, _ = generate_trajectory(env, inv_policy, args, starting_location=(2,3))
np.save(join(args.savepath, 'rollouts/rollout_inv_2_3.npy'), np.array(rollout)[None])
renderer.composite(join(args.savepath, 'rollouts/rollout_inv_2_3.pdf'), np.array(rollout)[None], ncol=1)