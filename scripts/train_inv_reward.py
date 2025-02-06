import numpy as np
from os.path import join
import os
import time
import random
import wandb

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import ExpTrajDataset, generate_trajectory

def plot_loss(loss):
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(join(args.savepath, 'loss_inv_reward_training.pdf'),
                        bbox_inches='tight',
                        dpi=400,
                        transparent=True)

def loss_function(exp_trajectory, sampled_trajectories, weights):
    n_batches = sampled_trajectories.shape[0]
    exp_trajectory_batch = exp_trajectory.repeat(n_batches, 1, 1)
    loss = torch.sum(weights*torch.square(exp_trajectory_batch - sampled_trajectories)) 
    normalised_loss = loss / (torch.sum(weights) * 6 * n_batches)
    return normalised_loss

def my_collate(batch):
    return list(batch)

def transform_exp_traj(exp_traj, normalizer,device):
    exp_traj_np = exp_traj.detach().cpu().numpy()
    obs = normalizer.normalize(exp_traj_np[:,:4], 'observations')
    act = normalizer.normalize(exp_traj_np[:,4:], 'actions')
    exp_traj_normed = np.concatenate((act, obs), axis=-1)
    return torch.from_numpy(exp_traj_normed).to(device)

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('inv')

if args.exp_traj_folder == None:
    args.exp_traj_folder = args.dataset

wandb.init(
    # set the wandb project where this run will be logged
    project='IRL with Diffuser',
    name = args.exp_name,
    mode='online'
    # track hyperparameters and run metadata
)

PROFILING = False

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, device=args.device
)

diffusion = diffusion_experiment.ema
diffusion.to(args.device)
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

#---------------------------- inv value function -----------------------------#
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# inv_value_function = InvValueFunction(
#     horizon=args.horizon,
#     transition_dim=observation_dim + action_dim,
#     cond_dim=observation_dim,
#     dim_mults=args.dim_mults,)

# inv_value_function.to(args.device)

inv_value_function_config = utils.Config(args.inv_network, horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,)

inv_value_function = inv_value_function_config()
inv_value_function.to(args.device)

print(type(inv_value_function))

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
exp_traj_dataset = ExpTrajDataset(n_trajectories=args.n_expert_traj, 
                                  device=args.device,
                                  folder=f'exp_trajectories/{args.exp_traj_folder}',
                                  horizon=args.horizon,
                                  n_sample_for_trajectory=args.n_sample_for_trajectory)

optimiser = torch.optim.Adam(inv_guide.parameters(), lr=args.lr)
loss_weight = torch.tensor([args.gamma_loss]).repeat(args.horizon)
exponents = torch.arange(args.horizon)
loss_weight = torch.pow(loss_weight, exponents).to(args.device)

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

for epoch in range(args.n_epochs):
    if PROFILING:
        epoch_time_start = time.time()
    
    print(f'epoch: {epoch}')
    epoch_loss = 0
    epoch_loss_n = 0
    optimiser.zero_grad()

    exp_traj_dataloader = DataLoader(exp_traj_dataset, 
                                 batch_size=args.inv_batch_size,
                                 shuffle=True,
                                 collate_fn=my_collate)
    exp_traj_dataloader = iter(exp_traj_dataloader)
    for _ in range(args.n_minibatch_per_epoch):
        batch = next(exp_traj_dataloader)
        for sample in batch:
            exp_traj, starting_obs = sample
            exp_traj = transform_exp_traj(exp_traj, dataset.normalizer, args.device)
            
            if PROFILING:
                iteration_time_start = time.time()

            conditions = {0: starting_obs}

            if PROFILING:
                sampling_time_start = time.time()

            _, _, sampled_trajectory = inv_policy(conditions, 
                                                    batch_size=args.batch_size, 
                                                    no_grad_diff_steps=args.no_grad_diff_steps, 
                                                    fast_sampling_batch_size = args.fast_sampling_batch_size, 
                                                    return_tensor = True, verbose = False)
            
            
            if PROFILING:
                sampling_time_end = time.time()
                sampling_times.append(sampling_time_end - sampling_time_start)
            
            clipped_sampled_trajectory = sampled_trajectory[:,:len(exp_traj),:]
            clipped_loss_weight = loss_weight[:len(exp_traj)]

            clipped_loss_weight = clipped_loss_weight[None,:,None]
            loss = loss_function(exp_traj, 
                                 clipped_sampled_trajectory, 
                                 clipped_loss_weight)

            if PROFILING:
                backprop_time_start = time.time()
            
            loss.backward()

            if PROFILING:
                backprop_time_end = time.time()
                backprop_times.append(backprop_time_end - backprop_time_start)
            
            
            epoch_loss = (loss.cpu().detach() + epoch_loss * epoch_loss_n) / (epoch_loss_n + 1)
            epoch_loss_n += 1 

            if PROFILING:
                iteration_time_end = time.time()
                iteration_times.append(iteration_time_end - iteration_time_start)
        
    optimiser.step()
    
    
    
    wandb.log({"loss": epoch_loss})
    print(epoch_loss)
    losses.append(epoch_loss)


    if PROFILING:
        epoch_time_end = time.time()
        epoch_times.append(epoch_time_end - epoch_time_start)

    if epoch % 10 == 0:
        plot_loss(losses)
        np.save(join(args.savepath, 'losses.npy'), np.array(losses, dtype=object), allow_pickle=True)
        torch.save(inv_value_function.state_dict(), 
                   join(args.savepath, 'model_weights.pth'))


wandb.finish()
print('Training finished')
plot_loss(losses)
np.save(join(args.savepath, 'losses.npy'), np.array(losses, dtype=object), allow_pickle=True)
torch.save(inv_value_function.state_dict(), join(args.savepath, 'model_weights.pth'))


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

for i in range(args.n_final_samples):
    rollout, _ = generate_trajectory(env, inv_policy, args)
    np.save(join(args.savepath, f'rollouts/rollout_inv_{i}.npy'), np.array(rollout)[None])
    renderer.composite(join(args.savepath, f'rollouts/rollout_inv_{i}.pdf'), np.array(rollout)[None], ncol=1)