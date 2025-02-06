from os.path import join
import os
import numpy as np
import torch
from gymnasium.spaces import Box

from imitation.data.types import Trajectory
from imitation.algorithms.bc import BC

import diffuser.utils as utils
import diffuser.datasets as datasets

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import  generate_trajectory, load_exp_trajectories, generate_trajectory_generic_policy

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

##--- Reward Function ---##
def reward_function(state):
    return -((6 - state[0])**2 + (6 - state[1])**2)

def cumulative_reward_function(trajectory):
    return np.sum(np.apply_along_axis(reward_function, 2, trajectory))



##--- Load Diffusers ---##


env = datasets.load_environment(args.dataset)
env_observation_space = Box(low=-np.inf, high=np.inf, shape=(1,4), dtype=np.float32)
env_action_space = Box(low = -1.0, high= 1.0, shape=(1,2), dtype=np.float32)


exp_trajectories = load_exp_trajectories(n_trajectories=args.n_expert_traj, 
                                         folder=f'exp_trajectories/{args.exp_traj_folder}')

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

inv_value_function = InvValueFunction(
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,)

irl_savepath = f'logs/{args.dataset}/irl/{args.irl_exp_name}/0'
args.savepath = f'logs/{args.dataset}/metrics/{args.irl_exp_name}_{args.n_timesteps}_{args.n_same_plan_actions}'

inv_value_function.load_state_dict(torch.load(join(irl_savepath, 
                                                   'model_weights.pth'),
                                              map_location=torch.device(args.device)))

inv_guide_config = utils.Config(args.guide, 
                                model=inv_value_function, 
                                verbose=False)
inv_guide = inv_guide_config()

base_policy_config = utils.Config(
    args.policy,
    guide=inv_guide,
    scale=0,
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

base_policy = base_policy_config()

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

##--- Transform expert trajectories to be used with imitation ---##

im_expert_trajectories = []

for traj in exp_trajectories:
    traj = torch.squeeze(traj)
    acts = traj[:-1, :2].numpy()
    obs = traj[:, 2:].numpy()
    im_traj = Trajectory(obs, acts, infos=None, terminal=True)
    im_expert_trajectories.append(im_traj)


##--- Behaviour Cloning ---##

bc = BC(observation_space = env_observation_space, 
        action_space = env_action_space, 
        rng = np.random.default_rng(seed=42), 
        demonstrations=im_expert_trajectories,)

bc.train(n_epochs=200)

##--- Compute Metrics ---##

base_diffuser_rewards = np.zeros(args.n_rollout)

base_diffuser_path = 'rollouts/base_diffuser'

if not os.path.exists(join(args.savepath, base_diffuser_path)):
    os.makedirs(join(args.savepath, base_diffuser_path))


for i in range(args.n_rollout):
    position = tuple(env.reset()[:2])
    rollout, _ = generate_trajectory(env, base_policy, args, 
                                     n_timesteps=args.n_timesteps, 
                                     n_same_plan_actions=args.n_same_plan_actions)

    rollout = np.array(rollout)[None]

    np.save(join(args.savepath, f'{base_diffuser_path}/rollout_{i}.npy'), rollout)
    renderer.composite(join(args.savepath, f'{base_diffuser_path}/rollout_{i}.pdf'), rollout, ncol=1)
    base_diffuser_rewards[i] = cumulative_reward_function(rollout)







##--- Base diffuser ---##
base_diffuser_rewards = np.zeros(args.n_rollout)

base_diffuser_path = 'rollouts/base_diffuser'
if not os.path.exists(join(args.savepath, base_diffuser_path)):
    os.makedirs(join(args.savepath, base_diffuser_path))

for i in range(args.n_rollout):

    rollout, _ = generate_trajectory(env, base_policy, args, 
                                     n_timesteps=args.n_timesteps, 
                                     n_same_plan_actions=args.n_same_plan_actions)

    rollout = np.array(rollout)[None]

    np.save(join(args.savepath, f'{base_diffuser_path}/rollout_{i}.npy'), rollout)
    renderer.composite(join(args.savepath, f'{base_diffuser_path}/rollout_{i}.pdf'), rollout, ncol=1)
    base_diffuser_rewards[i] = cumulative_reward_function(rollout)

##--- Guided diffuser ---##
guided_diffuser_rewards = np.zeros(args.n_rollout)

guided_diffuser_path = 'rollouts/guided_diffuser'
if not os.path.exists(join(args.savepath, guided_diffuser_path)):
    os.makedirs(join(args.savepath, guided_diffuser_path))

for i in range(args.n_rollout):

    rollout, _ = generate_trajectory(env, inv_policy, args,
                                     n_timesteps=args.n_timesteps, 
                                     n_same_plan_actions=args.n_same_plan_actions)

    rollout = np.array(rollout)[None]
    guided_diffuser_rewards[i] = cumulative_reward_function(rollout)

    np.save(join(args.savepath, f'{guided_diffuser_path}/rollout_{i}.npy'), rollout)
    renderer.composite(join(args.savepath, f'{guided_diffuser_path}/rollout_{i}.pdf'), rollout, ncol=1)
    guided_diffuser_rewards[i] = cumulative_reward_function(rollout)

bc_rewards = np.zeros(args.n_rollout)

bc_path = 'rollouts/bc'
if not os.path.exists(join(args.savepath, bc_path)):
    os.makedirs(join(args.savepath, bc_path))

for i in range(args.n_rollout):

    rollout, _ = generate_trajectory_generic_policy(env, bc.policy, args,
                                                    n_timesteps=args.n_timesteps)

    rollout = np.array(rollout)[None]
    bc_rewards[i] = cumulative_reward_function(rollout)

    np.save(join(args.savepath, f'{bc_path}/rollout_{i}.npy'), rollout)
    renderer.composite(join(args.savepath, f'{bc_path}/rollout_{i}.pdf'), rollout, ncol=1)
    bc_rewards[i] = cumulative_reward_function(rollout)

output_string = f'{base_diffuser_rewards.mean()} +- {base_diffuser_rewards.std()}'
output_string += f'\n{guided_diffuser_rewards.mean()} +- {guided_diffuser_rewards.std()}'
output_string += f'\n{bc_rewards.mean()} +- {bc_rewards.std()}'
print(output_string)

with open(join(args.savepath, 'metrics.txt'), 'w') as txt_file:
    txt_file.write(output_string)