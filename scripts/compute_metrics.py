from os.path import join
import os
import numpy as np
import torch

import diffuser.utils as utils
import diffuser.datasets as datasets
from diffuser.guides.policies import Policy

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import  generate_trajectory

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

##--- Reward Function ---##
def reward_function(state):
    return state[0]

def cumulative_reward_function(trajectory):
    return np.sum(np.apply_along_axis(reward_function, 2, trajectory))



##--- Load Diffusers ---##

env = datasets.load_environment(args.dataset)

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, 
                                            args.diffusion_loadpath, 
                                            epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer


policy = Policy(diffusion, dataset.normalizer)


env = datasets.load_environment(args.dataset)

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

irl_savepath = args.savepath.replace('metrics', 'irl')

inv_value_function.load_state_dict(torch.load(join(irl_savepath, 'model_weights.pth')))

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

##--- Base diffuser ---##
base_diffuser_rewards = np.zeros(args.n_rollout)

base_diffuser_path = 'rollouts/base_diffuser'
if not os.path.exists(join(args.savepath, base_diffuser_path)):
    os.makedirs(join(args.savepath, base_diffuser_path))

for i in range(args.n_rollout):

    rollout, _ = generate_trajectory(env, base_policy, args)

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

    rollout, _ = generate_trajectory(env, inv_policy, args)

    rollout = np.array(rollout)[None]
    guided_diffuser_rewards[i] = cumulative_reward_function(rollout)

    np.save(join(args.savepath, f'{guided_diffuser_path}/rollout_{i}.npy'), rollout)
    renderer.composite(join(args.savepath, f'{guided_diffuser_path}/rollout_{i}.pdf'), rollout, ncol=1)
    guided_diffuser_rewards[i] = cumulative_reward_function(rollout)

print(f'{base_diffuser_rewards.mean()} +- {base_diffuser_rewards.std()}')
print(f'{guided_diffuser_rewards.mean()} +- {guided_diffuser_rewards.std()}')
