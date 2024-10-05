import os
from os.path import join
import json

import torch
import numpy as np

import diffuser.utils as utils
import diffuser.datasets as datasets
import diffuser.sampling as sampling
from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import  generate_trajectory

## parameters needed
# --irl_exp_name


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

ROLLOUT_PER_JOB = 10

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

inv_value_function_config = utils.Config(args.inv_network, horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,)

inv_value_function = inv_value_function_config()
inv_value_function.to(args.device)

irl_savepath = f'logs/{args.dataset}/irl/{args.irl_exp_name}/0'
args.savepath = f'logs/{args.dataset}/metrics/{args.irl_exp_name}/{args.method_metric_name}'
if not os.path.exists(args.savepath):
    try:  
        os.makedirs(args.savepath)  
    except OSError as error:  
        print(error)

if args.load_weights:
    inv_value_function.load_state_dict(torch.load(join(irl_savepath, 
                                                    'model_weights.pth'),
                                                map_location=torch.device(args.device)))

inv_guide_config = utils.Config(args.guide, 
                                model=inv_value_function, 
                                verbose=False)
inv_guide = inv_guide_config()

policy_config = utils.Config(
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

policy = policy_config()

with open(f'logs/{args.dataset}/metrics/starting_positions.json', 'r') as file:
    starting_points = json.load(file)

starting_idx = (args.metrics_index-1) * ROLLOUT_PER_JOB
starting_points = starting_points[starting_idx:starting_idx+ROLLOUT_PER_JOB]

for starting_point, idx in zip(starting_points, range(starting_idx, starting_idx+ROLLOUT_PER_JOB)):
    rollout, _ = generate_trajectory(env, policy, args, 
                                     starting_location=tuple(starting_point), 
                                     n_timesteps=args.n_timesteps, 
                                     n_same_plan_actions=args.n_same_plan_actions,
                                     verbose=True)

    rollout = np.array(rollout)[None]

    np.save(f'{args.savepath}/rollout_{idx}.npy', rollout)
    renderer.composite(f'{args.savepath}/rollout_{idx}.pdf', rollout, ncol=1)
    renderer.composite(f'{args.savepath}/rollout_{idx}.jpg', rollout, ncol=1)