import json
import numpy as np
from os.path import join
import torch

#from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling

import diffuser.utils.trajectory as trajectory


NUM_EXP_TRAJECTORY = 64

FOLDER = 'exp_trajectories_mixed'

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('inv')

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, device=args.device
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed, device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

policy_config = utils.Config(
    args.policy,
    guide=guide,
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

for idx in range(NUM_EXP_TRAJECTORY):
    print(f'---------------------- {idx} ----------------------')
    exp_rollout, exp_trajectory = trajectory.generate_trajectory(env, policy, args, starting_location = (1.5, 3))
    torch.save(exp_trajectory, f'{FOLDER}/idx_{idx}.pt')
    renderer.composite(f'{FOLDER}/idx_{idx}_rollout.pdf', np.array(exp_rollout)[None], ncol=1)
