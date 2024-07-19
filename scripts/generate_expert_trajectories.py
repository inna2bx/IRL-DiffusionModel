import json
import numpy as np
from os.path import join
import torch

#from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling

def generate_trajectory(env, policy, starting_location = (1,1)):
    observation = env.reset_to_location(starting_location)
    rollout = [observation.copy()]

    trajectory = []

    total_reward = 0

    for t in range(env.max_episode_steps):

        print(f't: {t}')
        conditions = {0: observation}
        
        action, _ = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        observation_tensor = torch.from_numpy(observation)
        action_tensor = torch.from_numpy(action)
        trajectory.append(torch.cat((observation_tensor, action_tensor)))

        ## update rollout observations
        rollout.append(next_observation.copy())

        if terminal:
            break

        observation = next_observation
    
    trajectory = torch.stack(trajectory, dim= 0).reshape((1, env.max_episode_steps, 6))

    return rollout, trajectory


NUM_EXP_TRAJECTORY = 2

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

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
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
    _, exp_trajectory = generate_trajectory(env, policy)
    torch.save(exp_trajectory, f'exp_trajectories/idx_{idx}.pt')
