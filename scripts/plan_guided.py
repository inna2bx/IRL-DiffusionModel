import json
import numpy as np
from os.path import join
import pdb

#from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.sampling as sampling


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)
#---------------------------------- loading ----------------------------------#

#diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

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


#policy = Policy(diffusion, dataset.normalizer)

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

#---------------------------------- main loop ----------------------------------#
observation = env.reset_to_location((1, 1))

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
# target = env._target
# cond = {
#     diffusion.horizon - 1: np.array([*target, 0, 0]),
# }

## observations for rendering
rollout = [observation.copy()]

total_reward = 0


for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    #if t == 0:
        #cond[0] = observation

        #action, samples = policy(cond, batch_size=args.batch_size)
    conditions = {0: observation}
    
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    # actions = samples.actions[0]
    # sequence = samples.observations[0]
    # # pdb.set_trace()

    # # ####
    # if t < len(sequence) - 1:
    #     next_waypoint = sequence[t+1]
    # else:
    #     next_waypoint = sequence[-1].copy()
    #     next_waypoint[2:] = 0
    #     # pdb.set_trace()

    # ## can use actions or define a simple controller based on state predictions
    # action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    # # pdb.set_trace()
    # ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



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

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
















# import pdb

# import diffuser.sampling as sampling
# import diffuser.utils as utils


# #-----------------------------------------------------------------------------#
# #----------------------------------- setup -----------------------------------#
# #-----------------------------------------------------------------------------#

# class Parser(utils.Parser):
#     dataset: str = 'walker2d-medium-replay-v2'
#     config: str = 'config.locomotion'

# args = Parser().parse_args('plan')


# #-----------------------------------------------------------------------------#
# #---------------------------------- loading ----------------------------------#
# #-----------------------------------------------------------------------------#

# ## load diffusion model and value function from disk
# diffusion_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.diffusion_loadpath,
#     epoch=args.diffusion_epoch, seed=args.seed,
# )
# value_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.value_loadpath,
#     epoch=args.value_epoch, seed=args.seed,
# )

# ## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

# diffusion = diffusion_experiment.ema
# dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer

# ## initialize value guide
# value_function = value_experiment.ema
# guide_config = utils.Config(args.guide, model=value_function, verbose=False)
# guide = guide_config()

# logger_config = utils.Config(
#     utils.Logger,
#     renderer=renderer,
#     logpath=args.savepath,
#     vis_freq=args.vis_freq,
#     max_render=args.max_render,
# )

# ## policies are wrappers around an unconditional diffusion model and a value guide
# policy_config = utils.Config(
#     args.policy,
#     guide=guide,
#     scale=args.scale,
#     diffusion_model=diffusion,
#     normalizer=dataset.normalizer,
#     preprocess_fns=args.preprocess_fns,
#     ## sampling kwargs
#     sample_fn=sampling.n_step_guided_p_sample,
#     n_guide_steps=args.n_guide_steps,
#     t_stopgrad=args.t_stopgrad,
#     scale_grad_by_std=args.scale_grad_by_std,
#     verbose=False,
# )

# logger = logger_config()
# policy = policy_config()


# #-----------------------------------------------------------------------------#
# #--------------------------------- main loop ---------------------------------#
# #-----------------------------------------------------------------------------#

# env = dataset.env
# observation = env.reset()

# ## observations for rendering
# rollout = [observation.copy()]

# total_reward = 0
# for t in range(args.max_episode_length):

#     if t % 10 == 0: print(args.savepath, flush=True)

#     ## save state for rendering only
#     state = env.state_vector().copy()

#     ## format current observation for conditioning
#     conditions = {0: observation}
#     action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

#     ## execute action in environment
#     next_observation, reward, terminal, _ = env.step(action)

#     ## print reward and score
#     total_reward += reward
#     score = env.get_normalized_score(total_reward)
#     print(
#         f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
#         f'values: {samples.values} | scale: {args.scale}',
#         flush=True,
#     )

#     ## update rollout observations
#     rollout.append(next_observation.copy())

#     ## render every `args.vis_freq` steps
#     logger.log(t, samples, state, rollout)

#     if terminal:
#         break

#     observation = next_observation

# ## write results to json file at `args.savepath`
# logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
