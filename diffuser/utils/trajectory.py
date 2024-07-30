import torch
import os

def generate_trajectory(env, policy, args, starting_location = None, 
                        n_timesteps = None):
    if starting_location != None:
        observation = env.reset_to_location(starting_location)
    else:
        observation = env.reset()
    
    rollout = [observation.copy()]

    trajectory = []

    total_reward = 0
    if n_timesteps == None:
        n_timesteps = env.max_episode_steps

    for t in range(n_timesteps):

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



def load_exp_trajectories(n_trajectories = None, folder='exp_trajectories'):
    trajectory_files = os.listdir(folder)
    trajectory_files = [l for l in trajectory_files if l[-3:] == '.pt']
    n_files = len(trajectory_files)
    
    if n_trajectories != None and n_trajectories < n_files:
        trajectory_files = trajectory_files[:n_trajectories]

    exp_trajectories = []
    for file in trajectory_files:
        exp_trajectory = torch.load(f'exp_trajectories/{file}')
        exp_trajectories.append(exp_trajectory)
    
    return exp_trajectories