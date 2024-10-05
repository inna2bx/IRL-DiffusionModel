import os
import torch
import re
import numpy as np
from torch.utils.data import Dataset


def generate_exp_trajectory(env, policy, args, starting_location = None, 
                            n_timesteps = None, n_same_plan_actions = 1,
                            verbose = False, target=None, use_planner = True, 
                            renderer = None, idx=0, folder = '', distance = 0.3):
    if starting_location != None:
        observation = env.reset_to_location(starting_location)
    else:
        observation = env.reset()
        while np.linalg.norm(env.state_vector()[:2] - target) < distance:
            observation = env.reset()
    
    rollout = [observation.copy()]

    trajectory = []

    total_reward = 0
    if n_timesteps == None:
        n_timesteps = env.max_episode_steps

    t = 0
    done = False
    while t < n_timesteps and not done:
        conditions = {0: observation}
        if target != None:
            conditions[args.horizon-1] = np.array([*target, 0, 0])

        actions, samples = policy(conditions, 
                                  batch_size=args.batch_size, 
                                  verbose=verbose)
        
        renderer.composite(f'{folder}/plans/idx_{idx}-{t}_rollout.pdf', np.array(samples.observations[0])[None], ncol=1)
        renderer.composite(f'{folder}/plans/idx_{idx}-{t}_rollout.jpg', np.array(samples.observations[0])[None], ncol=1)
        actions = samples.actions[0]
        sequence = samples.observations[0]
        for step in range(n_same_plan_actions):
            if verbose:
                print(f't:{t+1}/{n_timesteps}')
            
            if t >= n_timesteps:
                break

            state = env.state_vector().copy()

            if use_planner:
                next_waypoint = sequence[step+1]
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            else:
                action = actions[step]
                    
            next_observation, reward, _, _ = env.step(action)

            if np.linalg.norm(env.state_vector()[:2] - target) < distance:
                done = True 

            total_reward += reward

            observation_tensor = torch.from_numpy(observation)
            action_tensor = torch.from_numpy(action)
            trajectory.append(torch.cat((observation_tensor, action_tensor)))

            ## update rollout observations
            rollout.append(next_observation.copy())

            observation = next_observation

            t += 1

    
    trajectory = torch.stack(trajectory, dim= 0).reshape((1, t, 6))

    return rollout, trajectory

def generate_trajectory(env, policy, args, starting_location = None, 
                        n_timesteps = None, n_same_plan_actions = 1,
                        verbose = False, target=None, use_planner = False):
    if starting_location != None:
        observation = env.reset_to_location(starting_location)
    else:
        observation = env.reset()
    
    rollout = [observation.copy()]

    trajectory = []

    total_reward = 0
    if n_timesteps == None:
        n_timesteps = env.max_episode_steps

    t = 0

    while t < n_timesteps:
        conditions = {0: observation}
        if target != None and t == 0:
            conditions[args.horizon-1] = np.array([*target, 0, 0])

        actions, samples = policy(conditions, 
                                  batch_size=args.batch_size, 
                                  verbose=verbose)
    
        actions = samples.actions[0]
        sequence = samples.observations[0]
        for step in range(n_same_plan_actions):
            if verbose:
                print(f't:{t+1}/{n_timesteps}')
            
            if t >= n_timesteps:
                break

            state = env.state_vector().copy()

            if use_planner:
                next_waypoint = sequence[step+1]
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            else:
                action = actions[step]
                    
            next_observation, reward, terminal, _ = env.step(action)
            total_reward += reward

            observation_tensor = torch.from_numpy(observation)
            action_tensor = torch.from_numpy(action)
            trajectory.append(torch.cat((observation_tensor, action_tensor)))

            ## update rollout observations
            rollout.append(next_observation.copy())

            observation = next_observation

            t += 1

    
    trajectory = torch.stack(trajectory, dim= 0).reshape((1, n_timesteps, 6))

    return rollout, trajectory


def generate_trajectory_generic_policy(env, policy, args, starting_location = None, 
                        n_timesteps = None, verbose = False):
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
        if verbose:
            print(f't:{t+1}/{n_timesteps}')
        action = torch.squeeze(policy(torch.Tensor(observation)[None, :])[0])
        next_observation, reward, terminal, _ = env.step(action.detach().numpy())
        total_reward += reward

        observation_tensor = torch.from_numpy(observation)
        action_tensor = action
        trajectory.append(torch.cat((observation_tensor, action_tensor)))

        ## update rollout observations
        rollout.append(next_observation.copy())

        if terminal:
            break

        observation = next_observation
    
    trajectory = torch.stack(trajectory, dim= 0).reshape((1, n_timesteps, 6))

    return rollout, trajectory


def load_exp_trajectories(n_trajectories = None, device = 'cpu', folder='exp_trajectories'):
    trajectory_files = os.listdir(folder)
    trajectory_files = [l for l in trajectory_files if l[-3:] == '.pt']
    n_files = len(trajectory_files)
    
    if n_trajectories != None and n_trajectories < n_files:
        trajectory_files = trajectory_files[:n_trajectories]

    exp_trajectories = []
    for file in trajectory_files:
        exp_trajectory = torch.load(f'{folder}/{file}', 
                                    map_location=torch.device(device))
        exp_trajectory = exp_trajectory.to(device)
        exp_trajectories.append(exp_trajectory)
    
    return exp_trajectories

def load_rollouts(n_trajectories = None, folder=''):
    trajectory_files = os.listdir(folder)
    trajectory_files = [l for l in trajectory_files if l[-4:] == '.npy']
    trajectory_files.sort(key= lambda r: int(re.search("[0-9]+", r).group()))
    n_files = len(trajectory_files)
    
    if n_trajectories != None and n_trajectories < n_files:
        trajectory_files = trajectory_files[:n_trajectories]

    exp_trajectories = []
    for file in trajectory_files:
        exp_trajectory = np.load(f'{folder}/{file}')
        exp_trajectories.append(exp_trajectory)
    
    return exp_trajectories

class ExpTrajDataset(Dataset):
    def __init__(self, n_trajectories, device, folder, horizon, n_sample_for_trajectory):
        exp_trajectories = load_exp_trajectories(n_trajectories=n_trajectories, 
                                                 device=device,
                                                 folder=folder)
        self.data = []
        for exp_traj in exp_trajectories:
            exp_traj = torch.squeeze(exp_traj)
            for _ in range(n_sample_for_trajectory):
                timestep = np.random.randint(0, len(exp_traj))
                sample = exp_traj[timestep:, :]
                if len(sample) > horizon:
                    sample = sample[:horizon, :]
                self.data.append(sample)
        
        self.starting_obs = [exp_traj[0, :4].reshape((4)) 
                             for exp_traj in self.data]
        
        self.starting_obs = [starting_ob.detach().cpu().numpy() 
                             for starting_ob in self.starting_obs]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chosen_exp_traj = self.data[idx]
        chosen_starting_obs = self.starting_obs[idx]
        return chosen_exp_traj, chosen_starting_obs
        
        

        