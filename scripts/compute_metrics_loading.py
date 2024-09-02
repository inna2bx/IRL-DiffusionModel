from os.path import join
import os
import numpy as np
import torch
from gymnasium.spaces import Box

import matplotlib.pyplot as plt

import diffuser.utils as utils
import diffuser.datasets as datasets
from diffuser.guides.policies import Policy

import diffuser.sampling as sampling

from diffuser.models.temporal import InvValueFunction
from diffuser.utils.trajectory import load_rollouts

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

##--- Reward Function ---##
# def reward_function(state):
#     return -(np.sqrt((6 - state[0])**2 + (6 - state[1])**2))

def reward_function(state):
     return state[0]

def cumulative_reward_function(trajectory):
    return np.sum(np.apply_along_axis(reward_function, 2, trajectory))


BASE_FOLDER = 'logs/maze2d-umaze-v1/metrics/release_H128_T64_d0.997_LimitsNormalizer_b1_condFalse/0'

base_diff_traj = load_rollouts(folder=f'{BASE_FOLDER}/rollouts/base_diffuser')
guided_diff_traj = load_rollouts(folder=f'{BASE_FOLDER}/rollouts/guided_diffuser')
bc_traj = load_rollouts(folder=f'{BASE_FOLDER}/rollouts/bc')

n_rollouts = len(base_diff_traj)


base_diff_rewards = np.zeros(n_rollouts)

for rollout,i in zip(base_diff_traj, range(n_rollouts)):
    base_diff_rewards[i] = cumulative_reward_function(rollout)



guided_diff_rewards = np.zeros(n_rollouts)

for rollout,i in zip(guided_diff_traj, range(n_rollouts)):
    guided_diff_rewards[i] = cumulative_reward_function(rollout)



bc_rewards = np.zeros(n_rollouts)

for rollout,i in zip(bc_traj, range(n_rollouts)):
    bc_rewards[i] = cumulative_reward_function(rollout)


# Step 2: Organize the data for plotting
data = [base_diff_rewards, guided_diff_rewards, bc_rewards]
x_labels = ['Base Diffuser', 'IRL Guided Diffuser', 'Behavioral Cloning']

# Step 3: Create the box plot
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=x_labels, patch_artist=True)

# Step 4: Customize the plot
plt.ylabel('Cumulative Rewards')
plt.title('Box Plot of Cumulative Rewards across 20 Trajectories (600 timesteps)')

# Optional: Adding a grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Step 5: Show the plot
plt.savefig('temp.png')
plt.show()


output_string = f'{base_diff_rewards.mean()} +- {base_diff_rewards.std()}'
output_string += f'\n{guided_diff_rewards.mean()} +- {guided_diff_rewards.std()}'
output_string += f'\n{bc_rewards.mean()} +- {bc_rewards.std()}'
print(output_string)

with open('temp.txt', 'w') as txt_file:
    txt_file.write(output_string)