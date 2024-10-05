import numpy as np
import matplotlib.pyplot as plt

import diffuser.utils as utils
from diffuser.utils.trajectory import load_rollouts


##--- Parameters --##


FOLDERS =[
    'bc',
    'irl-umaze-base/normal',
    'irl-tr-reward/normal',
    'irl-no-inv/normal',

]

METHODS = [
    'Behavioral Cloning',
    'Base Diffuser\nGuided by IRL',
    'Base Diffuser\nGuided by True Reward',
    'Base Diffuser\nNo Guiding',

]

##--- Reward Function ---##
# def reward_function(state):
#     return -((6 - state[0])**2 + (6 - state[1])**2)

def reward_function(state):
    return state[0]

def cumulative_reward_function(trajectory):
    return np.sum(np.apply_along_axis(reward_function, 2, trajectory))


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

save_folder = f'logs/{args.dataset}/metrics'

rewards_per_method = {}
for folder, method in zip(FOLDERS, METHODS):
    print(method)
    rollouts_folder = f'logs/{args.dataset}/metrics/{folder}'
    rollouts = load_rollouts(folder=rollouts_folder)
    
    rewards = np.zeros(len(rollouts))
    for idx, rollout in enumerate(rollouts):
        rewards[idx] = cumulative_reward_function(rollout)
    
    rewards_per_method[method] = rewards
    
## Data Plot

# Step 3: Create the box plot
plt.figure(figsize=(8, 6))
plt.boxplot(rewards_per_method.values(), 
            labels=rewards_per_method.keys(), 
            patch_artist=True)

# Step 4: Customize the plot
plt.ylabel('Cumulative Rewards')
plt.title('Box Plot of Cumulative Rewards across 100 Trajectories (300 timesteps)')

# Optional: Adding a grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Step 5: Show the plot
plt.savefig(f'{save_folder}/box_plot.png')
plt.savefig(f'{save_folder}/box_plot.pdf')

metrics_string = ''
for method, rewards in rewards_per_method.items() :
    mean = np.round(rewards.mean(), decimals=2)
    std = np.round(rewards.std(), decimals=2)
    str_method = method.replace("\n", "")
    metrics_string += f'{str_method}: {mean} +- {std}\n'

with open(f'{save_folder}/metrics.txt', 'w') as txt_file:
    txt_file.write(metrics_string)