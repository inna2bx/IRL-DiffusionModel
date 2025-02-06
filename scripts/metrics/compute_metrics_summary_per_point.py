import os
import numpy as np
import matplotlib.pyplot as plt

import diffuser.utils as utils
from diffuser.utils.trajectory import load_rollouts

FOLDERS =[
    'bc',
    'irl-medium-normal-gamma05/normal',
    'irl-medium-normal/normal',
    'irl-medium-normal-gamma09/normal',

]

METHODS = [
    'Behavioral Cloning',
    'Base Method Gamma 0.5',
    'Base Method Gamma 0.7',
    'Base Method Gamma 0.9',
]

TRAJECTORIES_PER_POINT = 50

POINTS_NAME = [
    'Top-Left',
    'Bottom-Left',
    'Top-Right'
]

##--- Reward Function ---##
def reward_function(state):
    return -np.sqrt((6 - state[0])**2 + (6 - state[1])**2)

def cumulative_reward_function(trajectory):
    return np.sum(np.apply_along_axis(reward_function, 2, trajectory))


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

save_folder = f'logs/{args.dataset}/metrics/_metrics_summary'
if not os.path.exists(save_folder):
    try:  
        os.mkdir(save_folder)  
    except OSError as error:  
        print(error)

rewards = np.zeros((len(METHODS), len(POINTS_NAME), TRAJECTORIES_PER_POINT))
for folder, idx_method, method in zip(FOLDERS, range(len(METHODS)), METHODS):
    print(method)
    rollouts_folder = f'logs/{args.dataset}/metrics/{folder}'
    rollouts = load_rollouts(folder=rollouts_folder)
    
    for point_idx, point_name in enumerate(POINTS_NAME):
        first_idx = point_idx * TRAJECTORIES_PER_POINT
        last_idx = first_idx + TRAJECTORIES_PER_POINT
        point_rollout = rollouts[first_idx:last_idx]

        for idx, rollout in enumerate(point_rollout):
            reward = cumulative_reward_function(rollout)
            #print(f'{method}, {point_name}, {idx} : {reward}')
            rewards[idx_method, point_idx, idx] = reward
        
    

for point_idx, point_name in enumerate(POINTS_NAME):
    reward_per_point = np.squeeze(rewards[:,point_idx,:])

    # Step 3: Create the box plot
    q = round(len(METHODS) * 2)
    plt.figure(figsize=(q+2, 10))
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.boxplot(reward_per_point.T, 
                labels=METHODS, 
                patch_artist=True, 
                showfliers=False)

    # Step 4: Customize the plot
    plt.ylabel('Cumulative Rewards')
    plt.suptitle('Box Plot of Cumulative Rewards', fontsize=24)
    plt.title(f'(50 trajectories starting from the {point_name} corner)', fontsize=16)

    # Optional: Adding a grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Step 5: Show the plot
    plt.savefig(f'{save_folder}/{point_name}_box_plot.png')
    plt.savefig(f'{save_folder}/{point_name}_box_plot.pdf')
    plt.clf()

    metrics_string = ''
    for idx_method, method in enumerate(METHODS) :
        mean = np.round(reward_per_point[idx_method, :].mean(), decimals=2)
        std = np.round(reward_per_point[idx_method, :].std(), decimals=2)
        str_method = method.replace("\n", "")
        metrics_string += f'{str_method}: {mean} +- {std}\n'

    with open(f'{save_folder}/{point_name}_metrics.txt', 'w') as txt_file:
        txt_file.write(metrics_string)