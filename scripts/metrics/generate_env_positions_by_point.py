import json
import diffuser.utils as utils
import diffuser.datasets as datasets
import numpy as np


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

#env = datasets.load_environment(args.dataset)

generating_points=[
    (1,1),
    (6,1),
    (1.3, 5.3)
]

NUMBER_OF_POINTS_PER_SEED = 50

starting_points = []

# for i in range(args.n_rollout):
#     starting_point = tuple(env.reset()[:2])
#     starting_points.append(starting_point)

for point in generating_points:
    for _ in range (NUMBER_OF_POINTS_PER_SEED):
        new_point = tuple(point + np.random.uniform(-0.3, 0.3, size=(2,)))
        starting_points.append(new_point)

with open(f'logs/{args.dataset}/metrics/starting_positions.json', 'w') as f:
    json.dump(starting_points, f)

