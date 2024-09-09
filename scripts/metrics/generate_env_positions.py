import json
import diffuser.utils as utils
import diffuser.datasets as datasets


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_guided'

args = Parser().parse_args('metrics')

env = datasets.load_environment(args.dataset)

starting_points = []

for i in range(args.n_rollout):
    starting_point = tuple(env.reset()[:2])
    starting_points.append(starting_point)

with open(f'logs/{args.dataset}/metrics/starting_positions.json', 'w') as f:
    json.dump(starting_points, f)

