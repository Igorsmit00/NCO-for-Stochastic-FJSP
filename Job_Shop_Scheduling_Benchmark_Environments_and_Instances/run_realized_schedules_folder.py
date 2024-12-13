import argparse
import os

import numpy as np
from tqdm import tqdm

from run_realized_schedule import main

parser = argparse.ArgumentParser()
parser.add_argument(
    "--solution_directory",
    default="results/cp_sat/or_tools_3600/stoch_25_VaR_95/fjsp/stoch/random/SD3/10x5+mix",
    type=str,
    help="The directory containing the solution files.",
)
parser.add_argument(
    "--instance_directory",
    default="fjsp/stoch/random/SD3/10x5+mix",
    type=str,
    help="The directory containing the instance files.",
)
parser.add_argument("--num_eval_realizations", type=int, default=1000)
parser.add_argument("--num_opt_realizations", type=int, default=0)
args = parser.parse_args()


def get_all_files(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


solution_instances = [
    f.replace("\\", "/") for f in get_all_files(args.solution_directory)
]
solution_instances = list(map(lambda x: x.split("/")[-2], solution_instances))

all_results = []
for instance_file in tqdm(solution_instances):
    print(instance_file)
    result = main(
        f"/{args.instance_directory}/{instance_file}",
        f"{args.solution_directory}/{instance_file}/CP_results.json",
        args.num_eval_realizations,
        args.num_opt_realizations,
        print_bool=False,
    )
    all_results.append(result)

print(
    f"Makespan deterministic instances = {np.mean([result[0] for result in all_results])}"
)
print(
    f"Avg makespan stochastic instances used to solve = {np.mean([result[1] for result in all_results])}"
)
print(
    f"95%-VaR makespan stochastic instances used to solve = {np.mean([result[2] for result in all_results])}"
)

print(
    f"Avg makespan stochastic instances evaluated = {np.mean([result[4] for result in all_results])}"
)
print(
    f"95%-VaR makespan stochastic instances evaluated = {np.mean([result[5] for result in all_results])}"
)
