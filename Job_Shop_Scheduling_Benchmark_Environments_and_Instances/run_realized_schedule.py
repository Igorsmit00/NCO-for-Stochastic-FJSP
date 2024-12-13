import argparse
import json

import numpy as np

from solution_methods.helper_functions import load_stochastic_job_shop_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem_instance",
        type=str,
        default="/fjsp/stoch/random/SD3/10x5+mix/10x5+mix_001.fjs",
    )
    parser.add_argument(
        "--existing_solution",
        type=str,
        default="results/cp_sat/or_tools_3600/stoch_10_VaR_95/fjsp/stoch/random/SD3/10x5+mix/10x5+mix_001.fjs/CP_results.json",
    )
    parser.add_argument("--num_eval_realizations", type=int, default=1000)
    parser.add_argument("--num_opt_realizations", type=int, default=10)
    configs = parser.parse_args()
    problem_instance = configs.problem_instance
    existing_solution = configs.existing_solution
    num_eval_realizations = configs.num_eval_realizations
    num_opt_realizations = configs.num_opt_realizations


def parse_existing_solution(filename: str) -> list[dict]:
    """
    Parses an existing solution from a file and returns the schedule.

    Args:
        filename (str): The path to the file containing the existing solution.

    Returns:
        list[dict]: The parsed schedule as a list of dictionaries.
    """
    with open(filename, "r") as f:
        data = json.load(f)
        if type(data) == list:
            schedule = data[0]["Schedule"]
        elif type(data) == dict:
            schedule = data["Schedule"]
    return schedule


def main(
    problem_instance,
    existing_solution,
    num_eval_realizations,
    num_opt_realizations,
    print_bool=True,
):

    det_jobShop, job_shop_envs = load_stochastic_job_shop_env(
        problem_instance, num_realizations=1251, load_deterministic=True
    )

    opt_job_shop_envs = job_shop_envs[:num_opt_realizations]
    eval_job_shop_envs = job_shop_envs[250 : 250 + num_eval_realizations]

    schedule = parse_existing_solution(existing_solution)

    # ((job_id, task_id, machine_id), start_time) sorted by start time
    sorted_tasks = sorted(
        [
            ((job["job"], task["task"], task["machine"]), task["start"])
            for job in schedule
            for task in job["tasks"]
        ],
        key=lambda x: x[1],
    )

    for task in sorted_tasks:
        job_id, operation_id, machine_id = task[0]
        job = det_jobShop.get_job(job_id)
        machine = det_jobShop.get_machine(machine_id)
        operation = job.operations[operation_id]
        processing_time = operation.processing_times[machine_id]
        machine.add_operation_to_schedule(
            operation, processing_time, det_jobShop._sequence_dependent_setup_times
        )
        for shop in opt_job_shop_envs:
            job = shop.get_job(job_id)
            machine = shop.get_machine(machine_id)
            operation = job.operations[operation_id]
            processing_time = operation.processing_times[machine_id]
            machine.add_operation_to_schedule(
                operation, processing_time, shop._sequence_dependent_setup_times
            )
        for shop in eval_job_shop_envs:
            job = shop.get_job(job_id)
            machine = shop.get_machine(machine_id)
            operation = job.operations[operation_id]
            processing_time = operation.processing_times[machine_id]
            machine.add_operation_to_schedule(
                operation, processing_time, shop._sequence_dependent_setup_times
            )
    if print_bool:
        print(f"Makespan deterministic instance = {det_jobShop.makespan}")
        print(
            f"Avg makespan stochastic instances used for solving = {np.mean([shop.makespan for shop in opt_job_shop_envs]) :.2f}"
        )
        print(
            f"95%-VaR makespan stochastic instances used for solving = {np.quantile([shop.makespan for shop in opt_job_shop_envs], 0.95) :.2f}"
        )
        print(
            f"All makespans used for solving = {[shop.makespan for shop in opt_job_shop_envs]}"
        )

        print(
            f"Avg makespan stochastic instances used for evaluation = {np.mean([shop.makespan for shop in eval_job_shop_envs]) :.2f}"
        )
        print(
            f"95%-VaR makespan stochastic instances used for evaluation = {np.quantile([shop.makespan for shop in eval_job_shop_envs], 0.95) :.2f}"
        )
        print(
            f"All makespans used for evaluation = {[shop.makespan for shop in eval_job_shop_envs]}"
        )

    return (
        det_jobShop.makespan,
        np.mean([shop.makespan for shop in opt_job_shop_envs]),
        (
            np.quantile([shop.makespan for shop in opt_job_shop_envs], 0.95)
            if len(opt_job_shop_envs) > 0
            else 0
        ),
        [shop.makespan for shop in opt_job_shop_envs],
        np.mean([shop.makespan for shop in eval_job_shop_envs]),
        np.quantile([shop.makespan for shop in eval_job_shop_envs], 0.95),
        [shop.makespan for shop in eval_job_shop_envs],
    )


if __name__ == "__main__":
    main(
        problem_instance, existing_solution, num_eval_realizations, num_opt_realizations
    )
