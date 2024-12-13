import os
import re
import sys
import time

import numpy as np
from tqdm import tqdm

from common_utils import *
from data_utils import pack_data_from_config, pack_data_from_config_realizations
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums as np_env_same
from fjsp_env_same_op_nums_torch_no_feat import (
    FJSPEnvForSameOpNumsNoFeat as torch_env_same,
)
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums as np_env_various
from fjsp_env_various_op_nums_torch_no_feat import (
    FJSPEnvForVariousOpNumsNoFeat as torch_env_various,
)
from params import configs

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def test_heuristic_method(
    data_set,
    stochastic_realizations,
    heuristic,
    seed,
    num_eval_realizations,
    data_source,
):
    """
        test one heuristic method on the given data
    :param data_set:  test data
    :param heuristic: the name of heuristic method
    :param seed: seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    result = []

    for i in tqdm(
        range(len(data_set[0])), file=sys.stdout, desc="progress", colour="blue"
    ):
        n_j = data_set[0][i].shape[0]
        n_op, n_m = data_set[1][i].shape
        if data_source in ["SD1", "BenchData"]:
            env = np_env_various(n_j=n_j, n_m=n_m)
        else:
            env = np_env_same(n_j=n_j, n_m=n_m)

        env.set_initial_data([data_set[0][i]], [data_set[1][i]])

        t1 = time.time()
        actions = []
        while True:
            action = heuristic_select_action(heuristic, env)
            actions.append(action)

            _, _, done = env.step(np.array([action]))

            if done:
                break

        t2 = time.time()
        # Repeat actions for the stochastic realizations
        if data_source in ["SD1", "BenchData"]:
            stoch_env = torch_env_various(n_j=n_j, n_m=n_m)
        else:
            stoch_env = torch_env_same(
                n_j=n_j,
                n_m=n_m,
            )
        stoch_env.set_initial_data(
            stochastic_realizations[i][0][0][0][250 : 250 + num_eval_realizations],
            stochastic_realizations[i][0][0][1][250 : 250 + num_eval_realizations],
            group_size=num_eval_realizations,
        )
        for action in actions:
            stoch_env.step(torch.tensor([action]).repeat(num_eval_realizations))

        avg_makespan = np.mean(stoch_env.current_makespan.numpy())
        VaR_makespan = np.percentile(stoch_env.current_makespan.numpy(), 95)
        result.append([env.current_makespan[0], avg_makespan, VaR_makespan, t2 - t1])

    return np.array(result)


def main():
    """
    test heuristic methods following the config and save the results:
    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    setup_seed(configs.seed_test)
    if not os.path.exists("./test_results"):
        os.makedirs("./test_results")

    test_data = pack_data_from_config(configs.data_source, configs.test_data)
    all_stochastic_realizations = []
    test_data_path = f"./data/{configs.data_source}/{configs.test_data[0]}"
    dist_str_name = (
        f"{configs.variance_dist}/" if configs.variance_dist != "lognormal" else ""
    )
    for instance in sorted(
        sorted(
            list(os.walk(test_data_path))[-1][-1],
            key=lambda s: int(re.findall("\d+", s)[0]),
        ),
        key=lambda s: int(re.findall("\d+", s)[-1]),
    ):
        stochastic_realizations = pack_data_from_config_realizations(
            f"stochastic_realizations/{configs.variance}/{dist_str_name}{configs.data_source}/{configs.test_data[0]}",
            [instance],
            max_realizations=1251,
        )
        all_stochastic_realizations.append(stochastic_realizations)

    if len(configs.test_method) == 0:
        test_method = ["FIFO", "MOR", "SPT", "MWKR"]
    else:
        test_method = configs.test_method

    for data in test_data:
        print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
        print("Test Methods:", test_method)
        print(f"test data name: {configs.data_source},{data[1]}")
        save_direc = f"./test_results/stochastic/{configs.variance}/{dist_str_name}{configs.data_source}/{data[1]}"

        if not os.path.exists(save_direc):
            os.makedirs(save_direc)
        for method in test_method:
            save_path = save_direc + f"/Result_{method}_{data[1]}.npy"

            if (not os.path.exists(save_path)) or configs.cover_heu_flag:
                print(f"Heuristic method : {method}")
                seed = configs.seed_test

                result_5_times = []
                # test 5 times, record average makespan and time.
                for j in range(5):
                    result = test_heuristic_method(
                        data[0],
                        all_stochastic_realizations,
                        method,
                        seed + j,
                        configs.num_eval_realizations,
                        configs.data_source,
                    )
                    result_5_times.append(result)

                    print(
                        f"the {j + 1}th deterministic makespan:", np.mean(result[:, 0])
                    )
                    print(f"the {j + 1}th avg makespan:", np.mean(result[:, 1]))
                    print(f"the {j + 1}th VaR makespan:", np.mean(result[:, 2]))
                result_5_times = np.array(result_5_times)
                save_result = np.mean(result_5_times, axis=0)
                print(f"testing results of {method}:")
                print(f"makespan(sampling) deterministic: ", save_result[:, 0].mean())
                print(f"makespan(sampling) avg: ", save_result[:, 1].mean())
                print(f"makespan(sampling) VaR: ", save_result[:, 2].mean())
                print(f"time: ", save_result[:, -1].mean())
                np.save(save_path, save_result)


if __name__ == "__main__":
    main()
