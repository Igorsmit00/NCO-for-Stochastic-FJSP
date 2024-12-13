import os
import re
import time

from tqdm import tqdm

from common_utils import *
from common_utils import setup_seed
from data_utils import pack_data_from_config, pack_data_from_config_realizations
from fjsp_env_same_op_nums_torch import FJSPEnvForSameOpNums
from fjsp_env_same_op_nums_torch_no_feat import FJSPEnvForSameOpNumsNoFeat
from fjsp_env_various_op_nums_torch import FJSPEnvForVariousOpNums
from fjsp_env_various_op_nums_torch_no_feat import FJSPEnvForVariousOpNumsNoFeat
from model.main_model import *
from model.PPO import PPO_initialize
from params import configs

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

ppo = PPO_initialize()
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def test_offline_parallel_greedy_strategy_stoch(
    data_set,
    stoch_data_set,
    model_path,
    seed,
    num_input_realizations,
    num_eval_realizations,
    data_source,
):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """

    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape

    for i in tqdm(
        range(len(data_set[0])), file=sys.stdout, desc="progress", colour="blue"
    ):
        if data_source in ["SD1", "BenchData"]:
            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            eval_env = FJSPEnvForVariousOpNumsNoFeat(n_j=n_j, n_m=n_m)
        else:
            env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
            eval_env = FJSPEnvForSameOpNumsNoFeat(n_j=n_j, n_m=n_m)
        state = env.set_initial_data(
            [data_set[0][i]] + stoch_data_set[i][0][0][0][:num_input_realizations],
            [data_set[1][i]] + stoch_data_set[i][0][0][1][:num_input_realizations],
            group_size=num_input_realizations + 1,
        )
        eval_env.set_initial_data(
            stoch_data_set[i][0][0][0][250 : 250 + num_eval_realizations],
            stoch_data_set[i][0][0][1][250 : 250 + num_eval_realizations],
            group_size=num_eval_realizations,
        )
        actions = []
        if isinstance(ppo.policy, (SPM_DAN)):
            t1 = time.time()
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor.unsqueeze(0),  # [1, N, 8]
                        op_mask=state.op_mask_tensor.unsqueeze(0),  # [1, N, N]
                        candidate=state.candidate_tensor.unsqueeze(0),  # [1, J]
                        fea_m=state.fea_m_tensor.unsqueeze(0),  # [1, M, 6]
                        mch_mask=state.mch_mask_tensor.unsqueeze(0),  # [1, M, M]
                        comp_idx=state.comp_idx_tensor.unsqueeze(0),  # [1, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor.unsqueeze(0),
                        fea_pairs=state.fea_pairs_tensor.unsqueeze(0),
                        num_samples=num_input_realizations,
                    )

                action = greedy_select_action(pi)
                actions.append(action)
                state, reward, done = env.step(
                    actions=action.repeat_interleave(num_input_realizations + 1)
                )

                if done.all():
                    break
            t2 = time.time()

        else:
            t1 = time.time()
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor[[0]],  # [1, N, 8]
                        op_mask=state.op_mask_tensor[[0]],  # [1, N, N]
                        candidate=state.candidate_tensor[[0]],  # [1, J]
                        fea_m=state.fea_m_tensor[[0]],  # [1, M, 6]
                        mch_mask=state.mch_mask_tensor[[0]],  # [1, M, M]
                        comp_idx=state.comp_idx_tensor[[0]],  # [1, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[[0]],
                        fea_pairs=state.fea_pairs_tensor[[0]],
                    )  # [1, J, M]

                action = greedy_select_action(pi)
                actions.append(action)
                state, reward, done = env.step(
                    actions=action.repeat_interleave(num_input_realizations + 1)
                )
                if done.all():
                    break
            t2 = time.time()

        for action in actions:
            eval_env.step(actions=action.repeat_interleave(num_eval_realizations))

        VaR_makespan = np.quantile(
            env.current_makespan.cpu().numpy().reshape(1, -1)[:, 1:], q=0.95, axis=1
        ).item()
        mean_makespan = np.mean(
            env.current_makespan.cpu().numpy().reshape(1, -1)[:, 1:], axis=1
        ).item()
        eval_VaR_makespan = np.quantile(
            eval_env.current_makespan.cpu().numpy().reshape(1, -1), q=0.95, axis=1
        ).item()
        eval_mean_makespan = np.mean(
            eval_env.current_makespan.cpu().numpy().reshape(1, -1), axis=1
        ).item()

        test_result_list.append(
            [
                env.current_makespan[0].cpu().numpy(),
                mean_makespan,
                VaR_makespan,
                eval_mean_makespan,
                eval_VaR_makespan,
                t2 - t1,
            ]
        )

    return np.array(test_result_list)


def test_offline_parallel_sampling_strategy_stoch(
    data_set,
    stoch_data_set,
    model_path,
    sample_times,
    seed,
    num_input_realizations,
    num_eval_realizations,
    data_source,
):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """

    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    indices = [(num_input_realizations + 1) * i for i in range(sample_times)]
    for i in tqdm(
        range(len(data_set[0])), file=sys.stdout, desc="progress", colour="blue"
    ):
        if data_source in ["SD1", "BenchData"]:
            env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)
            eval_env = FJSPEnvForVariousOpNumsNoFeat(n_j=n_j, n_m=n_m)
        else:
            env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
            eval_env = FJSPEnvForSameOpNumsNoFeat(n_j=n_j, n_m=n_m)
        state = env.set_initial_data(
            ([data_set[0][i]] + stoch_data_set[i][0][0][0][:num_input_realizations])
            * sample_times,
            ([data_set[1][i]] + stoch_data_set[i][0][0][1][:num_input_realizations])
            * sample_times,
            group_size=num_input_realizations + 1,
        )
        eval_env.set_initial_data(
            stoch_data_set[i][0][0][0][
                num_input_realizations : num_input_realizations + num_eval_realizations
            ]
            * sample_times,
            stoch_data_set[i][0][0][1][
                num_input_realizations : num_input_realizations + num_eval_realizations
            ]
            * sample_times,
            group_size=num_eval_realizations,
        )
        # eval_env.to_cpu()
        # torch.cuda.empty_cache()
        actions = []
        if isinstance(ppo.policy, (SPM_DAN)):
            t1 = time.time()
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.fea_j_tensor.shape[1],
                            state.fea_j_tensor.shape[2],
                        ),  # [sz_b, num_samples, N, 8]
                        op_mask=state.op_mask_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.op_mask_tensor.shape[1],
                            state.op_mask_tensor.shape[2],
                        ),  # [sz_b, num_samples, N, N]
                        candidate=state.candidate_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.candidate_tensor.shape[1],
                        ),  # [sz_b, J]
                        fea_m=state.fea_m_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.fea_m_tensor.shape[1],
                            state.fea_m_tensor.shape[2],
                        ),  # [sz_b, M, 6]
                        mch_mask=state.mch_mask_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.mch_mask_tensor.shape[1],
                            state.mch_mask_tensor.shape[2],
                        ),  # [sz_b, M, M]
                        comp_idx=state.comp_idx_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.comp_idx_tensor.shape[1],
                            state.comp_idx_tensor.shape[2],
                            state.comp_idx_tensor.shape[3],
                        ),  # [sz_b, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.dynamic_pair_mask_tensor.shape[1],
                            state.dynamic_pair_mask_tensor.shape[2],
                        ),  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor.view(
                            -1,
                            num_input_realizations + 1,
                            state.fea_pairs_tensor.shape[1],
                            state.fea_pairs_tensor.shape[2],
                            state.fea_pairs_tensor.shape[3],
                        ),  # [sz_b, J, M]
                        num_samples=num_input_realizations,
                    )

                action, _ = sample_action(pi)
                actions.append(action)
                state, reward, done = env.step(
                    actions=action.repeat_interleave(num_input_realizations + 1)
                )
                if done.all():
                    env.to_cpu()
                    break
            t2 = time.time()

        else:
            t1 = time.time()
            while True:
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor[indices],  # [1, N, 8]
                        op_mask=state.op_mask_tensor[indices],  # [1, N, N]
                        candidate=state.candidate_tensor[indices],  # [1, J]
                        fea_m=state.fea_m_tensor[indices],  # [1, M, 6]
                        mch_mask=state.mch_mask_tensor[indices],  # [1, M, M]
                        comp_idx=state.comp_idx_tensor[indices],  # [1, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[indices],
                        fea_pairs=state.fea_pairs_tensor[indices],
                    )  # [1, J, M]

                action, _ = sample_action(pi)
                actions.append(action)
                state, reward, done = env.step(
                    actions=action.repeat_interleave(num_input_realizations + 1)
                )
                if done.all():
                    env.to_cpu()
                    break
            t2 = time.time()
        for action in actions:
            eval_env.step(actions=action.repeat_interleave(num_eval_realizations))
            # Select the best mean and VaR makespans
        VaR_makespan = np.min(
            np.quantile(
                env.current_makespan.cpu().numpy().reshape(sample_times, -1)[:, 1:],
                q=0.95,
                axis=1,
            )
        )
        mean_makespan = np.min(
            np.mean(
                env.current_makespan.cpu().numpy().reshape(sample_times, -1)[:, 1:],
                axis=1,
            )
        )
        eval_VaR_makespan = np.min(
            np.quantile(
                eval_env.current_makespan.cpu().numpy().reshape(sample_times, -1),
                q=0.95,
                axis=1,
            )
        )
        eval_mean_makespan = np.min(
            np.mean(
                eval_env.current_makespan.cpu().numpy().reshape(sample_times, -1),
                axis=1,
            )
        )

        test_result_list.append(
            [
                env.current_makespan[0].cpu().numpy(),
                mean_makespan,
                VaR_makespan,
                eval_mean_makespan,
                eval_VaR_makespan,
                t2 - t1,
            ]
        )

        torch.cuda.empty_cache()

    return np.array(test_result_list)


def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    flag_sample = config.test_mode
    if not os.path.exists("./test_results/stochastic"):
        os.makedirs("./test_results/stochastic")

    # collect the path of test models
    test_model = []
    dist_str = (
        f"{config.variance_dist}/"
        if ((configs.variance_dist != "lognormal") and (not config.deter_model))
        else ""
    )
    for model_name in config.test_model:
        test_model.append(
            (
                f'./trained_network/{"random/" if not config.deter_model else ""}{dist_str}{config.model_source}/{model_name}.pth',
                model_name,
            )
        )

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)
    test_data_path = f"./data/{config.data_source}/{config.test_data[0]}"
    stochastic_test_data = []

    dist_str_name = (
        f"{config.variance_dist}/" if configs.variance_dist != "lognormal" else ""
    )
    for instance in sorted(
        sorted(
            list(os.walk(test_data_path))[-1][-1],
            key=lambda s: int(re.findall("\d+", s)[0]),
        ),
        key=lambda s: int(re.findall("\d+", s)[-1]),
    ):
        stochastic_realizations = pack_data_from_config_realizations(
            f"stochastic_realizations/{config.variance}/{dist_str_name}{config.data_source}/{config.test_data[0]}",
            [instance],
            max_realizations=1251,
        )
        stochastic_test_data.append(stochastic_realizations)
    if flag_sample:
        model_prefix = "DANIELS"
    else:
        model_prefix = "DANIELG"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f"./test_results/stochastic/{config.variance}/{dist_str}{config.data_source}/{data[1]}"
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            eval_num_str = (
                f"_{config.num_input_realizations}"
                if f"_{config.num_input_realizations}_" not in model[1]
                else ""
            )
            save_path = (
                save_direc
                + f"/Result_{model_prefix}+{model[1]}_{data[1]}_{eval_num_str}.npy"
            )
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(5):
                        result = test_offline_parallel_greedy_strategy_stoch(
                            data[0],
                            stochastic_test_data,
                            model[0],
                            config.seed_test,
                            configs.num_input_realizations,
                            configs.num_eval_realizations,
                            data_source=config.data_source,
                        )
                        result_5_times.append(result)

                        print(
                            f"the {j + 1}th deterministic makespan:",
                            np.mean(result[:, 0]),
                        )
                        print(f"the {j + 1}th avg makespan:", np.mean(result[:, 1]))
                        print(f"the {j + 1}th VaR makespan:", np.mean(result[:, 2]))
                    result_5_times = np.array(result_5_times)

                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print(f"makespan(greedy) deterministic: ", save_result[:, 0].mean())
                    print(f"makespan(greedy) avg: ", save_result[:, 1].mean())
                    print(f"makespan(greedy) VaR: ", save_result[:, 2].mean())
                    print(f"makespan(greedy) avg(eval): ", save_result[:, 3].mean())
                    print(f"makespan(greedy) VaR(eval): ", save_result[:, 4].mean())
                    print(f"time: ", save_result[:, -1].mean())

                else:
                    # Sample mode, test once.
                    print("Test mode: Sample")
                    save_result = test_offline_parallel_sampling_strategy_stoch(
                        data[0],
                        stochastic_test_data,
                        model[0],
                        config.sample_times,
                        config.seed_test,
                        configs.num_input_realizations,
                        configs.num_eval_realizations,
                        data_source=config.data_source,
                    )
                    print("testing results:")
                    print(
                        f"makespan(sampling) deterministic: ", save_result[:, 0].mean()
                    )
                    print(f"makespan(sampling) avg: ", save_result[:, 1].mean())
                    print(f"makespan(sampling) VaR: ", save_result[:, 2].mean())
                    print(f"makespan(sampling) avg(eval): ", save_result[:, 3].mean())
                    print(f"makespan(sampling) VaR(eval): ", save_result[:, 4].mean())
                    print(f"time: ", save_result[:, -1].mean())
                np.save(save_path, save_result)


if __name__ == "__main__":
    main(configs, False)
