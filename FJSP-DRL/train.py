import os
import random
import re
import sys
import time
from copy import deepcopy

from tqdm import tqdm

from common_utils import *
from common_utils import setup_seed, strToSuffix
from data_utils import (
    CaseGenerator,
    SD2_instance_generator,
    SD3_instance_generator,
    load_data_from_files,
    pack_data_from_config_realizations,
)
from fjsp_env_same_op_nums_torch import EnvState, FJSPEnvForSameOpNums
from fjsp_env_same_op_nums_torch_no_feat import FJSPEnvForSameOpNumsNoFeat
from fjsp_env_various_op_nums_torch import FJSPEnvForVariousOpNums
from fjsp_env_various_op_nums_torch_no_feat import FJSPEnvForVariousOpNumsNoFeat
from generate_random_instances import (
    create_stochastic_realizations,
    create_stochastic_realizations_random_beta,
    create_stochastic_realizations_random_beta_log_gamma_mix,
    create_stochastic_realizations_random_beta_log_mix,
    create_stochastic_realizations_random_variances,
)
from model.PPO import Memory, PPO_initialize
from params import configs

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)

device = torch.device(configs.device)
print(torch.cuda.get_device_name())
print(configs)


class Trainer:
    def __init__(self, config):

        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs
        self.train_step_type = config.train_step_type
        self.variance = config.variance
        self.variance_dist = config.variance_dist
        self.stoch_obj = config.stoch_obj
        self.VaR_alpha = config.VaR_alpha
        self.num_eval_realizations = config.num_eval_realizations
        self.num_input_realizations = config.num_input_realizations

        dist_str = f"{self.variance_dist}/" if self.variance_dist != "lognormal" else ""
        if not os.path.exists(
            f"./trained_network/{self.variance}/{dist_str}{self.data_source}"
        ):
            os.makedirs(
                f"./trained_network/{self.variance}/{dist_str}{self.data_source}"
            )
        if not os.path.exists(
            f"./train_log/{self.variance}/{dist_str}{self.data_source}/"
        ):
            os.makedirs(f"./train_log/{self.variance}/{dist_str}{self.data_source}/")

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        if self.data_source == "SD1":
            self.data_name = f"{self.n_j}x{self.n_m}"
        elif self.data_source == "SD2":
            self.data_name = f"{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}"
        elif self.data_source == "SD3":
            self.data_name = f"{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}"
        elif self.data_source == "BenchData":
            self.data_name = config.test_data[0]

        self.vali_data_path = (
            f"./data/data_train_vali/{self.data_source}/{self.data_name}"
        )
        if self.data_source == "BenchData":
            self.vali_data_path = f"./data/{self.data_source}/{self.data_name}"
        self.test_data_path = f"./data/{self.data_source}/{self.data_name}"
        network_arch = "attn_" if config.SAA_attention else ""
        obj_string = (
            f"{self.stoch_obj}"
            + f"{int(self.VaR_alpha * 100) if self.stoch_obj=='VaR' else ''}"
        )
        input_rea_str = (
            f"{self.num_input_realizations}_"
            if config.SAA_attention and self.num_input_realizations != 100
            else ""
        )

        dist_str_name = (
            f"{self.variance_dist}" if self.variance_dist != "lognormal" else ""
        )
        self.model_name = f"{self.variance[:3]}{dist_str_name}{self.data_name}{strToSuffix(config.model_suffix)}_stoch_{network_arch}{input_rea_str}{self.train_step_type}_{obj_string}_{config.layer_fea_output_dim[0]}_{config.layer_fea_output_dim[1]}_{config.hidden_dim_actor}_{config.hidden_dim_critic}"
        self.SAA_attention = config.SAA_attention

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        if self.data_source == "SD1":
            self.env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
            self.reward_env = FJSPEnvForVariousOpNumsNoFeat(self.n_j, self.n_m)
        elif self.data_source in ["SD2", "SD3"]:
            self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
            self.reward_env = FJSPEnvForSameOpNumsNoFeat(self.n_j, self.n_m)
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)
        all_stochastic_realizations = []
        for instance in sorted(
            list(os.walk(self.vali_data_path))[-1][-1],
            key=lambda s: int(re.findall("\d+", s)[-1]),
        ):
            vali_stochastic_realizations = pack_data_from_config_realizations(
                f"data_train_vali/stochastic_realizations/{self.variance}/{dist_str}{self.data_source}/{self.data_name}",
                [instance],
                max_realizations=1251,
            )
            all_stochastic_realizations.append(vali_stochastic_realizations)

        if self.data_source == "SD1":
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
            self.vali_reward_env = FJSPEnvForVariousOpNumsNoFeat(self.n_j, self.n_m)
        elif self.data_source in ["SD2", "SD3"]:
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
            self.vali_reward_env = FJSPEnvForSameOpNumsNoFeat(self.n_j, self.n_m)
        new_dataset_op_pt = []
        new_dataset_job_length = []
        for i in range(len(vali_data[1])):
            new_dataset_op_pt.append(vali_data[1][i])
            new_dataset_op_pt.extend(
                all_stochastic_realizations[i][0][0][1][: self.num_input_realizations]
            )  # [:10])
            new_dataset_job_length.append(vali_data[0][i])
            new_dataset_job_length.extend(
                all_stochastic_realizations[i][0][0][0][: self.num_input_realizations]
            )  # [:10])
        self.vali_env.set_initial_data(
            new_dataset_job_length,
            new_dataset_op_pt,
            group_size=self.num_input_realizations + 1,
        )

        self.vali_reward_env.set_initial_data(
            [
                element
                for i in range(len(vali_data[1]))
                for element in all_stochastic_realizations[i][0][0][0][
                    250 : 250 + self.num_eval_realizations
                ]
            ],
            [
                element
                for i in range(len(vali_data[1]))
                for element in all_stochastic_realizations[i][0][0][1][
                    250 : 250 + self.num_eval_realizations
                ]
            ],
            group_size=self.num_eval_realizations,
        )
        self.vali_reward_env.to_cpu()

        self.vali_env.to_cpu()
        torch.cuda.empty_cache()

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
        train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float("inf")

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(
            range(self.max_updates), file=sys.stdout, desc="progress", colour="blue"
        ):
            self.env.to_gpu()
            self.reward_env.to_gpu()
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                stochastic_realizations = []
                for instance in dataset_op_pt:
                    if self.variance == "fixed":
                        stochastic_realizations.append(
                            create_stochastic_realizations(
                                instance,
                                self.num_input_realizations
                                + self.num_eval_realizations,
                            )
                        )
                    elif self.variance == "random":
                        if self.variance_dist == "beta":
                            stochastic_realizations.append(
                                create_stochastic_realizations_random_beta(
                                    instance,
                                    self.num_input_realizations
                                    + self.num_eval_realizations,
                                )
                            )
                        elif self.variance_dist == "be_lo":
                            stochastic_realizations.append(
                                create_stochastic_realizations_random_beta_log_mix(
                                    instance,
                                    self.num_input_realizations
                                    + self.num_eval_realizations,
                                )
                            )
                        elif self.variance_dist == "be_lo_ga":
                            stochastic_realizations.append(
                                create_stochastic_realizations_random_beta_log_gamma_mix(
                                    instance,
                                    self.num_input_realizations
                                    + self.num_eval_realizations,
                                )
                            )
                        else:
                            stochastic_realizations.append(
                                create_stochastic_realizations_random_variances(
                                    instance,
                                    self.num_input_realizations
                                    + self.num_eval_realizations,
                                )
                            )
                if self.train_step_type == "random":
                    state = self.env.set_initial_data(
                        dataset_job_length, dataset_op_pt, stochastic_realizations
                    )
                elif self.train_step_type == "SAA":
                    new_dataset_op_pt = []
                    new_dataset_job_length = []
                    for i in range(len(dataset_op_pt)):
                        new_dataset_op_pt.append(dataset_op_pt[i])
                        new_dataset_op_pt.extend(
                            stochastic_realizations[i][: self.num_input_realizations]
                        )
                        new_dataset_job_length.extend(
                            [dataset_job_length[i]]
                            * (
                                len(
                                    stochastic_realizations[i][
                                        : self.num_input_realizations
                                    ]
                                )
                                + 1
                            )
                        )
                    state = self.env.set_initial_data(
                        new_dataset_job_length,
                        new_dataset_op_pt,
                        group_size=self.num_input_realizations + 1,
                    )
                    self.reward_env.set_initial_data(
                        [
                            element
                            for i in range(len(dataset_op_pt))
                            for element in [dataset_job_length[i]]
                            * self.num_eval_realizations
                        ],
                        [
                            element
                            for i in range(len(dataset_op_pt))
                            for element in stochastic_realizations[i][
                                self.num_input_realizations : self.num_input_realizations
                                + self.num_eval_realizations
                            ]
                        ],
                        group_size=self.num_eval_realizations,
                    )
            else:
                state = self.env.reset()
                self.reward_env.reset()

            if self.train_step_type == "random":
                ep_rewards = -deepcopy(self.env.init_quality)
            elif self.train_step_type == "SAA":
                # Mean
                if self.stoch_obj == "mean":
                    ep_rewards = -np.mean(
                        deepcopy(self.reward_env.init_quality.cpu().numpy()).reshape(
                            (self.num_envs, -1)
                        ),
                        axis=1,
                    )
                # VaR
                elif self.stoch_obj == "VaR":
                    prev_quantile_endtime_bound = np.quantile(
                        self.reward_env.max_endTime.cpu()
                        .numpy()
                        .reshape((self.num_envs, -1)),
                        q=self.VaR_alpha,
                        axis=1,
                    )
                    ep_rewards = -prev_quantile_endtime_bound

            while True:

                # state store
                if self.train_step_type == "random":
                    self.memory.push(state)
                if self.train_step_type == "SAA":
                    indices = [
                        i * (self.num_input_realizations + 1)
                        for i in range(self.num_envs)
                    ]
                    if self.SAA_attention:
                        mem_state = EnvState(
                            fea_j_tensor=state.fea_j_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_j_tensor.shape[1],
                                state.fea_j_tensor.shape[2],
                            ),
                            op_mask_tensor=state.op_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.op_mask_tensor.shape[1],
                                state.op_mask_tensor.shape[2],
                            ),
                            candidate_tensor=state.candidate_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.candidate_tensor.shape[1],
                            ),
                            fea_m_tensor=state.fea_m_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_m_tensor.shape[1],
                                state.fea_m_tensor.shape[2],
                            ),
                            mch_mask_tensor=state.mch_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.mch_mask_tensor.shape[1],
                                state.mch_mask_tensor.shape[2],
                            ),
                            comp_idx_tensor=state.comp_idx_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.comp_idx_tensor.shape[1],
                                state.comp_idx_tensor.shape[2],
                                state.comp_idx_tensor.shape[3],
                            ),
                            dynamic_pair_mask_tensor=state.dynamic_pair_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.dynamic_pair_mask_tensor.shape[1],
                                state.dynamic_pair_mask_tensor.shape[2],
                            ),
                            fea_pairs_tensor=state.fea_pairs_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_pairs_tensor.shape[1],
                                state.fea_pairs_tensor.shape[2],
                                state.fea_pairs_tensor.shape[3],
                            ),
                        )
                    else:
                        mem_state = EnvState(
                            fea_j_tensor=state.fea_j_tensor[indices],
                            op_mask_tensor=state.op_mask_tensor[indices],
                            candidate_tensor=state.candidate_tensor[indices],
                            fea_m_tensor=state.fea_m_tensor[indices],
                            mch_mask_tensor=state.mch_mask_tensor[indices],
                            comp_idx_tensor=state.comp_idx_tensor[indices],
                            dynamic_pair_mask_tensor=state.dynamic_pair_mask_tensor[
                                indices
                            ],
                            fea_pairs_tensor=state.fea_pairs_tensor[indices],
                        )
                    self.memory.push(mem_state)
                    self.memory.to_cpu()
                with torch.no_grad():
                    if self.train_step_type == "random":
                        pi_envs, vals_envs = self.ppo.policy_old(
                            fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                            op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                            candidate=state.candidate_tensor,  # [sz_b, J]
                            fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                            mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                            comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                            fea_pairs=state.fea_pairs_tensor,
                        )  # [sz_b, J, M]
                    elif self.train_step_type == "SAA":
                        if self.SAA_attention:
                            pi_envs, vals_envs = self.ppo.policy_old(
                                fea_j=state.fea_j_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.fea_j_tensor.shape[1],
                                    state.fea_j_tensor.shape[2],
                                ),  # [sz_b, num_samples, N, 8]
                                op_mask=state.op_mask_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.op_mask_tensor.shape[1],
                                    state.op_mask_tensor.shape[2],
                                ),  # [sz_b, num_samples, N, N]
                                candidate=state.candidate_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.candidate_tensor.shape[1],
                                ),  # [sz_b, J]
                                fea_m=state.fea_m_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.fea_m_tensor.shape[1],
                                    state.fea_m_tensor.shape[2],
                                ),  # [sz_b, M, 6]
                                mch_mask=state.mch_mask_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.mch_mask_tensor.shape[1],
                                    state.mch_mask_tensor.shape[2],
                                ),  # [sz_b, M, M]
                                comp_idx=state.comp_idx_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.comp_idx_tensor.shape[1],
                                    state.comp_idx_tensor.shape[2],
                                    state.comp_idx_tensor.shape[3],
                                ),  # [sz_b, M, M, J]
                                dynamic_pair_mask=state.dynamic_pair_mask_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.dynamic_pair_mask_tensor.shape[1],
                                    state.dynamic_pair_mask_tensor.shape[2],
                                ),  # [sz_b, J, M]
                                fea_pairs=state.fea_pairs_tensor.view(
                                    -1,
                                    self.num_input_realizations + 1,
                                    state.fea_pairs_tensor.shape[1],
                                    state.fea_pairs_tensor.shape[2],
                                    state.fea_pairs_tensor.shape[3],
                                ),  # [sz_b, J, M]
                                num_samples=self.num_input_realizations,
                            )
                        else:
                            pi_envs, vals_envs = self.ppo.policy_old(
                                fea_j=state.fea_j_tensor[indices],  # [sz_b, N, 8]
                                op_mask=state.op_mask_tensor[indices],  # [sz_b, N, N]
                                candidate=state.candidate_tensor[indices],  # [sz_b, J]
                                fea_m=state.fea_m_tensor[indices],  # [sz_b, M, 6]
                                mch_mask=state.mch_mask_tensor[indices],  # [sz_b, M, M]
                                comp_idx=state.comp_idx_tensor[
                                    indices
                                ],  # [sz_b, M, M, J]
                                dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                                    indices
                                ],  # [sz_b, J, M]
                                fea_pairs=state.fea_pairs_tensor[indices],
                            )  # [sz_b, J, M]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                if self.train_step_type == "random":
                    state, reward, done = self.env.step(
                        actions=action_envs.cpu().numpy()
                    )
                elif self.train_step_type == "SAA":
                    if self.SAA_attention:
                        state, reward, done = self.env.step(
                            action_envs.repeat_interleave(
                                self.num_input_realizations + 1
                            )
                        )
                    else:
                        state, reward, done = self.env.step(
                            actions=action_envs.repeat_interleave(
                                self.num_input_realizations + 1
                            )
                        )
                    _, reward, _ = self.reward_env.step(
                        action_envs.repeat_interleave(self.num_eval_realizations)
                    )
                if self.train_step_type == "random":
                    ep_rewards += reward
                elif self.train_step_type == "SAA":
                    # Reward for mean:
                    if self.stoch_obj == "mean":
                        reward = np.mean(reward.reshape((self.num_envs, -1)), axis=1)

                    # Reward for quantile 95%:
                    elif self.stoch_obj == "VaR":
                        quantile_endtime_bound = np.quantile(
                            self.reward_env.max_endTime.cpu()
                            .numpy()
                            .reshape((self.num_envs, -1)),
                            q=self.VaR_alpha,
                            axis=1,
                        )
                        reward = prev_quantile_endtime_bound - quantile_endtime_bound
                        prev_quantile_endtime_bound = quantile_endtime_bound
                    ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                if self.train_step_type == "random":
                    self.memory.done_seq.append(torch.from_numpy(done).to(device))
                    self.memory.reward_seq.append(reward)
                    self.memory.action_seq.append(action_envs)
                    self.memory.log_probs.append(action_logprob_envs)
                    self.memory.val_seq.append(vals_envs.squeeze(1))
                elif self.train_step_type == "SAA":
                    self.memory.done_seq.append(
                        torch.from_numpy(done[indices]).to(device)
                    )
                    self.memory.reward_seq.append(reward)
                    self.memory.action_seq.append(action_envs)
                    self.memory.log_probs.append(action_logprob_envs)
                    self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    self.env.to_cpu()
                    self.memory.to_cpu()
                    self.reward_env.to_cpu()
                    torch.cuda.empty_cache()
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            if self.stoch_obj == "mean":
                mean_obj_all_env = np.mean(
                    np.mean(
                        self.reward_env.current_makespan.cpu()
                        .numpy()
                        .reshape((self.num_envs, -1)),
                        axis=1,
                    )
                )
            elif self.stoch_obj == "VaR":
                mean_obj_all_env = np.mean(
                    np.quantile(
                        self.reward_env.current_makespan.cpu()
                        .numpy()
                        .reshape((self.num_envs, -1)),
                        q=self.VaR_alpha,
                        axis=1,
                    )
                )

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(
                    f"The validation quality is: {vali_result} (best : {self.record})"
                )
            ep_et = time.time()

            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                "Episode {}\t reward: {:.2f}\t objective: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}".format(
                    i_update + 1,
                    mean_rewards_all_env,
                    mean_obj_all_env,
                    loss,
                    ep_et - ep_st,
                )
            )

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
        save reward data & validation makespan data (during training) and the entire training time
        """
        dist_str = f"{self.variance_dist}/" if self.variance_dist != "lognormal" else ""
        file_writing_obj = open(
            f"./train_log/{self.variance}/{dist_str}{self.data_source}/"
            + "reward_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(
            f"./train_log/{self.variance}/{dist_str}{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f"./train_time.txt", "a")
        file_writing_obj3.write(
            f"model path: ./DANIEL_FJSP/trained_network/{self.variance}/{dist_str}{self.data_source}/{self.model_name}\t\ttraining time: "
            f"{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n"
        )

    def save_validation_log(self):
        """
        save the results of validation
        """
        dist_str = f"{self.variance_dist}/" if self.variance_dist != "lognormal" else ""
        file_writing_obj1 = open(
            f"./train_log/{self.variance}/{dist_str}{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
        """
        prepare_JobLength = [
            random.randint(self.op_per_job_min, self.op_per_job_max)
            for _ in range(self.n_j)
        ]
        dataset_JobLength = []
        dataset_OpPT = []
        for i in range(self.num_envs):
            if self.data_source == "SD1":
                case = CaseGenerator(
                    self.n_j,
                    self.n_m,
                    self.op_per_job_min,
                    self.op_per_job_max,
                    nums_ope=prepare_JobLength,
                    path="./test",
                    flag_doc=False,
                )
                JobLength, OpPT, _ = case.get_case(i)

            elif self.data_source == "SD2":
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            elif self.data_source == "SD3":
                JobLength, OpPT, _ = SD3_instance_generator(config=self.config)
            if self.data_source == "BenchData":
                dataset_JobLength, dataset_OpPT = load_data_from_files(
                    f"./data/{self.data_source}/{self.data_name}"
                )
                return dataset_JobLength, dataset_OpPT
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        self.vali_env.to_gpu()
        self.vali_reward_env.to_gpu()
        state = self.vali_env.reset()
        self.vali_reward_env.reset()

        if self.train_step_type == "SAA":
            indices = [i * (self.num_input_realizations + 1) for i in range(100)]
        while True:
            with torch.no_grad():
                if self.train_step_type == "random":
                    pi, _ = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                        op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                        candidate=state.candidate_tensor,  # [sz_b, J]
                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor,
                    )  # [sz_b, J, M]
                elif self.train_step_type == "SAA":
                    if self.SAA_attention:
                        pi, _ = self.ppo.policy_old(
                            fea_j=state.fea_j_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_j_tensor.shape[1],
                                state.fea_j_tensor.shape[2],
                            ),  # [sz_b, num_samples, N, 8]
                            op_mask=state.op_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.op_mask_tensor.shape[1],
                                state.op_mask_tensor.shape[2],
                            ),  # [sz_b, num_samples, N, N]
                            candidate=state.candidate_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.candidate_tensor.shape[1],
                            ),  # [sz_b, J]
                            fea_m=state.fea_m_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_m_tensor.shape[1],
                                state.fea_m_tensor.shape[2],
                            ),  # [sz_b, M, 6]
                            mch_mask=state.mch_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.mch_mask_tensor.shape[1],
                                state.mch_mask_tensor.shape[2],
                            ),  # [sz_b, M, M]
                            comp_idx=state.comp_idx_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.comp_idx_tensor.shape[1],
                                state.comp_idx_tensor.shape[2],
                                state.comp_idx_tensor.shape[3],
                            ),  # [sz_b, M, M, J]
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.dynamic_pair_mask_tensor.shape[1],
                                state.dynamic_pair_mask_tensor.shape[2],
                            ),  # [sz_b, J, M]
                            fea_pairs=state.fea_pairs_tensor.view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_pairs_tensor.shape[1],
                                state.fea_pairs_tensor.shape[2],
                                state.fea_pairs_tensor.shape[3],
                            ),  # [sz_b, J, M]
                            num_samples=self.num_input_realizations,
                        )
                    else:
                        pi, _ = self.ppo.policy_old(
                            fea_j=state.fea_j_tensor[indices],  # [sz_b, N, 8]
                            op_mask=state.op_mask_tensor[indices],  # [sz_b, N, N]
                            candidate=state.candidate_tensor[indices],  # [sz_b, J]
                            fea_m=state.fea_m_tensor[indices],  # [sz_b, M, 6]
                            mch_mask=state.mch_mask_tensor[indices],  # [sz_b, M, M]
                            comp_idx=state.comp_idx_tensor[indices],  # [sz_b, M, M, J]
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                                indices
                            ],  # [sz_b, J, M]
                            fea_pairs=state.fea_pairs_tensor[indices],
                        )  # [sz_b, J, M]

            action = greedy_select_action(pi)

            if self.train_step_type == "random":
                state, _, done = self.vali_env.step(actions=action.cpu().numpy())
            elif self.train_step_type == "SAA":
                state, _, done = self.vali_env.step(
                    actions=action.repeat_interleave(self.num_input_realizations + 1)
                )
                self.vali_reward_env.step(
                    action.repeat_interleave(self.num_eval_realizations)
                )

            if done.all():
                break

        self.ppo.policy.train()
        if self.train_step_type == "random":
            return self.vali_env.current_makespan
        elif self.train_step_type == "SAA":
            if self.stoch_obj == "mean":
                obj = np.mean(
                    self.vali_reward_env.current_makespan.cpu()
                    .numpy()
                    .reshape((100, -1)),
                    axis=1,
                )
            elif self.stoch_obj == "VaR":
                obj = np.quantile(
                    self.vali_reward_env.current_makespan.cpu()
                    .numpy()
                    .reshape((100, -1)),
                    q=self.VaR_alpha,
                    axis=1,
                )

        self.vali_env.to_cpu()
        self.vali_reward_env.to_cpu()
        torch.cuda.empty_cache()
        return obj

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        self.vali_env.to_gpu()
        self.vali_reward_env.to_gpu()
        state = self.vali_env.reset()
        self.vali_reward_env.reset()

        if self.train_step_type == "SAA":
            indices = [i * (self.num_input_realizations + 1) for i in range(100)]
        while True:
            with torch.no_grad():
                batch_idx = ~self.vali_env.done_flag
                if self.train_step_type == "random":
                    pi, _ = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                        op_mask=state.op_mask_tensor[batch_idx],  # [sz_b, N, N]
                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                            batch_idx
                        ],  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor[batch_idx],
                    )  # [sz_b, J, M]
                elif self.train_step_type == "SAA":

                    if self.SAA_attention:
                        pi, _ = self.ppo.policy_old(
                            fea_j=state.fea_j_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_j_tensor.shape[1],
                                state.fea_j_tensor.shape[2],
                            ),  # [sz_b, num_samples, N, 8]
                            op_mask=state.op_mask_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.op_mask_tensor.shape[1],
                                state.op_mask_tensor.shape[2],
                            ),  # [sz_b, num_samples, N, N]
                            candidate=state.candidate_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.candidate_tensor.shape[1],
                            ),  # [sz_b, J]
                            fea_m=state.fea_m_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_m_tensor.shape[1],
                                state.fea_m_tensor.shape[2],
                            ),  # [sz_b, M, 6]
                            mch_mask=state.mch_mask_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.mch_mask_tensor.shape[1],
                                state.mch_mask_tensor.shape[2],
                            ),  # [sz_b, M, M]
                            comp_idx=state.comp_idx_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.comp_idx_tensor.shape[1],
                                state.comp_idx_tensor.shape[2],
                                state.comp_idx_tensor.shape[3],
                            ),  # [sz_b, M, M, J]
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                                batch_idx
                            ].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.dynamic_pair_mask_tensor.shape[1],
                                state.dynamic_pair_mask_tensor.shape[2],
                            ),  # [sz_b, J, M]
                            fea_pairs=state.fea_pairs_tensor[batch_idx].view(
                                -1,
                                self.num_input_realizations + 1,
                                state.fea_pairs_tensor.shape[1],
                                state.fea_pairs_tensor.shape[2],
                                state.fea_pairs_tensor.shape[3],
                            ),  # [sz_b, J, M]
                            num_samples=self.num_input_realizations,
                        )
                    else:
                        batch_idx = batch_idx[indices]
                        pi, _ = self.ppo.policy_old(
                            fea_j=state.fea_j_tensor[indices][
                                batch_idx
                            ],  # [sz_b, N, 8]
                            op_mask=state.op_mask_tensor[indices][
                                batch_idx
                            ],  # [sz_b, N, N]
                            candidate=state.candidate_tensor[indices][
                                batch_idx
                            ],  # [sz_b, J]
                            fea_m=state.fea_m_tensor[indices][
                                batch_idx
                            ],  # [sz_b, M, 6]
                            mch_mask=state.mch_mask_tensor[indices][
                                batch_idx
                            ],  # [sz_b, M, M]
                            comp_idx=state.comp_idx_tensor[indices][
                                batch_idx
                            ],  # [sz_b, M, M, J]
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor[indices][
                                batch_idx
                            ],  # [sz_b, J, M]
                            fea_pairs=state.fea_pairs_tensor[indices][batch_idx],
                        )  # [sz_b, J, M]

            action = greedy_select_action(pi)

            if self.train_step_type == "random":
                state, _, done = self.vali_env.step(actions=action.cpu().numpy())
            elif self.train_step_type == "SAA":
                state, _, done = self.vali_env.step(
                    actions=action.repeat_interleave(self.num_input_realizations + 1)
                )
                self.vali_reward_env.step(
                    action.repeat_interleave(self.num_eval_realizations)
                )

            if done.all():
                break

        self.ppo.policy.train()
        if self.train_step_type == "random":
            return self.vali_env.current_makespan
        elif self.train_step_type == "SAA":
            if self.stoch_obj == "mean":
                obj = np.mean(
                    self.vali_reward_env.current_makespan.cpu()
                    .numpy()
                    .reshape((100, -1)),
                    axis=1,
                )
            elif self.stoch_obj == "VaR":
                obj = np.quantile(
                    self.vali_reward_env.current_makespan.cpu()
                    .numpy()
                    .reshape((100, -1)),
                    q=self.VaR_alpha,
                    axis=1,
                )

        self.vali_env.to_cpu()
        self.vali_reward_env.to_cpu()
        torch.cuda.empty_cache()
        return obj

    def save_model(self):
        """
        save the model
        """
        dist_str = f"{self.variance_dist}/" if self.variance_dist != "lognormal" else ""
        torch.save(
            self.ppo.policy.state_dict(),
            f"./trained_network/{self.variance}/{dist_str}{self.data_source}"
            f"/{self.model_name}.pth",
        )

    def load_model(self):
        """
        load the trained model
        """
        model_path = f"./trained_network/{self.data_source}/{self.model_name}.pth"
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))


def main():
    print("Start training...")
    trainer = Trainer(configs)
    trainer.train()


if __name__ == "__main__":
    main()
