import copy
import itertools
import sys

import numpy as np
import torch
from torch.masked import masked_tensor

from fjsp_env_same_op_nums_torch import EnvState


class FJSPEnvForVariousOpNums:
    """
    a batch of fjsp environments that have various number of operations
    """

    def __init__(self, n_j, n_m):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        self.op_fea_dim = 10
        self.mch_fea_dim = 8

    def set_static_properties(self):
        """
        define static properties
        """
        self.multi_env_mch_diag = (
            torch.eye(self.number_of_machines, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(self.number_of_envs, 1, 1)
        )

        self.env_idxs = torch.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat_interleave(self.number_of_jobs).view(
            self.number_of_envs, self.number_of_jobs
        )

        # [E, N]
        self.mask_dummy_node = torch.full(
            size=[self.number_of_envs, self.max_number_of_ops],
            fill_value=False,
            dtype=torch.bool,
        )

        cols = torch.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True

        a = self.mask_dummy_node[:, :, None]
        self.dummy_mask_fea_j = torch.tile(a, (1, 1, self.op_fea_dim))

        self.flag_exist_dummy_node = ~(
            self.env_number_of_ops == self.max_number_of_ops
        ).all()

    def set_initial_data(self, job_length_list, op_pt_list, group_size):

        self.group_size = group_size
        self.number_of_envs = len(job_length_list)
        self.job_length = torch.tensor(np.array(job_length_list))
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.env_number_of_ops = torch.tensor(
            [op_pt_list[k].shape[0] for k in range(self.number_of_envs)]
        )
        self.max_number_of_ops = torch.max(self.env_number_of_ops)

        self.set_static_properties()

        self.virtual_job_length = torch.clone(self.job_length)
        self.virtual_job_length[:, -1] += (
            self.max_number_of_ops - self.env_number_of_ops
        )
        # [E, N, M]
        self.op_pt = torch.tensor(
            np.concatenate(
                [
                    np.pad(
                        op_pt_list[k * self.group_size : (k + 1) * self.group_size],
                        (
                            (0, 0),
                            (
                                0,
                                self.max_number_of_ops.cpu()
                                - self.env_number_of_ops[k * self.group_size].cpu(),
                            ),
                            (0, 0),
                        ),
                        "constant",
                        constant_values=0,
                    )
                    for k in range(self.number_of_envs // self.group_size)
                ]
            )
        ).float()
        self.pt_lower_bound = torch.min(self.op_pt)
        self.pt_upper_bound = torch.max(self.op_pt)
        self.true_op_pt = self.op_pt.clone()

        self.op_pt = (self.op_pt - self.pt_lower_bound) / (
            self.pt_upper_bound - self.pt_lower_bound + 1e-8
        )

        self.process_relation = self.op_pt != 0
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = torch.sum(self.process_relation, 2)
        self.compatible_mch = torch.sum(self.process_relation, 1)

        self.unmasked_op_pt = self.op_pt.clone()

        head_op_id = torch.zeros((self.number_of_envs, 1))

        self.job_first_op_id = torch.cat(
            [head_op_id, torch.cumsum(self.job_length, dim=1)[:, :-1]], dim=1
        ).long()
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()
        self.init_op_mask()

        self.op_pt = masked_tensor(self.op_pt, mask=self.process_relation)

        self.op_mean_pt = torch.mean(self.op_pt, dim=2).get_data()

        self.op_min_pt = torch.amin(self.op_pt, dim=-1).get_data()
        self.op_max_pt = torch.amax(self.op_pt, dim=-1).get_data()
        self.pt_span = self.op_max_pt - self.op_min_pt

        self.mch_min_pt = torch.amax(self.op_pt, dim=1).get_data()
        self.mch_max_pt = torch.amax(self.op_pt, dim=1)

        self.op_ct_lb = self.op_min_pt.clone()
        self.op_ct_lb = compiled_func_2(
            self.number_of_envs,
            group_size,
            self.job_first_op_id,
            self.job_last_op_id,
            self.number_of_jobs,
            self.op_ct_lb,
        )
        self.op_match_job_left_op_nums = torch.cat(
            [
                torch.repeat_interleave(
                    self.job_length[k * self.group_size : (k + 1) * self.group_size],
                    repeats=self.virtual_job_length[k * self.group_size],
                    dim=1,
                )
                for k in range(self.number_of_envs // self.group_size)
            ],
            dim=0,
        )

        self.job_remain_work = list(
            itertools.chain(
                *[
                    list(
                        torch.stack(
                            [
                                torch.sum(
                                    self.op_mean_pt[
                                        k * self.group_size : (k + 1) * self.group_size,
                                        self.job_first_op_id[k * self.group_size][
                                            i
                                        ] : self.job_last_op_id[k * self.group_size][i]
                                        + 1,
                                    ],
                                    dim=1,
                                )
                                for i in range(self.number_of_jobs)
                            ]
                        ).permute(1, 0)
                    )
                    for k in range(self.number_of_envs // self.group_size)
                ]
            )
        )

        self.op_match_job_remain_work = torch.cat(
            [
                torch.repeat_interleave(
                    torch.stack(
                        self.job_remain_work[
                            k * self.group_size : (k + 1) * self.group_size
                        ]
                    ),
                    repeats=self.virtual_job_length[k * self.group_size],
                    dim=1,
                )
                for k in range(self.number_of_envs // self.group_size)
            ],
            dim=0,
        )

        self.construct_op_features()

        # shape reward
        self.init_quality = torch.max(self.op_ct_lb, dim=1)[0]

        self.max_endTime = self.init_quality.clone()
        # old
        self.mch_available_op_nums = self.compatible_mch.clone()
        self.mch_current_available_op_nums = self.compatible_mch.clone()
        self.candidate_pt = torch.cat(
            [
                self.unmasked_op_pt[
                    k * self.group_size : (k + 1) * self.group_size,
                    self.candidate[k * self.group_size],
                ]
                for k in range(self.number_of_envs // self.group_size)
            ],
            dim=0,
        )

        self.dynamic_pair_mask = self.candidate_pt == 0
        self.candidate_process_relation = self.dynamic_pair_mask.clone()
        self.mch_current_available_jc_nums = torch.sum(
            ~self.candidate_process_relation, dim=1
        )

        self.mch_mean_pt = torch.mean(self.op_pt, dim=1).get_data()
        self.mch_mean_pt[torch.isnan(self.mch_mean_pt)] = 0
        # construct machine features [E, M, 6]

        # construct Compete Tensor : [E, M, M, J]
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        # construct mch graph adjacency matrix : [E, M, M]
        self.init_mch_mask()
        self.construct_mch_features()

        self.construct_pair_features()

        self.old_state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )

        # old record
        self.old_op_mask = self.op_mask.clone()
        self.old_mch_mask = self.mch_mask.clone()
        self.old_op_ct_lb = self.op_ct_lb.clone()
        self.old_op_match_job_left_op_nums = self.op_match_job_left_op_nums.clone()
        self.old_op_match_job_remain_work = self.op_match_job_remain_work.clone()
        self.old_init_quality = self.init_quality.clone()
        self.old_candidate_pt = self.candidate_pt.clone()
        self.old_candidate_process_relation = self.candidate_process_relation.clone()
        self.old_mch_current_available_op_nums = (
            self.mch_current_available_op_nums.clone()
        )
        self.old_mch_current_available_jc_nums = (
            self.mch_current_available_jc_nums.clone()
        )
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        self.initial_vars()
        self.op_mask = self.old_op_mask.clone()
        self.mch_mask = self.old_mch_mask.clone()
        self.op_ct_lb = self.old_op_ct_lb.clone()
        self.op_match_job_left_op_nums = self.old_op_match_job_left_op_nums.clone()
        self.op_match_job_remain_work = self.old_op_match_job_remain_work.clone()
        self.init_quality = self.old_init_quality
        self.max_endTime = self.init_quality.clone()
        self.candidate_pt = self.old_candidate_pt.clone()
        self.candidate_process_relation = self.old_candidate_process_relation.clone()
        self.mch_current_available_op_nums = (
            self.old_mch_current_available_op_nums.clone()
        )
        self.mch_current_available_jc_nums = (
            self.old_mch_current_available_jc_nums.clone()
        )
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.done_flag = torch.full(
            size=(self.number_of_envs,), fill_value=0, dtype=torch.bool
        )
        self.current_makespan = torch.full((self.number_of_envs,), float("-inf"))
        self.mch_queue = torch.full(
            size=[
                self.number_of_envs,
                self.number_of_machines,
                self.max_number_of_ops + 1,
            ],
            fill_value=-99,
            dtype=torch.int32,
        )
        self.mch_queue_len = torch.zeros(
            (self.number_of_envs, self.number_of_machines), dtype=torch.int32
        )
        self.mch_queue_last_op_id = torch.zeros(
            (self.number_of_envs, self.number_of_machines), dtype=torch.int32
        )
        self.op_ct = torch.zeros((self.number_of_envs, self.max_number_of_ops))
        self.mch_free_time = torch.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = torch.zeros(
            (self.number_of_envs, self.number_of_machines)
        )

        self.mch_waiting_time = torch.zeros(
            (self.number_of_envs, self.number_of_machines)
        )
        self.mch_working_flag = torch.zeros(
            (self.number_of_envs, self.number_of_machines)
        )

        self.next_schedule_time = torch.zeros(self.number_of_envs)
        self.candidate_free_time = torch.zeros(
            (self.number_of_envs, self.number_of_jobs)
        )

        self.true_op_ct = torch.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = torch.zeros(
            (self.number_of_envs, self.number_of_jobs)
        )
        self.true_mch_free_time = torch.zeros(
            (self.number_of_envs, self.number_of_machines)
        )

        self.candidate = self.job_first_op_id.clone()

        self.unscheduled_op_nums = torch.clone(self.env_number_of_ops)
        self.mask = torch.full((self.number_of_envs, self.number_of_jobs), False)

        self.op_scheduled_flag = torch.zeros(
            (self.number_of_envs, self.max_number_of_ops)
        )
        self.op_waiting_time = torch.zeros(
            (self.number_of_envs, self.max_number_of_ops)
        )
        self.op_remain_work = torch.zeros((self.number_of_envs, self.max_number_of_ops))

        self.op_available_mch_nums = (
            self.compatible_op.clone() / self.number_of_machines
        )
        self.pair_free_time = torch.zeros(
            (self.number_of_envs, self.number_of_jobs, self.number_of_machines)
        )
        self.remain_process_relation = self.process_relation.clone()

        self.delete_mask_fea_j = torch.full(
            (self.number_of_envs, self.max_number_of_ops, self.op_fea_dim), False
        )

    def step(self, actions):
        self.incomplete_env_idx = torch.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(
            self.number_of_envs - torch.sum(self.done_flag)
        )

        chosen_job = (actions // self.number_of_machines).int()
        chosen_mch = (actions % self.number_of_machines).int()
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job].int()

        if (
            self.reverse_process_relation[
                self.incomplete_env_idx, chosen_op, chosen_mch
            ]
        ).any():
            print(
                f"FJSP_Env.py Error from choosing action: Op {chosen_op} can't be processed by Mch {chosen_mch}"
            )
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (
            chosen_op != self.job_last_op_id[self.incomplete_env_idx, chosen_job]
        )
        self.candidate[self.incomplete_env_idx, chosen_job] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_job] = ~candidate_add_flag

        self.mch_queue[
            self.incomplete_env_idx,
            chosen_mch,
            self.mch_queue_len[self.incomplete_env_idx, chosen_mch],
        ] = chosen_op

        self.mch_queue_len[self.incomplete_env_idx, chosen_mch] += 1

        # [E]
        chosen_op_st = torch.maximum(
            self.candidate_free_time[self.incomplete_env_idx, chosen_job],
            self.mch_free_time[self.incomplete_env_idx, chosen_mch],
        )

        self.op_ct[self.incomplete_env_idx, chosen_op] = (
            chosen_op_st + self.op_pt[self.incomplete_env_idx, chosen_op, chosen_mch]
        ).get_data()
        self.candidate_free_time[self.incomplete_env_idx, chosen_job] = self.op_ct[
            self.incomplete_env_idx, chosen_op
        ]
        self.mch_free_time[self.incomplete_env_idx, chosen_mch] = self.op_ct[
            self.incomplete_env_idx, chosen_op
        ]

        true_chosen_op_st = torch.maximum(
            self.true_candidate_free_time[self.incomplete_env_idx, chosen_job],
            self.true_mch_free_time[self.incomplete_env_idx, chosen_mch],
        )
        self.true_op_ct[self.incomplete_env_idx, chosen_op] = (
            true_chosen_op_st
            + self.true_op_pt[self.incomplete_env_idx, chosen_op, chosen_mch]
        )
        self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] = (
            self.true_op_ct[self.incomplete_env_idx, chosen_op]
        )
        self.true_mch_free_time[self.incomplete_env_idx, chosen_mch] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op
        ]

        self.current_makespan[self.incomplete_env_idx] = torch.maximum(
            self.current_makespan[self.incomplete_env_idx],
            self.true_op_ct[self.incomplete_env_idx, chosen_op],
        )
        self.candidate_pt, self.candidate_process_relation = compiled_func_3(
            self.incomplete_env_idx,
            self.group_size,
            candidate_add_flag,
            self.candidate_pt,
            self.unmasked_op_pt,
            chosen_job,
            chosen_op,
            self.candidate_process_relation,
            self.reverse_process_relation,
        )

        candidateFT_for_compare = self.candidate_free_time.unsqueeze(2)
        mchFT_for_compare = self.mch_free_time.unsqueeze(1)
        self.pair_free_time = torch.maximum(candidateFT_for_compare, mchFT_for_compare)

        pair_free_time = self.pair_free_time[self.incomplete_env_idx]
        schedule_matrix = masked_tensor(
            pair_free_time,
            mask=~self.candidate_process_relation[self.incomplete_env_idx],
        )

        self.next_schedule_time[self.incomplete_env_idx] = torch.amin(
            schedule_matrix.view(self.number_of_incomplete_envs, -1), dim=1
        ).get_data()

        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1

        self.deleted_op_nodes = torch.logical_and(
            (self.op_ct <= self.next_schedule_time.unsqueeze(1) + 1e-5),
            self.op_scheduled_flag,
        )

        self.delete_mask_fea_j = torch.tile(
            self.deleted_op_nodes.unsqueeze(2), (1, 1, self.op_fea_dim)
        )

        self.update_op_mask()

        self.mch_queue_last_op_id[self.incomplete_env_idx, chosen_mch] = chosen_op

        self.unscheduled_op_nums[self.incomplete_env_idx] -= 1

        diff = (
            self.op_ct[self.incomplete_env_idx, chosen_op]
            - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        )

        self.op_ct_lb, self.op_match_job_left_op_nums, self.op_match_job_remain_work = (
            compiled_func(
                self.incomplete_env_idx,
                chosen_job,
                chosen_op,
                self.op_ct_lb,
                self.job_first_op_id,
                self.job_last_op_id,
                self.op_mean_pt,
                self.op_match_job_left_op_nums,
                self.op_match_job_remain_work,
                self.group_size,
                diff,
            )
        )

        self.op_waiting_time = torch.zeros(
            (self.number_of_envs, self.max_number_of_ops)
        )
        self.op_waiting_time[self.env_job_idx, self.candidate] = (
            ~self.mask
        ) * torch.maximum(
            self.next_schedule_time.unsqueeze(1) - self.candidate_free_time,
            torch.tensor(0),
        ) + self.mask * self.op_waiting_time[
            self.env_job_idx, self.candidate
        ]

        self.op_remain_work = torch.maximum(
            self.op_ct - self.next_schedule_time.unsqueeze(1), torch.tensor(0)
        )

        self.construct_op_features()

        self.dynamic_pair_mask = self.candidate_process_relation.clone()

        self.unavailable_pairs = pair_free_time > self.next_schedule_time[
            self.incomplete_env_idx, None, None
        ].repeat(1, pair_free_time.shape[1], self.pair_free_time.shape[2])
        self.dynamic_pair_mask[self.incomplete_env_idx] = torch.logical_or(
            self.dynamic_pair_mask[self.incomplete_env_idx], self.unavailable_pairs
        )

        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        self.mch_current_available_jc_nums = torch.sum(~self.dynamic_pair_mask, dim=1)
        self.mch_current_available_op_nums[
            self.incomplete_env_idx
        ] -= self.process_relation[self.incomplete_env_idx, chosen_op].int()

        mch_free_duration = (
            self.next_schedule_time[self.incomplete_env_idx].unsqueeze(1)
            - self.mch_free_time[self.incomplete_env_idx]
        )
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag[self.incomplete_env_idx] = mch_free_flag + 0.0
        self.mch_waiting_time[self.incomplete_env_idx] = (
            ~mch_free_flag
        ) * mch_free_duration

        self.mch_remain_work[self.incomplete_env_idx] = torch.maximum(
            -mch_free_duration, torch.tensor(0)
        )

        self.construct_mch_features()

        self.construct_pair_features()

        reward = self.max_endTime - torch.amax(self.op_ct_lb, dim=1)
        self.max_endTime = torch.amax(self.op_ct_lb, dim=1)

        self.state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )
        self.done_flag = self.done()

        return self.state, reward.cpu().numpy(), self.done_flag.cpu().numpy()

    def done(self):
        return self.step_count >= self.env_number_of_ops

    def construct_op_features(self):

        self.fea_j = torch.stack(
            (
                self.op_scheduled_flag,
                self.op_ct_lb,
                self.op_min_pt,
                self.pt_span,
                self.op_mean_pt,
                self.op_waiting_time,
                self.op_remain_work,
                self.op_match_job_left_op_nums,
                self.op_match_job_remain_work,
                self.op_available_mch_nums,
            ),
            axis=2,
        )

        if self.flag_exist_dummy_node:
            mask_all = torch.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def norm_operation_features(self, mask):
        self.fea_j[mask] = 0
        num_delete_nodes = torch.count_nonzero(mask[:, :, 0], dim=1)

        num_delete_nodes = num_delete_nodes.unsqueeze(1)
        num_left_nodes = self.max_number_of_ops - num_delete_nodes

        num_left_nodes = torch.maximum(num_left_nodes, torch.tensor(1e-8))

        mean_fea_j = torch.sum(self.fea_j, dim=1) / num_left_nodes
        temp = torch.where(self.delete_mask_fea_j, mean_fea_j.unsqueeze(1), self.fea_j)
        var_fea_j = torch.var(temp, dim=1, correction=0)

        std_fea_j = torch.sqrt(var_fea_j * self.max_number_of_ops / num_left_nodes)

        self.fea_j = (temp - mean_fea_j.unsqueeze(1)) / (std_fea_j.unsqueeze(1) + 1e-8)

    def construct_mch_features(self):

        self.fea_m = torch.stack(
            (
                self.mch_current_available_jc_nums,
                self.mch_current_available_op_nums,
                self.mch_min_pt,
                self.mch_mean_pt,
                self.mch_waiting_time,
                self.mch_remain_work,
                self.mch_free_time,
                self.mch_working_flag,
            ),
            axis=2,
        )

        self.norm_machine_features()

    def norm_machine_features(self):
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = torch.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs.unsqueeze(1)
        num_left_mchs = self.number_of_machines - num_delete_mchs

        num_left_mchs = torch.maximum(num_left_mchs, torch.tensor(1e-8))

        mean_fea_m = torch.sum(self.fea_m, axis=1) / num_left_mchs

        temp = torch.where(self.delete_mask_fea_m, mean_fea_m.unsqueeze(1), self.fea_m)
        var_fea_m = torch.var(temp, axis=1, correction=0)

        std_fea_m = torch.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = (temp - mean_fea_m.unsqueeze(1)) / (std_fea_m.unsqueeze(1) + 1e-8)

    def construct_pair_features(self):

        remain_op_pt = masked_tensor(
            self.op_pt.get_data(), mask=self.remain_process_relation
        )

        chosen_op_max_pt = torch.unsqueeze(
            self.op_max_pt[self.env_job_idx, self.candidate], dim=-1
        )

        max_remain_op_pt = torch.amax(
            torch.amax(remain_op_pt, dim=1, keepdim=True), dim=2, keepdim=True
        )
        max_remain_op_pt = max_remain_op_pt.get_data().masked_fill(
            ~max_remain_op_pt.get_mask(), 0 + 1e-8
        )

        mch_max_remain_op_pt = torch.amax(remain_op_pt, dim=1, keepdim=True)
        mch_max_remain_op_pt = mch_max_remain_op_pt.get_data().masked_fill(
            ~mch_max_remain_op_pt.get_mask(), 0 + 1e-8
        )

        pair_max_pt = (
            torch.amax(
                torch.amax(self.candidate_pt, dim=1, keepdim=True), dim=2, keepdims=True
            )
            + 1e-8
        )

        mch_max_candidate_pt = torch.amax(self.candidate_pt, dim=1, keepdim=True) + 1e-8

        pair_wait_time = self.op_waiting_time[
            self.env_job_idx, self.candidate
        ].unsqueeze(2) + self.mch_waiting_time.unsqueeze(1)

        chosen_job_remain_work = (
            torch.unsqueeze(
                self.op_match_job_remain_work[self.env_job_idx, self.candidate], dim=-1
            )
            + 1e-8
        )

        self.fea_pairs = torch.stack(
            (
                self.candidate_pt,
                self.candidate_pt / chosen_op_max_pt,
                self.candidate_pt / mch_max_candidate_pt,
                self.candidate_pt / max_remain_op_pt,
                self.candidate_pt / mch_max_remain_op_pt,
                self.candidate_pt / pair_max_pt,
                self.candidate_pt / chosen_job_remain_work,
                pair_wait_time,
            ),
            dim=-1,
        )

    def update_mch_mask(self):

        self.mch_mask = (
            self.logic_operator(self.remain_process_relation, return_float=False)
            .sum(axis=-1)
            .bool()
        )
        self.delete_mask_fea_m = torch.tile(
            ~(torch.sum(self.mch_mask, keepdims=True, axis=-1).bool()),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):

        self.mch_mask = (
            self.logic_operator(self.remain_process_relation, return_float=False)
            .sum(axis=-1)
            .bool()
        )
        self.delete_mask_fea_m = torch.tile(
            ~(torch.sum(self.mch_mask, keepdim=True, axis=-1).bool()),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):

        self.op_mask = torch.full(
            size=(self.number_of_envs, self.max_number_of_ops, 3),
            fill_value=0.0,
            dtype=torch.float32,
        )
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self):

        object_mask = torch.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = torch.logical_or(object_mask, self.op_mask).float()

    def logic_operator(self, x, flagT=True, return_float=True):
        if flagT:
            x = x.transpose(2, 1)
        d1 = torch.unsqueeze(x, 2)
        d2 = torch.unsqueeze(x, 1)

        return (
            torch.logical_and(d1, d2).float()
            if return_float
            else torch.logical_and(d1, d2)
        )

    def to_cpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cpu())

        for name, value in vars(self.old_state).items():
            if isinstance(value, torch.Tensor):
                setattr(self.old_state, name, value.cpu())

        for name, value in vars(self.state).items():
            if isinstance(value, torch.Tensor):
                setattr(self.state, name, value.cpu())

    def to_gpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cuda())

        for name, value in vars(self.old_state).items():
            if isinstance(value, torch.Tensor):
                setattr(self.old_state, name, value.cuda())

        if hasattr(self, "state"):
            for name, value in vars(self.state).items():
                if isinstance(value, torch.Tensor):
                    setattr(self.state, name, value.cuda())


@torch.jit.script
def compiled_func(
    incomplete_env_idx,
    chosen_job,
    chosen_op,
    op_ct_lb,
    job_first_op_id,
    job_last_op_id,
    op_mean_pt,
    op_match_job_left_op_nums,
    op_match_job_remain_work,
    group_size: int,
    diff,
):
    for k, j in enumerate(torch.unique((incomplete_env_idx // group_size))):
        op_ct_lb[
            j * group_size : (j + 1) * group_size,
            chosen_op[k * group_size] : job_last_op_id[
                j * group_size, chosen_job[k * group_size]
            ]
            + 1,
        ] += (
            diff[k * group_size : (k + 1) * group_size]
            .unsqueeze(1)
            .repeat(
                1,
                -int(chosen_op[k * group_size].item())
                + int(job_last_op_id[j * group_size, chosen_job[k * group_size]] + 1),
            )
        )
        op_match_job_left_op_nums[
            j * group_size : (j + 1) * group_size,
            job_first_op_id[
                j * group_size, chosen_job[k * group_size]
            ] : job_last_op_id[j * group_size, chosen_job[k * group_size]]
            + 1,
        ] -= 1
        op_match_job_remain_work[
            j * group_size : (j + 1) * group_size,
            job_first_op_id[
                j * group_size, chosen_job[k * group_size]
            ] : job_last_op_id[j * group_size, chosen_job[k * group_size]]
            + 1,
        ] -= (
            op_mean_pt[j * group_size : (j + 1) * group_size, chosen_op[k * group_size]]
            .unsqueeze(1)
            .repeat(
                1,
                -int(job_first_op_id[j * group_size, chosen_job[k * group_size]].item())
                + int(
                    job_last_op_id[j * group_size, chosen_job[k * group_size]].item()
                    + 1
                ),
            )
        )

    return op_ct_lb, op_match_job_left_op_nums, op_match_job_remain_work


@torch.jit.script
def compiled_func_2(
    number_of_envs: int,
    group_size: int,
    job_first_op_id,
    job_last_op_id,
    number_of_jobs: int,
    op_ct_lb,
):
    for k in range(number_of_envs // group_size):
        for i in range(number_of_jobs):
            op_ct_lb[
                k * group_size : (k + 1) * group_size,
                job_first_op_id[k * group_size][i] : job_last_op_id[k * group_size][i]
                + 1,
            ] = torch.cumsum(
                op_ct_lb[
                    k * group_size : (k + 1) * group_size,
                    job_first_op_id[k * group_size][i] : job_last_op_id[k * group_size][
                        i
                    ]
                    + 1,
                ],
                dim=1,
            )
    return op_ct_lb


@torch.jit.script
def compiled_func_3(
    incomplete_env_idx,
    group_size: int,
    candidate_add_flag,
    candidate_pt,
    unmasked_op_pt,
    chosen_job,
    chosen_op,
    candidate_process_relation,
    reverse_process_relation,
):
    for k, j in enumerate(torch.unique((incomplete_env_idx // group_size))):
        if candidate_add_flag[k * group_size]:
            candidate_pt[
                j * group_size : (j + 1) * group_size, chosen_job[k * group_size]
            ] = unmasked_op_pt[
                j * group_size : (j + 1) * group_size, chosen_op[k * group_size] + 1
            ]
            candidate_process_relation[
                j * group_size : (j + 1) * group_size, chosen_job[k * group_size]
            ] = reverse_process_relation[
                j * group_size : (j + 1) * group_size, chosen_op[k * group_size] + 1
            ]
        else:
            candidate_process_relation[
                j * group_size : (j + 1) * group_size, chosen_job[k * group_size]
            ] = 1
    return candidate_pt, candidate_process_relation
