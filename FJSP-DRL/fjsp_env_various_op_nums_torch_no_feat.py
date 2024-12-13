import sys

import numpy as np
import torch
from torch.masked import masked_tensor


class FJSPEnvForVariousOpNumsNoFeat:
    """
    a batch of fjsp environments that have various number of operations
    """

    def __init__(self, n_j, n_m):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m

    def set_static_properties(self):
        """
        define static properties
        """

        self.env_idxs = torch.arange(self.number_of_envs)

        # [E, N]

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

        head_op_id = torch.zeros((self.number_of_envs, 1))

        self.job_first_op_id = torch.cat(
            [head_op_id, torch.cumsum(self.job_length, dim=1)[:, :-1]], dim=1
        ).long()
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()

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
            self.group_size,
            self.job_first_op_id,
            self.job_last_op_id,
            self.op_ct_lb,
            self.number_of_jobs,
        )

        # shape reward
        self.init_quality = torch.max(self.op_ct_lb, dim=1)[0]

        self.max_endTime = self.init_quality.clone()
        # construct machine features [E, M, 6]

        # old record
        self.old_init_quality = self.init_quality.clone()
        # state
        return {"makespan": self.current_makespan}

    def reset(self):
        self.initial_vars()

        self.init_quality = self.old_init_quality
        self.max_endTime = self.init_quality.clone()

        return {"makespan": self.current_makespan}

    def initial_vars(self):
        self.step_count = 0
        self.done_flag = torch.full(
            size=(self.number_of_envs,), fill_value=0, dtype=torch.bool
        )
        self.current_makespan = torch.full((self.number_of_envs,), float("-inf"))
        self.op_ct = torch.zeros((self.number_of_envs, self.max_number_of_ops))
        self.mch_free_time = torch.zeros((self.number_of_envs, self.number_of_machines))

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

    def step(self, actions):
        self.incomplete_env_idx = torch.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(
            self.number_of_envs - torch.sum(self.done_flag)
        )

        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job]

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

        diff = (
            self.op_ct[self.incomplete_env_idx, chosen_op]
            - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        )

        self.op_ct_lb = compiled_func(
            self.incomplete_env_idx,
            self.group_size,
            chosen_op,
            chosen_job,
            self.job_last_op_id,
            self.op_ct_lb,
            diff,
        )

        reward = self.max_endTime - torch.amax(self.op_ct_lb, dim=1)
        self.max_endTime = torch.amax(self.op_ct_lb, dim=1)

        self.done_flag = self.done()

        return {"makespan": self.current_makespan}, reward.cpu().numpy(), self.done_flag

    def done(self):
        return self.step_count >= self.env_number_of_ops

    def to_cpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cpu())

    def to_gpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cuda())


@torch.jit.script
def compiled_func(
    incomplete_env_idx,
    group_size: int,
    chosen_op,
    chosen_job,
    job_last_op_id,
    op_ct_lb,
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
    return op_ct_lb


@torch.jit.script
def compiled_func_2(
    number_of_envs: int,
    group_size: int,
    job_first_op_id,
    job_last_op_id,
    op_ct_lb,
    number_of_jobs: int,
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
