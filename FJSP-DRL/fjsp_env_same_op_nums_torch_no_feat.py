import sys

import numpy as np
import torch
from torch.masked import masked_tensor


class FJSPEnvForSameOpNumsNoFeat:
    """
        a batch of fjsp environments that have the same number of operations

        let E/N/J/M denote the number of envs/operations/jobs/machines
        Remark: The index of operations has been rearranged in natural order
        eg. {O_{11}, O_{12}, O_{13}, O_{21}, O_{22}}  <--> {0,1,2,3,4}

        Attributes:

        job_length: the number of operations in each job (shape [J])
        op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$

        candidate: the index of candidates  [sz_b, J]
        fea_j: input operation feature vectors with shape [sz_b, N, 8]
        op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        fea_m: input operation feature vectors with shape [sz_b, M, 6]
        mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking incompatible op-mch pairs
        fea_pairs: pair features with shape [sz_b, J, M, 8]
    """

    def __init__(self, n_j, n_m):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m

    def set_static_properties(self):
        """
            define static properties
        """

        self.env_idxs = torch.arange(self.number_of_envs)
        self.op_idx = torch.arange(self.number_of_ops).unsqueeze(0)

    def set_initial_data(self, job_length_list, op_pt_list, group_size):
        """
            initialize the data of the instances

        :param job_length_list: the list of 'job_length'
        :param op_pt_list: the list of 'op_pt'
        """

        self.number_of_envs = len(job_length_list)
        self.job_length = torch.tensor(np.array(job_length_list))
        self.op_pt = torch.tensor(np.array(op_pt_list))
        self.number_of_ops = self.op_pt.shape[1]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.set_static_properties()

        # [E, N, M]
        self.pt_lower_bound = torch.min(self.op_pt)
        self.pt_upper_bound = torch.max(self.op_pt)
        self.true_op_pt = self.op_pt.clone()

        # normalize the processing time
        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # bool 3-d array formulating the compatible relation with shape [E,N,M]
        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        head_op_id = torch.zeros((self.number_of_envs, 1))

        # the index of first operation of each job ([E,J])
        self.job_first_op_id = torch.cat([head_op_id, torch.cumsum(self.job_length, dim=1)[:, :-1]], dim=1).long()
        # the index of last operation of each job ([E,J])
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.initial_vars()


        self.op_pt = masked_tensor(self.op_pt, mask=self.process_relation)

        """
            compute operation raw features
        """
        self.op_mean_pt = torch.mean(self.op_pt, dim=2).get_data()

        self.op_min_pt = torch.amin(self.op_pt, dim=-1).get_data()
        self.op_max_pt = torch.amax(self.op_pt, dim=-1).get_data()
        self.pt_span = self.op_max_pt - self.op_min_pt
        # [E, M]
        self.mch_min_pt = torch.amax(self.op_pt, dim=1).get_data()
        self.mch_max_pt = torch.amax(self.op_pt, dim=1)

        # the estimated lower bound of complete time of operations
        self.op_ct_lb = self.op_min_pt.clone()

        indices_k = torch.arange(self.number_of_envs).unsqueeze(-1).repeat_interleave(
            self.number_of_jobs, dim=1).unsqueeze(-1)
        indices_i = [[list(range(self.job_first_op_id[0, i], self.job_last_op_id[0, i] + 1))
                      for i in range(self.number_of_jobs)]] * self.number_of_envs
        self.op_ct_lb = torch.cumsum(self.op_ct_lb[indices_k, indices_i], dim=2).view(self.number_of_envs, -1)

        # shape reward
        self.init_quality = torch.max(self.op_ct_lb, dim=1)[0]

        self.max_endTime = self.init_quality.clone()
        """
            compute machine raw features
        """
        # construct machine features [E, M, 6]

        # old record
        self.old_init_quality = self.init_quality.clone()
        # state
        return {"makespan": self.current_makespan}

    def reset(self):
        """
           reset the environments
        :return: the state
        """
        self.initial_vars()

        # copy the old data
        self.init_quality = self.old_init_quality
        self.max_endTime = self.init_quality.clone()

        # copy the old state
        return {"makespan": self.current_makespan}

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.step_count = 0
        # the array that records the makespan of all environments
        self.current_makespan = torch.full((self.number_of_envs,), float("-inf"))
        # the complete time of operations ([E,N])
        self.op_ct = torch.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = torch.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate_free_time = torch.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = torch.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = torch.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = torch.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = self.job_first_op_id.clone()


    def step(self, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]

        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag

        # the start processing time of chosen operations
        chosen_op_st = torch.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
                                  self.mch_free_time[self.env_idxs, chosen_mch])

        self.op_ct[self.env_idxs, chosen_op] = (chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op, chosen_mch]).get_data()
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

        true_chosen_op_st = torch.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
                                       self.true_mch_free_time[self.env_idxs, chosen_mch])
        self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        self.current_makespan = torch.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])

        """
            update the state
        """

        # update operation raw features
        diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

        mask1 = (self.op_idx >= chosen_op.unsqueeze(1)) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1).unsqueeze(1))
        self.op_ct_lb[mask1] += torch.tile(diff.unsqueeze(1), (1, self.number_of_ops))[mask1]

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.max_endTime - torch.amax(self.op_ct_lb, dim=1)
        self.max_endTime = torch.amax(self.op_ct_lb, dim=1)

        return {"makespan": self.current_makespan}, reward.cpu().numpy(), self.done()

    def done(self):
        """
            compute the done flag
        """
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def to_cpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cpu())

    def to_gpu(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.cuda())
