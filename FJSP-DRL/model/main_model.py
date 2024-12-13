import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import nonzero_averaging
from model.attention_layer import *
from model.scenario_processing_module_layers import (
    ScenarioProcessingModuleWithoutAggregation,
)
from model.sub_layers import *


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = (
            config.fea_j_input_dim
            if not config.SAA_attention
            else config.fea_j_input_dim + config.SAA_attention_dim
        )
        self.fea_m_input_dim = (
            config.fea_m_input_dim
            if not config.SAA_attention
            else config.fea_m_input_dim + config.SAA_attention_dim
        )
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i]
                    * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i]
                    * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = (
                candidate.unsqueeze(-1).repeat(1, 1, fea_j.shape[-1]).type(torch.int64)
            )

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul, fea_j_jc).reshape(
                sz_b, M, M, -1
            )
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1,
        ).to(device)
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1,
        ).to(device)

    def forward(
        self,
        fea_j,
        op_mask,
        candidate,
        fea_m,
        mch_mask,
        comp_idx,
        dynamic_pair_mask,
        fea_pairs,
    ):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = (
            Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat(
            (
                Fea_j_JC_serialized,
                Fea_m_serialized,
                Fea_Gj_input,
                Fea_Gm_input,
                fea_pairs,
            ),
            dim=-1,
        )

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v


class SPM_DAN(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(SPM_DAN, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8
        self.SAA_attention_dim = config.SAA_attention_dim
        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim
            + self.pair_input_dim
            + config.SAA_attention_dim,
            config.hidden_dim_actor,
            1,
        ).to(device)
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1,
        ).to(device)

        self.fea_j_linear = nn.Linear(10, config.SAA_attention_dim)
        self.fea_m_linear = nn.Linear(8, config.SAA_attention_dim)

        self.fea_j_embedder = nn.Sequential(
            ScenarioProcessingModuleWithoutAggregation(
                dim_in=config.SAA_attention_dim,
                dim_out=config.SAA_attention_dim,
                num_heads=1,
                num_inds=16,
                ln=False,
            )
        )

        self.fea_m_embedder = nn.Sequential(
            ScenarioProcessingModuleWithoutAggregation(
                dim_in=config.SAA_attention_dim,
                dim_out=config.SAA_attention_dim,
                num_heads=1,
                num_inds=16,
                ln=False,
            )
        )

        self.fea_pair_embedder = nn.Sequential(
            ScenarioProcessingModuleWithoutAggregation(
                dim_in=config.SAA_attention_dim,
                dim_out=config.SAA_attention_dim,
                num_heads=1,
                num_inds=16,
                ln=False,
            )
        )
        self.pair_linear = nn.Linear(8, config.SAA_attention_dim)

    def forward(
        self,
        fea_j,
        op_mask,
        candidate,
        fea_m,
        mch_mask,
        comp_idx,
        dynamic_pair_mask,
        fea_pairs,
        num_samples,
    ):
        sz_b, num_samples, M, _, J = comp_idx.size()

        fea_j_or = fea_j.clone()
        fea_j = fea_j.view(-1, num_samples, fea_j.shape[-1])
        fea_j = self.fea_j_linear(fea_j)
        fea_j = torch.cat(
            (
                fea_j_or[:, 0, :, :],
                self.fea_j_embedder(fea_j[:, 1:, :])
                .reshape(sz_b, num_samples - 1, -1, fea_j.shape[-1])
                .mean(dim=1),
            ),
            dim=-1,
        )

        fea_m_or = fea_m.clone()
        fea_m = fea_m.view(-1, num_samples, fea_m.shape[-1])
        fea_m = self.fea_m_linear(fea_m)
        fea_m = torch.cat(
            (
                fea_m_or[:, 0, :, :],
                self.fea_m_embedder(fea_m[:, 1:, :])
                .reshape(sz_b, num_samples - 1, -1, fea_m.shape[-1])
                .mean(dim=1),
            ),
            dim=-1,
        )
        op_mask = op_mask[:, 0, :, :]
        candidate = candidate[:, 0, :]
        mch_mask = mch_mask[:, 0, :, :]
        comp_idx = comp_idx[:, 0, :, :, :]
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )

        d = fea_j.size(-1)
        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = (
            Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs_or = fea_pairs[:, 0, :, :, :].view(sz_b, -1, self.pair_input_dim)
        fea_pairs_pt = fea_pairs[:, :, :, :, :]
        fea_pairs_pt = self.pair_linear(fea_pairs_pt)

        fea_pairs_cat = torch.cat(
            (
                fea_pairs_or,
                self.fea_pair_embedder(
                    fea_pairs_pt.permute(0, 2, 3, 1, 4).flatten(0, 2)[:, 1:, :]
                )
                .mean(dim=1)
                .view(sz_b, -1, self.SAA_attention_dim),
            ),
            dim=-1,
        )

        candidate_feature = torch.cat(
            (
                Fea_j_JC_serialized,
                Fea_m_serialized,
                Fea_Gj_input,
                Fea_Gm_input,
                fea_pairs_cat,
            ),
            dim=-1,
        )

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        dynamic_pair_mask = dynamic_pair_mask[:, 0, :, :]
        # masking incompatible op-mch pairs
        candidate_scores = torch.masked_fill(
            candidate_scores, mask=dynamic_pair_mask.view(sz_b, -1), value=float("-inf")
        )
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
