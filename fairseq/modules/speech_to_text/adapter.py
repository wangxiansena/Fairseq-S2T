import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import LayerNorm

logger = logging.getLogger(__name__)


class CTCCompressStrategy:
    @staticmethod
    def avg(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device)

    @staticmethod
    def weighted(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

    @staticmethod
    def softmax(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(
                    prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]], dtype=torch.float32
                )
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix


class Adapter(nn.Module):
    def __init__(self, dim, adapter_type, dictionary_size, embed_tokens=None, strategy=None):
        super().__init__()

        dim = dim
        self.adapter_type = adapter_type
        self.cal_linear = False
        self.cal_context = False

        if self.adapter_type in ["linear", "league", "gated_league", "gated_league2"]:
            self.cal_linear = True
            self.linear_adapter = nn.Sequential(
                nn.Linear(dim, 2 * dim),
                nn.ReLU(),
                nn.Linear(2 * dim, dim),
                LayerNorm(dim),
            )

        if self.adapter_type in ["context", "league", "gated_league", "gated_league2", "inter_league"]:
            self.cal_context = True
            self.embed_adapter = nn.Linear(dim, dictionary_size, bias=False)    # reverse for initialization
            nn.init.normal_(self.embed_adapter.weight, mean=0, std=dim ** -0.5)
            self.embed_norm = strategy.get("embed_norm", False)
            if self.embed_norm:
                self.embed_ln = LayerNorm(dim)
            if embed_tokens is not None:
                self.embed_adapter.weight = embed_tokens.weight

        if self.adapter_type == "gated_league":
            self.gate_linear = nn.Linear(2 * dim, dim)
        elif self.adapter_type == "gated_league2":
            self.gate_linear1 = nn.Linear(dim, dim)
            self.gate_linear2 = nn.Linear(dim, dim)

        # additional strategy
        if self.adapter_type == "shrink":
            assert strategy is not None
            ctc_compress_strategy = strategy.get("ctc_compress_strategy", "avg")
            self.ctc_compress = getattr(CTCCompressStrategy, ctc_compress_strategy)
            logger.info("CTC Compress Strategy: %s" % ctc_compress_strategy)

        if self.cal_context:
            self.distribution_cutoff = strategy.get("distribution_cutoff", None)
            self.distribution_temperature = strategy.get("ctc_temperature", 1.0)
            self.gumbel = strategy.get("gumbel", False)
            self.distribution_hard = strategy.get("distribution_hard", False)
            self.ground_truth_ratio = strategy.get("gt_ratio", 0)
            self.drop_prob = strategy.get("drop_prob", 0)

            if self.distribution_cutoff is not None:
                logger.info("Distribution cutoff: %d" % self.distribution_cutoff)
            if self.distribution_temperature != 1.0:
                logger.info("Temperature: %f" % self.distribution_temperature)
            if self.gumbel:
                logger.info("Gumbel softmax.")
            if self.distribution_hard:
                logger.info("Hard distribution.")
            if self.drop_prob != 0:
                logger.info("Drop probability: %f" % self.drop_prob)

        self.out_norm = strategy.get("out_norm", False)
        if self.out_norm:
            self.out_ln = LayerNorm(dim)

    def forward(self, x, padding=None, oracle=None, oracle_mask=None):
        representation, logit = x
        seq_len, bsz, dim = representation.size()

        linear_out = None
        soft_out = None
        if self.cal_linear:
            linear_out = self.linear_adapter(representation)
        if self.cal_context:
            if self.training and self.gumbel:
                distribution = F.gumbel_softmax(logit, tau=self.distribution_temperature, hard=self.distribution_hard)
            else:
                distribution = F.softmax(logit / self.distribution_temperature, dim=-1)

            vocab_size = distribution.size(-1)
            distribution = distribution.contiguous().view(-1, vocab_size)
            org_distribution = distribution

            if self.distribution_cutoff is not None:
                cutoff = min(int(self.distribution_cutoff), vocab_size - 1)

                # threshold = org_distribution.sort(dim=-1, descending=True)[0][:, :, cutoff:cutoff+1]
                # distribution = torch.where(
                #     org_distribution > threshold, org_distribution, torch.zeros_like(org_distribution)
                # )

                # threshold = org_distribution.sort(dim=-1, descending=True)[0][:, :, :cutoff].sum(-1, keepdim=True)
                # distribution = torch.where(
                #     threshold > 0.9, org_distribution, torch.zeros_like(org_distribution)
                # )
                # distribution = distribution.view(-1, vocab_size)

                distribution[:, 0] = 0
                distribution = distribution / distribution.sum(-1, keepdim=True)

            if self.ground_truth_ratio > 0 and oracle is not None:
                oracle = oracle.unsqueeze(-1)
                oracle_one_hot = (oracle == torch.arange(vocab_size, device=oracle.device).unsqueeze(0)).\
                    to(distribution.dtype).transpose(0, 1)
                oracle_mask = oracle_mask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, vocab_size)
                modify_dist = oracle_mask * oracle_one_hot + ~oracle_mask * org_distribution
                soft_out = torch.mm(modify_dist.view(-1, vocab_size), self.embed_adapter.weight).view(seq_len, bsz, -1)
            else:
                soft_out = torch.mm(distribution, self.embed_adapter.weight).view(seq_len, bsz, -1)

            if self.embed_norm:
                soft_out = self.embed_ln(soft_out)

        if self.adapter_type == "linear":
            out = linear_out

        elif self.adapter_type == "context":
            out = soft_out

        elif self.adapter_type == "league":
            if self.training and self.drop_prob > 0 and torch.rand(1).uniform_() < self.drop_prob:
                if torch.rand(1).uniform_() < 0.5:
                    out = linear_out
                else:
                    out = soft_out
            else:
                out = linear_out + soft_out

        elif self.adapter_type == "gated_league":
            coef = (self.gate_linear(torch.cat([linear_out, soft_out], dim=-1))).sigmoid()
            out = coef * linear_out + (1 - coef) * soft_out

        elif self.adapter_type == "inter_league":
            out = representation + soft_out

        elif self.adapter_type == "none":
            out = representation

        elif self.adapter_type == "shrink":
            if self.training and self.gumbel:
                distribution = F.gumbel_softmax(logit, tau=self.distribution_temperature, hard=self.distribution_hard)
            else:
                distribution = F.softmax(logit / self.distribution_temperature, dim=-1)

            lengths = (~padding).long().sum(-1)
            with torch.no_grad():
                batch_predicted = []
                prob_ctc = distribution.transpose(0, 1)  # T x B x D -> B x T x D
                for b in range(prob_ctc.shape[0]):
                    predicted = prob_ctc[b][: lengths[b]].argmax(-1).tolist()
                    batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                new_lengths = [len(p) for p in batch_predicted]
                weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths,
                                                   prob_ctc.dtype, prob_ctc.device)

            # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
            representation = representation.permute(1, 2, 0)
            compressed_output = representation.bmm(weights_matrix).type_as(representation)  # B x C x T'
            out = compressed_output.permute(2, 0, 1)

            out_lengths = lengths.new(new_lengths)
            padding = lengths_to_padding_mask(out_lengths)

        else:
            out = None
            logging.error("Unsupported adapter type: {}.".format(self.adapter_type))

        if self.out_norm:
            out = self.out_ln(out)

        return out, padding
