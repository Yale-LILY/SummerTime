# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLECriterion(nn.Module):
    """
    Class to define loss give input, model output and groundtruth
    """

    def __init__(self, opt, module):
        super().__init__()
        self.opt = opt
        self.ignore_index = (
            self.opt["IGNORE_INDEX"]
            if "IGNORE_INDEX" in self.opt
            else module.tokenizer.pad_token_id
        )

    def forward(self, vocab_logprob, batch):
        extended_vocab_size = vocab_logprob.shape[2]
        y = batch["decoder_input_ids"]

        if "USE_BOS_TOKEN" in self.opt:
            y = y[:, 1:]

        if "USE_EOS_TOKEN" in self.opt:
            vocab_logprob = vocab_logprob[:, :-1, :]

        loss = F.nll_loss(
            vocab_logprob.contiguous().view(-1, extended_vocab_size),
            y.contiguous().view(-1),
            ignore_index=self.ignore_index,
        )

        return loss
