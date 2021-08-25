# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


def set_dropout_prob(p):
    global dropout_p
    dropout_p = p


def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = Variable(
        1.0
        / (1 - p)
        * torch.bernoulli((1 - p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)),
        requires_grad=False,
    )
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)
