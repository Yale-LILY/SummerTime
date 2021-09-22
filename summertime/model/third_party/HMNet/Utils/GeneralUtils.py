# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import re
import logging
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import unicodedata
import sys
from torch.autograd import Variable

from .Constants import *

logger = logging.getLogger(__name__)


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n / decay)  # exponential decay over 100 updates
            self.sum = alpha * self.sum + (1 - alpha) * val * n
            self.count = alpha * self.count + (1 - alpha) * n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count


class BaseBatchGen:
    """
    This is a base class for batch generators that use infinibatch.

    The interfaces below are required to work with LegacyTask.

    For new tasks, the interfaces are not restricted (the methods and their signatures don't
    have to be same as the base class). They should have minimum assumption or dependency
    on other components in the system. Task classes can use them accordingly.
    """

    def __init__(
        self,
        task_args,
        dataset_label,
        model_config=None,
        tokenizer=None,
        world_size=1,
        rank=0,
        seed=None,
    ):
        """
        Args:
            task_args (dict): dictionary records arguments
            dataset_label (str): 'train', 'dev' or 'test'
            model_config: config of the model
            tokenizer: tokenizer used to process text
            world_size (int): total number of GPUs
            rank (int): order of current GPU
            seed (int): random seed
        """
        self.opt = task_args
        self.dataset_label = dataset_label
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.evaluation = dataset_label in ["dev", "test"]

        self._iter = None

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        raise NotImplementedError()

    @property
    def iterator(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __iter__(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __next__(self):
        return next(self._iter)


def move_batch_to_device(batch, device):
    """
    Move the batch to the device.
    It should be called before feeding the batch to the model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
        device (torch.device): device to move the batch to
    Returns:
        return_batch: same type as the input batch with internal tensors moved to device
    """
    if torch.is_tensor(batch):
        return_batch = batch.to(device)
    elif isinstance(batch, list):
        return_batch = [move_batch_to_device(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return_batch = tuple(move_batch_to_device(t, device) for t in batch)
    elif isinstance(batch, dict):
        return_batch = {}
        for k in batch:
            return_batch[k] = move_batch_to_device(batch[k], device)
    else:
        logger.debug(
            f"Can not move type {type(batch)} to device. Skipping it in the batch."
        )
        return_batch = batch

    return return_batch
