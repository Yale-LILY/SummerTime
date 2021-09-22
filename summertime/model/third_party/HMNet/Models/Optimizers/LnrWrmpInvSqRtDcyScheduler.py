# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from torch.optim.lr_scheduler import LambdaLR


class LnrWrmpInvSqRtDcyScheduler(LambdaLR):
    """Inverse Square Root learning rate schedule used in T5"""

    def __init__(self, optimizer, warmup_steps, warmup_init_lr, warmup_end_lr):
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
        super(LnrWrmpInvSqRtDcyScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=-1
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (self.warmup_init_lr + step * self.lr_step) / self.warmup_end_lr
        else:
            return 1.0 / float(math.sqrt(step / float(self.warmup_steps)))

    def get_last_lr(self):
        return self.get_lr()
