# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Task:
    """
    This class is the ensemble of two classes: BatchGen and Eval.
    The `setup_task` function defines tasks w.r.t the three components based
    on the `task_name`.
    """

    def __init__(self, batch_gen, evaluator):
        self.batch_gen = batch_gen
        self.evaluator = evaluator

    @classmethod
    def setup_task(cls, task_name, opt, save_dir):

        if task_name == "HMNet":
            from summertime.model.third_party.HMNet.Utils.HMNet.InfinibatchLoader import (
                HMNetBatchGen,
            )

            batch_gen = HMNetBatchGen
            from summertime.model.third_party.HMNet.Evaluation.ROUGEEval import ROUGEEval

            evaluator = ROUGEEval(opt["datadir"], save_dir, opt)
        else:
            assert False
            print("ERROR: Task {} not defined".format(task_name))

        return cls(batch_gen, evaluator)
