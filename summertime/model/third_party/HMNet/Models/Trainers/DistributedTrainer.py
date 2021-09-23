# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

from pkg_resources import parse_version
from summertime.model.third_party.HMNet.Models.Trainers.BaseTrainer import BaseTrainer
from summertime.model.third_party.HMNet.Utils.GeneralUtils import bcolors
from summertime.model.third_party.HMNet.Utils.distributed import distributed


class DistributedTrainer(BaseTrainer):
    def __init__(self, opt):
        super().__init__(opt)

        self.seed = int(self.opt["SEED"]) if "SEED" in self.opt else 0

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        (
            self.opt["device"],
            _,
            self.opt["world_size"],
            self.opt["local_size"],
            self.opt["rank"],
            self.opt["local_rank"],
            _,
            self.opt["run"],
        ) = distributed(opt, not self.use_cuda)

        self.getSaveFolder()
        self.opt["logFile"] = f"log_{self.opt['rank']}.txt"
        self.saveConf()

        self.high_pytorch_version = parse_version(torch.__version__) >= parse_version(
            "1.2.0"
        )
        if self.opt["rank"] == 0:
            print(
                bcolors.OKGREEN,
                torch.__version__,
                bcolors.ENDC,
                "is",
                "high" if self.high_pytorch_version else "low",
            )

        if self.use_cuda:
            # torch.cuda.manual_seed_all(self.seed)
            # ddp: only set seed on GPU associated with this process
            torch.cuda.manual_seed(self.seed)

        # ddp: print stats and update learning rate
        if self.opt["rank"] == 0:
            print(
                "Number of GPUs is",
                bcolors.OKGREEN,
                self.opt["world_size"],
                bcolors.ENDC,
            )
            # print('Boost learning rate from', bcolors.OKGREEN, self.opt['START_LEARNING_RATE'], bcolors.ENDC, 'to',
            #     bcolors.OKGREEN, self.opt['START_LEARNING_RATE'] * self.opt['world_size'], bcolors.ENDC)
            print(
                "Effective batch size is increased from",
                bcolors.OKGREEN,
                self.opt["MINI_BATCH"],
                bcolors.ENDC,
                "to",
                bcolors.OKGREEN,
                self.opt["MINI_BATCH"] * self.opt["world_size"],
                bcolors.ENDC,
            )

        self.grad_acc_steps = 1
        if "GRADIENT_ACCUMULATE_STEP" in self.opt:
            if self.opt["rank"] == 0:
                print(
                    "Gradient accumulation steps =",
                    bcolors.OKGREEN,
                    self.opt["GRADIENT_ACCUMULATE_STEP"],
                    bcolors.ENDC,
                )
                # print('Boost learning rate from', bcolors.OKGREEN, self.opt['START_LEARNING_RATE'], bcolors.ENDC, 'to',
                # bcolors.OKGREEN, self.opt['START_LEARNING_RATE'] * self.opt['world_size'] * self.opt['GRADIENT_ACCUMULATE_STEP'], bcolors.ENDC)
                print(
                    "Effective batch size =",
                    bcolors.OKGREEN,
                    self.opt["MINI_BATCH"]
                    * self.opt["world_size"]
                    * self.opt["GRADIENT_ACCUMULATE_STEP"],
                    bcolors.ENDC,
                )
            self.grad_acc_steps = int(self.opt["GRADIENT_ACCUMULATE_STEP"])
        # self.opt['START_LEARNING_RATE'] *= self.opt['world_size'] * self.grad_acc_steps

    def tb_log_scalar(self, name, value, step):
        if self.opt["rank"] == 0:
            if self.tb_writer is None:
                self.tb_writer = SummaryWriter(
                    os.path.join(self.saveFolder, "tensorboard")
                )
            self.tb_writer.add_scalar(name, value, step)

    def log(self, s):
        # When 'OFFICIAL' flag is set in the config file, the program does not output logs
        if self.is_official:
            return
        try:
            if self.logFileHandle is None:
                self.logFileHandle = open(
                    os.path.join(self.saveFolder, self.opt["logFile"]), "a"
                )
            self.logFileHandle.write(s + "\n")
        except Exception as e:
            print("ERROR while writing log file:", e)
        print(s)

    def getSaveFolder(self):
        runid = 1
        while True:
            saveFolder = os.path.join(
                self.opt["datadir"],
                self.opt["basename"] + "_conf~",
                "run_" + str(runid),
            )
            if not os.path.isdir(saveFolder):
                if self.opt["world_size"] > 1:
                    torch.distributed.barrier()
                if self.opt["rank"] == 0:
                    os.makedirs(saveFolder)
                self.saveFolder = saveFolder
                if self.opt["world_size"] > 1:
                    torch.distributed.barrier()
                print(
                    "Saving logs, model, checkpoint, and evaluation in "
                    + self.saveFolder
                )
                return
            runid = runid + 1

    def saveConf(self):
        if self.opt["rank"] == 0:
            super().saveConf()
