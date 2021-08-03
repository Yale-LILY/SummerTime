# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

class BaseTrainer():
    def __init__(self, opt):
        self.opt = opt
        if self.opt['cuda'] == True:
            self.use_cuda = True
            print('Using Cuda\n')
        else:
            self.use_cuda = False
            print('Using CPU\n')

        self.is_official = 'OFFICIAL' in self.opt
        self.opt['logFile'] = 'log.txt'
        self.saveFolder = None
        self.logFileHandle = None
        self.tb_writer = None

    def log(self, s):
        # In official case, the program does not output logs
        if self.is_official:
            return
        try:
            if self.logFileHandle is None:
                self.logFileHandle = open(os.path.join(self.saveFolder, self.opt['logFile']), 'a')
            self.logFileHandle.write(s + '\n')
        except Exception as e:
            print('ERROR while writing log file:', e)
        print(s)

    def getSaveFolder(self):
        runid = 1
        while True:
            saveFolder = os.path.join(self.opt['datadir'], self.opt['basename']+'_conf~', 'run_' + str(runid))
            if not os.path.exists(saveFolder):
                self.saveFolder = saveFolder
                os.makedirs(self.saveFolder)
                print('Saving logs, model and evaluation in ' + self.saveFolder)
                return
            runid = runid + 1

    # save copy of conf file
    def saveConf(self):
        # with open(self.opt['confFile'], encoding='utf-8') as f:
        #     with open(os.path.join(self.saveFolder, 'conf_copy.tsv'), 'w', encoding='utf-8') as fw:
        #         for line in f:
        #             fw.write(line)
        with open(os.path.join(self.saveFolder, 'conf_copy.tsv'), 'w', encoding='utf-8') as fw:
            for k in self.opt:
                fw.write('{0}\t{1}\n'.format(k, self.opt[k]))

    def train(self):
        pass

    def load(self):
        pass
