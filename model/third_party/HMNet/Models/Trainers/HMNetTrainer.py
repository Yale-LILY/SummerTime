# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from datetime import datetime
import os
import sys
import importlib
import json
import random
import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.third_party.HMNet.Models.Trainers.DistributedTrainer import DistributedTrainer
from model.third_party.HMNet.Models.Trainers.Tasks import Task
from model.third_party.HMNet.Utils.GeneralUtils import AverageMeter, BaseBatchGen, bcolors

from model.third_party.HMNet.DataLoader import iterators


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class WrappedModel(nn.Module):
    def __init__(self, model, criterion):
        super(WrappedModel, self).__init__()
        self.add_module('model', model)
        self.add_module('criterion', criterion)

    def forward(self, batch):
        output = self.model(batch)
        loss = self.criterion(output, batch)
        return loss


class HMNetTrainer(DistributedTrainer):
    '''
        The trainer class for HMNet model training (pre-train and fine-tune.)
        Its train() and eval() methods are intended to directly called to
        start training and evaluation respectively.

        Before running, the trainer must contain proper Task, Criterion, and Optimizer
        instances.

    '''

    def __init__(self, opt):
        super().__init__(opt)
        self.task = Task.setup_task(self.opt['TASK'], self.opt, self.saveFolder)

    def is_gradient_accumulation_boundary(self):
        return (self.updates + 1) % self.grad_acc_steps == 0

    def get_batch_generator(self, dataset_label):
        batch_generator = self.task.batch_gen(self.opt,
                                              dataset_label=dataset_label,
                                              model_config=self.module.config,
                                              tokenizer=self.module.tokenizer,
                                              world_size=self.opt['world_size'],
                                              rank=self.opt['rank'],
                                              seed=self.seed)
        if isinstance(batch_generator, BaseBatchGen):
            # If it is a wrapper class of an infinibatch iterator,
            # get the internal infnitibatch iterator.
            batch_generator = batch_generator.iterator
        self.log(f"Loaded data on rank {self.opt['rank']}.")
        return batch_generator

    def set_up_model(self):
        # instantiate module (tokenizer should be contained in module as self.module.tokenizer)
        try:
            model_module = importlib.import_module(
                'model.third_party.HMNet.Models.Networks.' + self.opt['MODEL'])
            model_class = getattr(model_module, self.opt['MODEL'])
            self.module = model_class(self.opt)
        except Exception as e:
            self.log(e)
            self.log("ERROR: Model {} is unknown".format(self.opt['MODEL']))
            assert False

        # calculate total trainable parameters
        pytorch_total_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        self.log("Total trainable parameters: {}".format(pytorch_total_params))

        # instantiate criterion
        try:
            criterion_module = importlib.import_module(
                'model.third_party.HMNet.Models.Criteria.' + self.opt['CRITERION'])
            criterion_class = getattr(criterion_module, self.opt['CRITERION'])
            self.criterion = criterion_class(self.opt, self.module)
        except Exception as e:
            self.log(e)
            self.log("ERROR: Criterion {} is unknown".format(
                self.opt['CRITERION']))
            assert False

        self.module.to(self.opt['device'])

    def get_optimizer_params_config(self, optimizer_class):
        optimizer_parameters = {}
        sig = inspect.signature(optimizer_class)
        for param_name in sig.parameters.keys():
            if param_name == 'lr':
                optimizer_parameters[param_name] = self.opt['START_LEARNING_RATE']
            if param_name not in ['params', 'lr'] and param_name.upper() in self.opt:
                optimizer_parameters[param_name] = self.opt[param_name.upper()]
        return optimizer_parameters

    def get_lr_scheduler_params_config(self, lr_scheduler_class):
        lr_scheduler_parameters = {}
        sig = inspect.signature(lr_scheduler_class)
        for param_name in sig.parameters.keys():
            if param_name not in ['optimizer'] and param_name.upper() in self.opt:
                lr_scheduler_parameters[param_name] = self.opt[param_name.upper()]
        return lr_scheduler_parameters

    def set_up_optimizer_and_lr_scheduler(self):

        parameters = self.module.get_training_parameters()

        # instantiate optimizer
        try:  # first try pytorch native optimizer
            optimizer_class = getattr(optim, self.opt['OPTIMIZER'])
            self.log('Using pytorch native optimizier: {}'.format(
                self.opt['OPTIMIZER']))
        except:
            try:  # then try custom optimizer inside Models.Optimizers
                optimizer_module = importlib.import_module(
                    'model.third_party.HMNet.Models.Optimizers.' + self.opt['OPTIMIZER'])
                optimizer_class = getattr(
                    optimizer_module, self.opt['OPTIMIZER'])
                self.log('Using custom optimizer: {}'.format(
                    self.opt['OPTIMIZER']))
            except Exception as e:
                self.log(e)
                self.log("ERROR: Optimizer {} is unknown".format(
                    self.opt['OPTIMIZER']))
                assert False

        optimizer_parameters = self.get_optimizer_params_config(optimizer_class)
        self.log(f"Optimizer parameters: {optimizer_parameters}")
        self.optimizer = optimizer_class(parameters, **optimizer_parameters)
        self.optimizer.zero_grad()

        # instantiate lr scheduler
        try:  # first look for pytorch native lr scheduler
            lr_scheduler_class = getattr(lr_scheduler, self.opt['LR_SCHEDULER'])
            self.log('Using pytorch native lr scheduler: {}'.format(
                self.opt['LR_SCHEDULER']))
        except:
            try:  # then look for custom lr scheduler inside Models.Optimizers
                lr_scheduler_module = importlib.import_module(
                    'model.third_party.HMNet.Models.Optimizers.' + self.opt['LR_SCHEDULER'])
                lr_scheduler_class = getattr(
                    lr_scheduler_module, self.opt['LR_SCHEDULER'])
                self.log('Using custom lr scheduler: {}'.format(
                    self.opt['LR_SCHEDULER']))
            except Exception as e:
                self.log(e)
                self.log("ERROR: LR Scheduler {} is unknown".format(
                    self.opt['LR_SCHEDULER']))
                assert False

        lr_scheduler_parameters = self.get_lr_scheduler_params_config(lr_scheduler_class)
        self.log(f"Lr scheduler parameters: {lr_scheduler_parameters}")
        self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_parameters)

    def initialize_fp16_DDP(self):
        '''
        Wrap the module and criterion to a single network, then depending on the settings,
        wrap the network with apex amp module for fp16 training, and wrap the network with
        pytorch DDP module for distributed data parallel training
        '''
        self.network = WrappedModel(self.module, self.criterion)
        self.network.to(self.opt['device'])

        if self.opt['fp16']:
            from apex import amp
            self.network, self.optimizer = amp.initialize(
                self.network, self.optimizer, opt_level=self.opt['fp16_opt_level'])

        if self.opt['world_size'] > 1:
            self.network = torch.nn.parallel.DistributedDataParallel(self.network,
                                                                        device_ids=[
                                                                            self.opt['local_rank']],
                                                                        output_device=self.opt['local_rank'],
                                                                        find_unused_parameters=True
                                                                    )
            self.log(f"Wrapped model with DDP on rank {self.opt['rank']}.")
            assert self.module is self.network.module.model
        else:
            assert self.module is self.network.model

    def eval(self):
        if self.opt['rank'] == 0:
            self.log('-----------------------------------------------')
            self.log("Evaluating model ... ")
        self.set_up_model()

        for eval_dataset in ['dev', 'test']:
            batch_generator_eval = self.get_batch_generator(eval_dataset)

            self.task.evaluator.reset_best_score(set_high=True)
            result, score, got_better_score = self.task.evaluator.eval_batches(
                self.module, batch_generator_eval, self.saveFolder, eval_dataset)
            if self.opt['rank'] == 0:
                self.log("{0} results breakdown\n{1}".format(
                    eval_dataset, result))

    def eval_return_results(self):
        if self.opt['rank'] == 0:
            self.log('-----------------------------------------------')
            self.log("Evaluating model ... ")
        self.set_up_model()

        for eval_dataset in ['test']:
            batch_generator_eval = self.get_batch_generator(eval_dataset)

            self.task.evaluator.reset_best_score(set_high=True)
            result, score, got_better_score = self.task.evaluator.eval_batches(
                self.module, batch_generator_eval, self.saveFolder, eval_dataset)
            if self.opt['rank'] == 0:
                self.log("{0} results breakdown\n{1}".format(
                    eval_dataset, result))
        return result

    def train(self):
        self.log(f"train on rank {self.opt['rank']}")
        if self.opt['rank'] == 0:
            self.log('-----------------------------------------------')
            self.log("Initializing model...")

        self.set_up_model()  # setup self.module as original model
        self.network = None
        self.train_batch_generator = self.get_batch_generator('train')
        if isinstance(self.train_batch_generator, iterators.CheckpointableIterator):
            # training batch generator is infinite
            self.updates_per_epoch = self.opt['UPDATES_PER_EPOCH']
        else:
            self.updates_per_epoch = len(self.train_batch_generator)
        self.updates = 0
        self.optim_steps = 0
        self.start_epoch_idx = 0
        self.start_batch_idx = 0

        self.set_up_optimizer_and_lr_scheduler()
        self.initialize_fp16_DDP()
        if 'RESUME' in self.opt:
            # Resume complete training states, including optimizer, lr_scheduler, train batch generator, and updates count
            # from the checkpoint location indicated in a .json file
            self.load_checkpoint()

        ######################
        # Start the main loop
        ######################

        numEpochs = self.opt['MAX_NUM_EPOCHS']
        self.train_loss = AverageMeter()  # track the average training loss
        self.acc_loss = 0.
        # after every 'SAVE_PER_UPDATE_NUM' updates, it will save a checkpoint by setting save_a_checkpoint to True temporarily
        save_a_checkpoint = False
        for epoch in range(self.start_epoch_idx, numEpochs):
            self.current_epoch_idx = epoch
            self.log('Epoch {}'.format(epoch))

            startTime = datetime.now()

            for batch_idx, batch in enumerate(self.train_batch_generator):
                if self.current_epoch_idx == self.start_epoch_idx:
                    if isinstance(self.train_batch_generator, iterators.CheckpointableIterator):
                        batch_idx += self.start_batch_idx
                    elif batch_idx < self.start_batch_idx:
                        continue
                self.current_batch_idx = batch_idx

                # after every 'SAVE_PER_UPDATE_NUM' updates, save a checkpoint
                if ('SAVE_PER_UPDATE_NUM' in self.opt) and (self.updates + 1) % self.opt['SAVE_PER_UPDATE_NUM'] == 0:
                    # Make sure the next update is going to update the weights and zero the gradients, then we can checkpoint
                    assert self.is_gradient_accumulation_boundary()
                    save_a_checkpoint = True

                # update
                self.update(batch)

                if save_a_checkpoint:
                    # evaluate at the checkpointed moment, and log the results
                    if self.task.evaluator is not None:
                        evaluate_label = 'update_' + str(self.updates)
                        eval_dataset = 'dev'
                        batches = self.get_batch_generator(eval_dataset)
                        result, score, got_better_score = self.task.evaluator.eval_batches(
                            self.module, batches, self.saveFolder, evaluate_label)
                        self.tb_log_scalar('Eval/score', score, self.updates)
                        if got_better_score:
                            self.log("Got new better score on rank-{0} evaluator, at updates {1}".format(
                                self.opt['rank'], self.updates))
                        self.log("Updates {0} - {1}: Current Score: {2:.3f} (best Score: {3:.3f})".format(
                            self.updates, eval_dataset, score, self.task.evaluator.best_score))
                        self.log(
                            "Current results breakdown\n{0}".format(result))
                        self.log("Best results breakdown\n{0}".format(
                            self.task.evaluator.best_res))
                    # save complete training states, including model weights, optimizer, lr_scheduler, batch generator, and updates count
                    self.save_checkpoint(self.updates)
                    save_a_checkpoint = False

                # logging
                if (batch_idx % 10 == 0) or (epoch == 0 and batch_idx <= 50) or "DEBUG" in self.opt:
                    if self.opt['rank'] == 0:
                        batch_size = batch["encoder_input_ids"].shape[0]
                        self.log('epochs[{0:6}] updates[{1:6}] bsz[{2:d}] train loss[{3:.5f}] avg train loss[{4:.5f}] learning rate[{5:.5e}] remaining[{6}]'.format(
                            epoch, self.updates, batch_size, self.train_loss.val, self.train_loss.avg, self.lr_scheduler.get_lr()[0],
                            str((datetime.now() - startTime) / (batch_idx + 1) * (self.updates_per_epoch - batch_idx - 1)).split('.')[0]))

                        self.tb_log_scalar('Loss/train_val', self.train_loss.val, self.updates)
                        self.tb_log_scalar('Loss/train_avg', self.train_loss.avg, self.updates)
                        self.tb_log_scalar('Learning Rate/lr', self.lr_scheduler.get_lr()[0], self.updates)
                        
                # if "DEBUG" in self.opt and batch_idx > 200:  # exist early for DEBUG mode
                #     break

                if isinstance(self.train_batch_generator, iterators.CheckpointableIterator) and batch_idx + 1 == self.updates_per_epoch:
                    break

            self.log('This epoch takes' + str(datetime.now() - startTime))
            self.log("PROGRESS: {0:.2f}%".format(
                100.0 * (epoch + 1) / numEpochs))
            self.log('Config file is at ' + self.opt['confFile'])

            if "DEBUG" in self.opt:  # exist early for DEBUG mode
                break

    def update(self, batch):
        # forward loss, backward propagation, model update, and one step of optimization and lr scheduler
        self.network.train()
        # put the batch to the device
        # @TODO make this more general, maybe have a self.task.move_batch(batch, device)
        # so the trainer decides when and where to move batches, and task tells how
        if isinstance(batch, tuple):
            batch = tuple(t.to(self.opt['device']) for t in batch)
        elif isinstance(batch, list):
            batch = [t.to(self.opt['device']) for t in batch]
        elif isinstance(batch, dict):
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(self.opt['device'])
        else:
            assert torch.is_tensor(batch)
            batch = batch.to(self.opt['device'])

        # determine whether gradient sync can be skiped or not for this update
        skip_gradient_sync = False
        if self.opt['world_size'] > 1 and not self.is_gradient_accumulation_boundary():
            if not self.opt['fp16']:
                # https://krishansubudhi.github.io/deeplearning/2020/02/06/apex-gradient-accumulation.html
                # When using fp16, if we skip grad sync during grad accumulation, the grad sync at the 
                # grad accumulation boundary cannot properly sync the whole accumulated grad.
                # So with fp16 on, we have to sync even if it's not grad accumulation boundary.
                if self.high_pytorch_version:
                    skip_gradient_sync = True


        # forward
        if skip_gradient_sync:
            with self.network.no_sync():
                loss = self.network(batch)
        else:
            loss = self.network(batch)
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        self.acc_loss += loss
        #self.log(f"forward() done on rank {self.opt['rank']}")
        # print(loss.item())

        # backward
        def backward(loss_tensor):
            if self.opt['fp16']:
                from apex import amp
                with amp.scale_loss(loss_tensor, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_tensor.backward()

        if skip_gradient_sync:
            with self.network.no_sync():
                backward(loss)
        else:
            if "DEBUG" in self.opt and self.opt['rank'] == 0:
                self.log("Performing synchronized backward at step {0}".format(
                    self.optim_steps))
            backward(loss)
        #self.log(f"backward() done on rank {self.opt['rank']}")

        # step
        if self.is_gradient_accumulation_boundary():
            if self.opt['world_size'] > 1:
                # ddp: use all_reduce to sum up values of self.acc_loss over all processes
                # the operations happens in place (i.e., the value of self.acc_loss is replaced) and all processes received the updated value
                torch.distributed.all_reduce(
                    self.acc_loss, torch.distributed.ReduceOp.SUM)
                self.acc_loss /= self.opt['world_size']
            self.train_loss.update(self.acc_loss.data, 1)
            self.acc_loss = 0.
            if 'GRAD_CLIPPING' in self.opt:
                if self.opt['fp16']:
                    from apex import amp
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), self.opt['GRAD_CLIPPING'])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.opt['GRAD_CLIPPING'])
            self.optim_steps += 1
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        self.updates += 1
        #self.log(f"step() done on rank {self.opt['rank']}")

    def save_checkpoint(self, tag):
        '''
        Save complete training states, including model weights, optimizer, lr_scheduler,
        fp16 loss scaler, random state, batch generator, and updates count
        Also save a model with save_pretrained API for model transfer
        '''
        self.log('Saving checkpoint...')
        resume_epoch_idx = self.current_epoch_idx
        resume_batch_idx = self.current_batch_idx + 1
        if resume_batch_idx == self.updates_per_epoch:
            resume_batch_idx = 0
            resume_epoch_idx += 1

        if self.opt['fp16']:
            from apex import amp
        if self.opt['rank'] == 0:
            save_dir = os.path.join(self.saveFolder, str(tag))
            os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'training_states.pt')
            state = {'network': self.network.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                        'amp': amp.state_dict() if self.opt['fp16'] else None,
                        'optim_steps': self.optim_steps,
                        'updates': self.updates,
                        'updates_per_epoch': self.updates_per_epoch,
                        'start_epoch_idx': resume_epoch_idx,
                        'start_batch_idx': resume_batch_idx
                        }

            torch.save(state, save_path)
        if self.opt['world_size'] > 1:
            torch.distributed.barrier()
        save_dir = os.path.join(self.saveFolder, str(tag))
        assert os.path.isdir(save_dir)

        random_state_path = os.path.join(save_dir, 'random_state_rank_{:04d}'.format(self.opt['rank']))
        random_state = {'random': random.getstate(),
                        'numpy_random': np.random.get_state(),
                        'torch_random': torch.get_rng_state(),
                        'torch_cuda_random': torch.cuda.get_rng_state(device=self.opt['device']) if self.use_cuda else None
                        }
        torch.save(random_state, random_state_path)

        if isinstance(self.train_batch_generator, iterators.CheckpointableIterator):
            # save batch generators for all ranks
            batch_generator_file_path = os.path.join(
                save_dir, 'batch_generator_checkpoint_rank_{:04d}'.format(self.opt['rank']))
            batch_generator_state = self.train_batch_generator.getstate()
            torch.save(batch_generator_state, batch_generator_file_path)
        else:
            self.log(
                "Batch generator is not checkpointable. Cannot save to checkpoint.")

        if self.opt['rank'] == 0:
            self.module.save_pretrained(save_dir)

        if self.opt['rank'] == 0:
            # save the latest checkpoint location to json file
            checkpoint_location = {'checkpoint_tag': str(tag),
                                   'checkpoint_path': os.path.relpath(self.saveFolder, start=self.opt['datadir'])}
            json.dump(checkpoint_location, open(os.path.join(self.opt['datadir'], self.opt['basename']+'_resume_checkpoint.json'), 'w', encoding='utf-8'))
        self.log(f'Finished saving checkpoint and model to {save_dir}.')

    def load_model(self, model_path):
        # Load the model only, without any training states, using the from_pretrained API
        self.module = self.module.from_pretrained(model_path)
        self.module.to(self.opt['device'])

    def load_checkpoint(self):
        '''
        Load complete training states, including model weights, optimizer, lr_scheduler,
        fp16 loss scaler, random state, batch generator, and updates count
        '''
        try:
            # load the checkpoint location from json file
            checkpoint_location = json.load(
                open(os.path.join(self.opt['datadir'], self.opt['basename']+'_resume_checkpoint.json'), encoding='utf-8'))
            checkpoint_path = os.path.join(
                self.opt['datadir'], checkpoint_location['checkpoint_path'], checkpoint_location['checkpoint_tag'])
            tag = checkpoint_location['checkpoint_tag']
            if not os.path.isdir(checkpoint_path):
                if self.opt['rank'] == 0:
                    self.log("Checkpoint path {} not exist. Continue without loading checkpoint".format(
                        checkpoint_path))
                return
        except:
            if self.opt['rank'] == 0:
                self.log(f"Cannot find checkpoint path from {self.opt['basename']+'_resume_checkpoint.json'}.\n"
                         f"Make sure {os.path.join(self.opt['datadir'], self.opt['basename']+'_resume_checkpoint.json')} exists.\n"
                         f"Continue without loading checkpoint")
            return
        # save a copy of the resumed checkpoint location in the save folder of current run
        if self.opt['rank'] == 0:
            json.dump(checkpoint_location, open(os.path.join(self.saveFolder, 'resumed_checkpoint.json'), 'w', encoding='utf-8'))

        self.log(f'Loading checkpoint from {checkpoint_path}...')
        load_path = os.path.join(checkpoint_path, 'training_states.pt')
        state = torch.load(load_path, map_location=self.opt['device'])
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        if self.opt['fp16']:
            from apex import amp
            amp.load_state_dict(state['amp'])
        self.optim_steps = state['optim_steps']
        self.updates = state['updates']
        self.start_epoch_idx = state['start_epoch_idx']
        self.start_batch_idx = state['start_batch_idx']
        assert self.updates_per_epoch == state['updates_per_epoch']
        assert self.start_batch_idx < self.updates_per_epoch

        random_state_path = os.path.join(checkpoint_path, 'random_state_rank_{:04d}'.format(self.opt['rank']))
        random_state = torch.load(random_state_path, map_location='cpu')
        random.setstate(random_state['random'])
        np.random.set_state(random_state['numpy_random'])
        torch.set_rng_state(random_state['torch_random'])
        if self.use_cuda:
            torch.cuda.set_rng_state(random_state['torch_cuda_random'], device=self.opt['device'])

        if 'RESET_DATA_LOADER' not in self.opt and isinstance(self.train_batch_generator, iterators.CheckpointableIterator):
            batch_generator_file_path = os.path.join(
                checkpoint_path, 'batch_generator_checkpoint_rank_{:04d}'.format(self.opt['rank']))
            batch_generator_state = torch.load(
                batch_generator_file_path, map_location='cpu')
            self.train_batch_generator.setstate(batch_generator_state)
        else:
            self.log(
                "No need to resume batch generator or batch generator is not checkpointable. Didn't load from checkpoint.")
        self.log(f'Finished loading checkpoint from {checkpoint_path}.')
