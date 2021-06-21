from .base_model import SummModel
import argparse
import os
import torch
import gzip
import json
import uuid
from nltk import word_tokenize
import sys
from .third_party.HMNet.Models.Trainers.HMNetTrainer import HMNetTrainer
from .third_party.HMNet.Utils.Arguments import Arguments

# TODO: test the dependencies that are needed to be installed
# TODO: let them copy-past HMNet or help them move it here
class HMNetModel(SummModel):
    # static variables
    model_name = "HMNET"
    is_extractive = False
    is_neural = True

    def __init__(self):
        super(HMNetModel, self).__init__()

        self.opt = self._parse_args()
        self.model = HMNetTrainer(self.opt)

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='HMNet: Pretrain or fine-tune models for HMNet model.')
        parser.add_argument('command', default='evaluate', help='Command: train/evaluate')
        parser.add_argument('conf_file', default='./hmnet/config/dialogue_conf', help='Path to the BigLearn conf file.')
        parser.add_argument('--PYLEARN_MODEL', help='Overrides this option from the conf file.')
        parser.add_argument('--master_port', help='Overrides this option default', default=None)
        parser.add_argument('--cluster', help='local, philly or aml', default='local')
        parser.add_argument('--dist_init_path', help='Distributed init path for AML', default='./tmp')
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument('--no_cuda', action='store_true', help="Disable cuda.")
        parser.add_argument('--config_overrides', help='Override parameters on config, VAR=val;VAR=val;...')

        cmdline_args = parser.parse_args()
        command = cmdline_args.command
        conf_file = cmdline_args.conf_file
        conf_args = Arguments(conf_file)
        opt = conf_args.readArguments()

        if cmdline_args.config_overrides:
            for config_override in cmdline_args.config_overrides.split(';'):
                config_override = config_override.strip()
                if config_override:
                    var_val = config_override.split('=')
                    assert len(var_val) == 2, f"Config override '{var_val}' does not have the form 'VAR=val'"
                    conf_args.add_opt(opt, var_val[0], var_val[1], force_override=True)

        opt['cuda'] = torch.cuda.is_available() and not cmdline_args.no_cuda
        opt['confFile'] = conf_file
        if 'datadir' not in opt:
            opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is
        opt['basename'] = os.path.basename(conf_file)  # conf_file specifies where the name of save folder is
        opt['command'] = command

        # combine cmdline_args into opt dictionary
        for key, val in cmdline_args.__dict__.items():
            # if val is not None and key not in ['command', 'conf_file']:
            if val is not None:
                opt[key] = val

        return opt

    def summarize(self, corpus, queries=None):
        if corpus is None:
            print("Corpus not found, using default AMI dataset.")
            # TODO: where do we save checkpoints
        else:
            print(f"HMNet model: processing document of {corpus.__len__()} samples")
            # transform the original dataset to "dialogue" input
            # we only use test set path for evaluation
            self._set_path(self.opt)
            self._preprocess(corpus, os.path.join(os.path.dirname(self.opt['test_file']), 'test'))

        return self._evaluate()

    def _set_path(self, opt, name=None):
        # generate an uuid for the dataset
        if name is None:
            name = uuid.uuid1()
        dir_path = "./hmnet/preprocessed_datasets/{}".format(name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # TODO: set the role dict here
        # TODO: download the pretraining model here
        for split in ["TRAIN", "VALID", "TEST"]:
            dataset_path = os.path.join(dir_path, split.lower())
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            meta_path = os.path.join(dir_path, "{}.json".format(split.lower()))
            new_json = [{
                "source":
                {
                    "dataset": dataset_path
                },
                "task": "meeting",
                "name": name
            }]

            opt["{}_FILE".format(split)] = meta_path
            with open(meta_path, 'w', encoding='utf-8') as file:
                json.dump(meta_path, new_json)

    def _evaluate(self):
        if self.opt['rank'] == 0:
            self.model.log('-----------------------------------------------')
            self.model.log("Evaluating model ... ")

        self.model.set_up_model()

        eval_dataset = 'test'
        batch_generator_eval = self.model.get_batch_generator(eval_dataset)
        self.model.task.evaluator.reset_best_score(set_high=True)
        result, score, got_better_score = self.model.task.evaluator.eval_batches(
            self.model.module, batch_generator_eval, self.model.saveFolder, eval_dataset)

        if self.opt['rank'] == 0:
            self.model.log("{0} results breakdown\n{1}".format(
                eval_dataset, result))
        return result # TODO: this result is not what we want

    def _preprocess(self, corpus, test_path):
        # TODO: add role vector
        for i, sample in enumerate(corpus):
            new_sample = {'id': i, 'meeting': [], 'summary': []}
            if isinstance(sample.source, str):
                raise RuntimeError("Error: the input of HMNet should be dialogues, rather than documents.")

            # add query to the top of the turns
            if sample.query is not None:
                tokenized_turn = word_tokenize(sample.query)
                new_sample['meeting'].append({
                    'speaker': 'query',
                    'role': 'query',  # TODO: refine this role
                    'utt': {
                        'word': tokenized_turn,
                        'pos_id': [0] * len(tokenized_turn),  # TODO: add Scipy pos tag
                        'ent_id': [73] * len(tokenized_turn)
                    }
                })

            # add all the turns one by one
            for turn in sample.source:
                turn = [x.strip() for x in turn.split(':')]
                tokenized_turn = word_tokenize(turn[1])
                new_sample['meeting'].append({
                    'speaker': turn[0],
                    'role': turn[0], # TODO: refine this role
                    'utt': {
                        'word': tokenized_turn,
                        'pos_id': [0]*len(tokenized_turn), # TODO: add Scipy pos tag
                        'ent_id': [73]*len(tokenized_turn)
                    }
                })
            new_sample['summary'].append(corpus.summary)

            #save to the gzip
            file_path = os.path.join(test_path, "split_{}.jsonl.gz".format(i))
            with gzip.open(file_path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(new_sample)+'\n')


    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = ("A HMNet model finetuned on CNN-DM dataset for summarization.\n\n"
                        "Strengths:\n - High performance on dialogue summarization task.\n\n"
                        "Weaknesses:\n - Not suitable for datasets other than dialogues.\n\n"
                        "Initialization arguments:\n "
                        " - `corpus`: Unlabelled corpus of documents.\n")
        print(f"{basic_description} \n {'#' * 20} \n {more_details}")
