from model.base_model import SummModel
import argparse
import os
import torch
import gzip
import json
from model.third_party.HMNet.Models.Trainers.HMNetTrainer import HMNetTrainer
from model.third_party.HMNet.Utils.Arguments import Arguments

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser'])
# tagger = nlp.get_pipe('tagger')
# ner = nlp.get_pipe('ner')
# POS = {w: i for i, w in enumerate([''] + list(tagger.labels))}
# ENT = {w: i for i, w in enumerate([''] + list(ner.move_names))}
# These two dicts are adapted from SpaCy 2.3.1, since HMNet's embedding for POS and ENT is fixed
POS = {'': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-RRB-': 5, '.': 6, ':': 7, 'ADD': 8, 'AFX': 9, 'CC': 10, 'CD': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'HYPH': 15, 'IN': 16, 'JJ': 17, 'JJR': 18, 'JJS': 19, 'LS': 20, 'MD': 21, 'NFP': 22, 'NN': 23, 'NNP': 24, 'NNPS': 25, 'NNS': 26, 'PDT': 27, 'POS': 28, 'PRP': 29, 'PRP$': 30, 'RB': 31, 'RBR': 32, 'RBS': 33, 'RP': 34, 'SYM': 35, 'TO': 36, 'UH': 37, 'VB': 38, 'VBD': 39, 'VBG': 40, 'VBN': 41, 'VBP': 42, 'VBZ': 43, 'WDT': 44, 'WP': 45, 'WP$': 46, 'WRB': 47, 'XX': 48, '_SP': 49, '``': 50}
ENT = {'': 0, 'B-ORG': 1, 'B-DATE': 2, 'B-PERSON': 3, 'B-GPE': 4, 'B-MONEY': 5, 'B-CARDINAL': 6, 'B-NORP': 7, 'B-PERCENT': 8, 'B-WORK_OF_ART': 9, 'B-LOC': 10, 'B-TIME': 11, 'B-QUANTITY': 12, 'B-FAC': 13, 'B-EVENT': 14, 'B-ORDINAL': 15, 'B-PRODUCT': 16, 'B-LAW': 17, 'B-LANGUAGE': 18, 'I-ORG': 19, 'I-DATE': 20, 'I-PERSON': 21, 'I-GPE': 22, 'I-MONEY': 23, 'I-CARDINAL': 24, 'I-NORP': 25, 'I-PERCENT': 26, 'I-WORK_OF_ART': 27, 'I-LOC': 28, 'I-TIME': 29, 'I-QUANTITY': 30, 'I-FAC': 31, 'I-EVENT': 32, 'I-ORDINAL': 33, 'I-PRODUCT': 34, 'I-LAW': 35, 'I-LANGUAGE': 36, 'L-ORG': 37, 'L-DATE': 38, 'L-PERSON': 39, 'L-GPE': 40, 'L-MONEY': 41, 'L-CARDINAL': 42, 'L-NORP': 43, 'L-PERCENT': 44, 'L-WORK_OF_ART': 45, 'L-LOC': 46, 'L-TIME': 47, 'L-QUANTITY': 48, 'L-FAC': 49, 'L-EVENT': 50, 'L-ORDINAL': 51, 'L-PRODUCT': 52, 'L-LAW': 53, 'L-LANGUAGE': 54, 'U-ORG': 55, 'U-DATE': 56, 'U-PERSON': 57, 'U-GPE': 58, 'U-MONEY': 59, 'U-CARDINAL': 60, 'U-NORP': 61, 'U-PERCENT': 62, 'U-WORK_OF_ART': 63, 'U-LOC': 64, 'U-TIME': 65, 'U-QUANTITY': 66, 'U-FAC': 67, 'U-EVENT': 68, 'U-ORDINAL': 69, 'U-PRODUCT': 70, 'U-LAW': 71, 'U-LANGUAGE': 72, 'O': 73}


class HMNetModel(SummModel):
    # static variables
    model_name = "HMNET"
    is_extractive = False
    is_neural = True
    is_dialogue_based = True

    def __init__(self):
        super(HMNetModel, self).__init__()
        self.root_path = self._get_root()
        self.opt = self._parse_args()
        self.model = HMNetTrainer(self.opt)

    def _get_root(self):
        root_path = os.getcwd()
        while 'model' not in os.listdir(root_path):
            root_path = os.path.dirname(root_path)
        root_path = os.path.join(root_path, 'model/dialogue')
        return root_path

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='HMNet: Pretrain or fine-tune models for HMNet model.')
        parser.add_argument('--command', default='evaluate', help='Command: train/evaluate')
        parser.add_argument('--conf_file',
                            default=os.path.join(self.root_path, 'hmnet/config/dialogue.conf'),
                            help='Path to the BigLearn conf file.')
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
        print(f"HMNet model: processing document of {corpus.__len__()} samples")
        # transform the original dataset to "dialogue" input
        # we only use test set path for evaluation
        data_folder = os.path.join(os.path.dirname(self.opt['datadir']),
                                   'ExampleRawData/meeting_summarization/AMI_proprec/test')

        self._create_datafolder(data_folder)
        self._preprocess(corpus, data_folder)

        # return self.model.eval()
        results = self._evaluate()

        return results

    def _evaluate(self):
        if self.opt['rank'] == 0:
            self.model.log('-----------------------------------------------')
            self.model.log("Evaluating model ... ")

        self.model.set_up_model()

        eval_dataset = 'test'
        batch_generator_eval = self.model.get_batch_generator(eval_dataset)
        predictions = self._eval_batches(
            self.model.module, batch_generator_eval, self.model.saveFolder, eval_dataset)

        return predictions

    def _eval_batches(self, module, dev_batches, save_folder, label=''):
        max_sent_len = int(self.opt['MAX_GEN_LENGTH'])

        print('Decoding current model ... \nSaving folder is {}'.format(save_folder))
        print('Each sample will cost about 10 second.')

        predictions = []  # prediction of tokens from model
        if not isinstance(module.tokenizer, list):
            decoder_tokenizer = module.tokenizer
        elif len(module.tokenizer) == 1:
            decoder_tokenizer = module.tokenizer[0]
        elif len(module.tokenizer) == 2:
            decoder_tokenizer = module.tokenizer[1]
        else:
            assert False, f"len(module.tokenizer) > 2"

        with torch.no_grad():
            for j, dev_batch in enumerate(dev_batches):
                for b in dev_batch:
                    if torch.is_tensor(dev_batch[b]):
                        dev_batch[b] = dev_batch[b].to(self.opt['device'])

                beam_search_res = module(dev_batch, beam_search=True, max_sent_len=max_sent_len)
                pred = [[t[0] for t in x] if len(x) > 0 else [[]] for x in beam_search_res]
                predictions.extend([[self._convert_tokens_to_string(decoder_tokenizer, tt) for tt in t] for t in pred])

                if ("DEBUG" in self.opt and j >= 10) or j >= self.model.task.evaluator.eval_batches_num:
                    # in debug mode (decode first 10 batches) ortherwise decode first self.eval_batches_num bathes
                    break

        top1_predictions = [x[0] for x in predictions]
        return top1_predictions

    def _convert_tokens_to_string(self, tokenizer, tokens):
        if 'EVAL_TOKENIZED' in self.opt:
            tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
        if 'EVAL_LOWERCASE' in self.opt:
            tokens = [t.lower() for t in tokens]
        if 'EVAL_TOKENIZED' in self.opt:
            return ' '.join(tokens)
        else:
            return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)

    def _preprocess(self, corpus, test_path):
        samples = []
        for i, sample in enumerate(corpus):
            new_sample = {'id': i, 'meeting': [], 'summary': []}
            if isinstance(sample, str):
                raise RuntimeError("Error: the input of HMNet should be dialogues, rather than documents.")

            # add all the turns one by one
            for turn in sample:
                turn = [x.strip() for x in turn.split(':')]
                tokenized_turn = nlp(turn[1])
                # In case we can't find proper entity in move_names
                ent_id = []
                pos_id = []
                for token in tokenized_turn:
                    ent = token.ent_iob_+'-'+token.ent_type_ if token.ent_iob_ != 'O' else 'O'
                    ent_id.append(ENT[ent] if ent in ENT else ENT[''])

                    pos = token.tag_
                    pos_id.append(POS[pos] if pos in POS else POS[''])

                new_sample['meeting'].append({
                    'speaker': turn[0],
                    'role': '',
                    'utt': {
                        'word': [str(token) for token in tokenized_turn],
                        'pos_id': pos_id,
                        'ent_id': ent_id
                    }
                })
            new_sample['summary'].append("This is a dummy summary. HMNet will filter out the sample w/o summary!")
            samples.append(new_sample)
            # save to the gzip
            file_path = os.path.join(test_path, "split_{}.jsonl.gz".format(i))
            with gzip.open(file_path, 'wt', encoding='utf-8') as file:
                file.write(json.dumps(new_sample))

    def _clean_datafolder(self, data_folder):
        for name in os.listdir(data_folder):
            name = os.path.join(data_folder, name)
            if '.gz' in name:
                os.remove(name)

    def _create_datafolder(self, data_folder):
        if os.path.exists(data_folder):
            self._clean_datafolder(data_folder)
        else:
            os.makedirs(data_folder)
        with open(os.path.join(os.path.dirname(data_folder), 'test_ami.json'), 'w', encoding='utf-8') as file:
            json.dump([
                        {
                            "source":
                            {
                                "dataset": "../ExampleRawData/meeting_summarization/AMI_proprec/test/"
                            },
                            "task": "meeting",
                            "name": "ami"
                        }
                      ], file)

        with open(os.path.join(os.path.dirname(os.path.dirname(data_folder)), 'role_dict_ext.json'), 'w') as file:
            json.dump({}, file)

    @classmethod
    def show_capability(cls) -> None:
        basic_description = cls.generate_basic_description()
        more_details = ("A HMNet model finetuned on CNN-DM dataset for summarization.\n\n"
                        "Strengths:\n - High performance on dialogue summarization task.\n\n"
                        "Weaknesses:\n - Not suitable for datasets other than dialogues.\n\n"
                        "Initialization arguments:\n "
                        " - `corpus`: Unlabelled corpus of documents.\n")
        print(f"{basic_description} \n {'#' * 20} \n {more_details}")
