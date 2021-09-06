# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import shutil
from string import ascii_uppercase
from tqdm.auto import tqdm
from model.third_party.HMNet.Evaluation.OldROUGEEval import rouge
from model.third_party.HMNet.ThirdParty.ROUGE import pyrouge
from shutil import copyfile
# from mpi4py import MPI
import torch
import logging
import json


def write_json_res(
    output_file, tokenizers, x_ids, y_ids, x_tokens, y_tokens, predictions, gts
):
    data = []

    # for x_id, y_id, x_token, y_token, preds, gt in zip(x_ids, y_ids, x_tokens, y_tokens, predictions, gts):
    # x_id = tokenizers[0].decode(x_id, skip_special_tokens=False) if x_id.dim() == 1 else tokenizers[0].convert_tokens_to_string(x_token)
    # y_id = tokenizers[1].decode(y_id, skip_special_tokens=False) if y_id.dim() == 1 else tokenizers[1].convert_tokens_to_string(y_token)
    for x_token, y_token, preds, gt in zip(x_tokens, y_tokens, predictions, gts):
        data.append(
            {
                # 'x_ids': x_id,
                # 'y_ids': y_id,
                "x_tokens": x_token if isinstance(x_token, str) else " ".join(x_token),
                "y_tokens": y_token if isinstance(y_token, str) else " ".join(y_token),
                "predictions": preds,
                "gt": gt,
            }
        )

    json.dump(data, output_file, indent=4, ensure_ascii=False)


logger = logging.getLogger(__name__)

"""
This code can only be run within docker "rouge", because of the usage of rouge-perl
"""


"""" In ROUGE parlance, your summaries are ‘system’ summaries and the gold standard summaries are ‘model’ summaries.
The summaries should be in separate folders, whose paths are set with the system_dir and model_dir variables.
All summaries should contain one sentence per line."""


class ROUGEEval:
    """
    Wrapper class for pyrouge.
    Compute ROUGE given predictions and references for summarization evaluation.
    """

    def __init__(self, run_dir, save_dir, opt):
        self.run_dir = run_dir
        self.save_dir = save_dir
        self.opt = opt

        # use relative path to make it work on Philly
        self.pyrouge_dir = os.path.join(
            os.path.dirname(__file__), "../ThirdParty/ROUGE/ROUGE-1.5.5/"
        )

        self.eval_batches_num = self.opt.get("EVAL_BATCHES_NUM", float("Inf"))
        self.best_score = -float("Inf")
        self.best_res = {}

    def reset_best_score(self, set_high=False):
        if set_high:
            self.best_score = float("Inf")
        else:
            self.best_score = -float("Inf")

    def make_html_safe(self, s):
        s = s.replace("<", "&lt;")
        s = s.replace(">", "&gt;")
        return s

    def print_to_rouge_dir(
        self, summaries, dir, suffix, split_chars, special_char_dict=None
    ):
        for idx, summary in enumerate(summaries):
            fname = os.path.join(dir, "%06d_%s.txt" % (idx, suffix))
            with open(fname, "wb") as f:
                sents = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", summary)
                for i, sent in enumerate(sents):
                    if split_chars:
                        # sent = re.sub(r'([\u4e00-\u9fff])', r' \1 ', sent)
                        for x in re.finditer(r"([\u4e00-\u9fff])", sent):
                            if not x.group(1) in special_char_dict:
                                special_char_dict[x.group(1)] = len(special_char_dict)
                            sent = sent.replace(
                                x.group(1), " {} ".format(special_char_dict[x.group(1)])
                            )
                    if i == len(sents) - 1:
                        to_print = sent.encode("utf-8")
                    else:
                        to_print = sent.encode("utf-8") + "\n".encode("utf-8")
                    f.write(to_print)

    def print_to_rouge_dir_gt(self, summaries, dir, suffix, split_chars):
        if split_chars:
            char_dict = {}

        for idx, summary in enumerate(summaries):
            for ref_idx, sub_summary in enumerate(summary.split(" ||| ")):
                fname = os.path.join(
                    dir, "%s.%06d_%s.txt" % (ascii_uppercase[ref_idx], idx, suffix)
                )
                with open(fname, "wb") as f:
                    sents = re.split(
                        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", sub_summary
                    )
                    for i, sent in enumerate(sents):
                        if split_chars:
                            for x in re.finditer(r"([\u4e00-\u9fff])", sent):
                                if not x.group(1) in char_dict:
                                    char_dict[x.group(1)] = len(char_dict)
                                sent = sent.replace(
                                    x.group(1), " {} ".format(char_dict[x.group(1)])
                                )

                        if i == len(sents) - 1:
                            to_print = sent.encode("utf-8")
                        else:
                            to_print = sent.encode("utf-8") + "\n".encode("utf-8")
                        f.write(to_print)

        if split_chars:
            return char_dict

    # def filter_empty(self, predictions, groundtruths):
    #     new_predicitons = []
    #     new_groundtruths = []
    #
    #     for pred, gt in zip(predictions, groundtruths):
    #         if len(gt) == 0:
    #             continue
    #         new_groundtruths.append(gt)
    #         if len(pred) == 0:
    #             new_predicitons.append('<ept>')
    #         else:
    #             new_predicitons.append(pred)
    #     return new_predicitons, new_groundtruths

    def _convert_tokens_to_string(self, tokenizer, tokens):
        if "EVAL_TOKENIZED" in self.opt:
            tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
        if "EVAL_LOWERCASE" in self.opt:
            tokens = [t.lower() for t in tokens]
        if "EVAL_TOKENIZED" in self.opt:
            return " ".join(tokens)
        else:
            return tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
            )

    def eval_batches(self, module, dev_batches, save_folder, label=""):
        max_sent_len = int(self.opt["MAX_GEN_LENGTH"])

        logger.info(
            "Decoding current model ... \nSaving folder is {}".format(save_folder)
        )

        predictions = []  # prediction of tokens from model
        x_tokens = []  # input tokens
        y_tokens = []  # groundtruths tokens
        x_ids = []  # input token ids
        y_ids = []  # groundtruths token ids
        gts = []  # groundtruths string
        got_better_score = False
        # err = 0
        if not isinstance(module.tokenizer, list):
            encoder_tokenizer = module.tokenizer
            decoder_tokenizer = module.tokenizer
        elif len(module.tokenizer) == 1:
            encoder_tokenizer = module.tokenizer[0]
            decoder_tokenizer = module.tokenizer[0]
        elif len(module.tokenizer) == 2:
            encoder_tokenizer = module.tokenizer[0]
            decoder_tokenizer = module.tokenizer[1]
        else:
            assert False, f"len(module.tokenizer) > 2"

        with torch.no_grad():
            for j, dev_batch in enumerate(dev_batches):
                for b in dev_batch:
                    if torch.is_tensor(dev_batch[b]):
                        dev_batch[b] = dev_batch[b].to(self.opt["device"])

                beam_search_res = module(
                    dev_batch, beam_search=True, max_sent_len=max_sent_len
                )
                pred = [
                    [t[0] for t in x] if len(x) > 0 else [[]] for x in beam_search_res
                ]
                predictions.extend(
                    [
                        [
                            self._convert_tokens_to_string(decoder_tokenizer, tt)
                            for tt in t
                        ]
                        for t in pred
                    ]
                )

                gts.extend(
                    [
                        self._convert_tokens_to_string(decoder_tokenizer, t)
                        for t in dev_batch["decoder_tokens"]
                    ]
                )
                x_tokens.extend(dev_batch["encoder_tokens"])
                y_tokens.extend(dev_batch["decoder_tokens"])

                if ("DEBUG" in self.opt and j >= 10) or j >= self.eval_batches_num:
                    # in debug mode (decode first 10 batches) ortherwise decode first self.eval_batches_num bathes
                    break

        # use MPI to gather results from all processes / GPUs
        # the result of the gather operation is a list of sublists
        # each sublist corresponds to the list created on one of the MPI processes (or GPUs, respectively)
        # we flatten this list into a "simple" list
        assert len(predictions) == len(
            gts
        ), "len(predictions): {0}, len(gts): {1}".format(len(predictions), len(gts))
        # comm = MPI.COMM_WORLD
        # predictions = comm.gather(predictions, root=0)
        # x_tokens = comm.gather(x_tokens, root=0)
        # y_tokens = comm.gather(y_tokens, root=0)
        # if GPU numbers are high (>=8), passing x_ids, y_ids to a rank 0 will cause out of memory
        # x_ids = comm.gather(x_ids, root=0)
        # y_ids = comm.gather(y_ids, root=0)
        # gts = comm.gather(gts, root=0)
        if self.opt["rank"] == 0:
            # flatten lists
            predictions = [item for sublist in predictions for item in sublist]
            y_tokens = [item for sublist in y_tokens for item in sublist]
            x_tokens = [item for sublist in x_tokens for item in sublist]
            # x_ids = [item for sublist in x_ids for item in sublist]
            # y_ids = [item for sublist in y_ids for item in sublist]
            gts = [item for sublist in gts for item in sublist]
            # import pdb; pdb.set_trace()
            assert (
                len(predictions) == len(y_tokens) == len(x_tokens) == len(gts)
            ), "len(predictions): {0}, len(y_tokens): {1}, len(x_tokens): {2}, len(gts): {3}".format(
                len(predictions), len(y_tokens), len(x_tokens), len(gts)
            )

            # write intermediate results only on rank 0
            if not os.path.isdir(os.path.join(save_folder, "intermediate_results")):
                os.makedirs(os.path.join(save_folder, "intermediate_results"))
            top_1_predictions = [pred[0] for pred in predictions]
            with open(
                os.path.join(
                    save_folder, "intermediate_results", "res_" + label + ".json"
                ),
                "w",
                encoding="utf-8",
            ) as output_file:
                write_json_res(
                    output_file,
                    [encoder_tokenizer, decoder_tokenizer],
                    x_ids,
                    y_ids,
                    x_tokens,
                    y_tokens,
                    predictions,
                    gts,
                )
            try:
                result = self.eval(top_1_predictions, gts)
            except Exception as e:
                logger.exception("ROUGE Eval ERROR")
                result = {}
                score = -float("Inf")
                pass  # this happens when no overlapping between pred and gts
            else:
                rouge_su4 = rouge(top_1_predictions, gts)  # f, prec, recall
                result = {
                    "ROUGE_1": result["rouge_1_f_score"] * 100.0,
                    "ROUGE_1_Prc": result["rouge_1_precision"] * 100.0,
                    "ROUGE_1_Rcl": result["rouge_1_recall"] * 100.0,
                    "ROUGE_2": result["rouge_2_f_score"] * 100.0,
                    "ROUGE_2_Prc": result["rouge_2_precision"] * 100.0,
                    "ROUGE_2_Rcl": result["rouge_2_recall"] * 100.0,
                    "ROUGE_L": result["rouge_l_f_score"] * 100.0,
                    "ROUGE_L_Prc": result["rouge_l_precision"] * 100.0,
                    "ROUGE_L_Rcl": result["rouge_l_recall"] * 100.0,
                    "ROUGE_SU4": rouge_su4["rouge_su4_f_score"] * 100.0,
                }

                score = result["ROUGE_1"]
                if score > self.best_score:
                    copyfile(
                        os.path.join(
                            save_folder,
                            "intermediate_results",
                            "res_" + label + ".json",
                        ),
                        os.path.join(
                            save_folder,
                            "intermediate_results",
                            "res_" + label + ".best.json",
                        ),
                    )
                    self.best_score = score
                    self.best_res = result
                    got_better_score = True

        else:
            result = {}
            score = -float("Inf")
            got_better_score = False

        return result, score, got_better_score

    def eval(self, predictions, groundtruths):
        # predictions, groundtruths = self.filter_empty(predictions, groundtruths)
        predictions = [self.make_html_safe(w) for w in predictions]
        groundtruths = [self.make_html_safe(w) for w in groundtruths]
        pred_dir = os.path.join(self.save_dir, "predictions")
        if os.path.exists(pred_dir):
            shutil.rmtree(pred_dir)
        os.makedirs(pred_dir)

        gt_dir = os.path.join(self.save_dir, "groundtruths")
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.makedirs(gt_dir)

        special_char_dict = self.print_to_rouge_dir_gt(
            groundtruths, gt_dir, "gt", "SPLIT_CHARS_FOR_EVAL" in self.opt
        )
        self.print_to_rouge_dir(
            predictions,
            pred_dir,
            "pred",
            "SPLIT_CHARS_FOR_EVAL" in self.opt,
            special_char_dict,
        )

        r = pyrouge.Rouge155(self.pyrouge_dir)
        r.system_dir = pred_dir
        r.model_dir = gt_dir
        r.system_filename_pattern = "(\d+)_pred.txt"
        r.model_filename_pattern = "[A-Z].#ID#_gt.txt"
        results = r.output_to_dict(r.convert_and_evaluate())
        return results
