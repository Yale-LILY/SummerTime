# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import math
import numpy as np
import random
import time
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Models.Networks.Layers import dropout, set_seq_dropout
from Models.Networks.Transformer import EncoderBlock, LayerNorm, Embedder, Splitter, Attention, MLP
from ThirdParty.Huggingface.Transformers.src.transformers import tokenization_transfo_xl 
from ThirdParty.Huggingface.Transformers.src.transformers.modeling_encoder_decoder import calc_banned_ngram_tokens, calc_banned_bad_words_ids, top_k_top_p_filtering, BeamHypotheses
import sys
import os

# These two dicts are adapted from SpaCy 2.3.1, since HMNet's embedding for POS and ENT is fixed
POS = {'': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-RRB-': 5, '.': 6, ':': 7, 'ADD': 8, 'AFX': 9, 'CC': 10, 'CD': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'HYPH': 15, 'IN': 16, 'JJ': 17, 'JJR': 18, 'JJS': 19, 'LS': 20, 'MD': 21, 'NFP': 22, 'NN': 23, 'NNP': 24, 'NNPS': 25, 'NNS': 26, 'PDT': 27, 'POS': 28, 'PRP': 29, 'PRP$': 30, 'RB': 31, 'RBR': 32, 'RBS': 33, 'RP': 34, 'SYM': 35, 'TO': 36, 'UH': 37, 'VB': 38, 'VBD': 39, 'VBG': 40, 'VBN': 41, 'VBP': 42, 'VBZ': 43, 'WDT': 44, 'WP': 45, 'WP$': 46, 'WRB': 47, 'XX': 48, '_SP': 49, '``': 50}
ENT = {'': 0, 'B-ORG': 1, 'B-DATE': 2, 'B-PERSON': 3, 'B-GPE': 4, 'B-MONEY': 5, 'B-CARDINAL': 6, 'B-NORP': 7, 'B-PERCENT': 8, 'B-WORK_OF_ART': 9, 'B-LOC': 10, 'B-TIME': 11, 'B-QUANTITY': 12, 'B-FAC': 13, 'B-EVENT': 14, 'B-ORDINAL': 15, 'B-PRODUCT': 16, 'B-LAW': 17, 'B-LANGUAGE': 18, 'I-ORG': 19, 'I-DATE': 20, 'I-PERSON': 21, 'I-GPE': 22, 'I-MONEY': 23, 'I-CARDINAL': 24, 'I-NORP': 25, 'I-PERCENT': 26, 'I-WORK_OF_ART': 27, 'I-LOC': 28, 'I-TIME': 29, 'I-QUANTITY': 30, 'I-FAC': 31, 'I-EVENT': 32, 'I-ORDINAL': 33, 'I-PRODUCT': 34, 'I-LAW': 35, 'I-LANGUAGE': 36, 'L-ORG': 37, 'L-DATE': 38, 'L-PERSON': 39, 'L-GPE': 40, 'L-MONEY': 41, 'L-CARDINAL': 42, 'L-NORP': 43, 'L-PERCENT': 44, 'L-WORK_OF_ART': 45, 'L-LOC': 46, 'L-TIME': 47, 'L-QUANTITY': 48, 'L-FAC': 49, 'L-EVENT': 50, 'L-ORDINAL': 51, 'L-PRODUCT': 52, 'L-LAW': 53, 'L-LANGUAGE': 54, 'U-ORG': 55, 'U-DATE': 56, 'U-PERSON': 57, 'U-GPE': 58, 'U-MONEY': 59, 'U-CARDINAL': 60, 'U-NORP': 61, 'U-PERCENT': 62, 'U-WORK_OF_ART': 63, 'U-LOC': 64, 'U-TIME': 65, 'U-QUANTITY': 66, 'U-FAC': 67, 'U-EVENT': 68, 'U-ORDINAL': 69, 'U-PRODUCT': 70, 'U-LAW': 71, 'U-LANGUAGE': 72, 'O': 73}


class MeetingNet_Transformer(nn.Module):
    def __init__(self, opt):
        super(MeetingNet_Transformer, self).__init__()

        self.opt = opt
        self.use_cuda = (self.opt['cuda'] == True)
        self.config = {}

        # load tokenizer
        self.tokenizer_class = getattr(tokenization_transfo_xl, opt['PRE_TOKENIZER'])
        self.pretrained_tokenizer_path = os.path.join(opt['datadir'], opt['PRE_TOKENIZER_PATH'])
        if not os.path.isdir(self.pretrained_tokenizer_path):
            '''
            This if-else statement makes sure the pre-trained tokenizer exists
            If it does not exist, it assumes the input string is the HuggingFace tokenizer name,
            and downloads it from their website.
            '''
            self.pretrained_tokenizer_path = opt['PRE_TOKENIZER_PATH']
        else:
            print('Loading Tokenizer from {}...'.format(self.pretrained_tokenizer_path))

        # here is a simple workaround to make sure all special tokens are not None
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_tokenizer_path)
        special_tokens_tuple_list = [("eos_token", 128), ("unk_token", 129), ("pad_token", 130), ("bos_token", 131)]

        for special_token_name, special_token_id_offset in special_tokens_tuple_list:
            if getattr(self.tokenizer, special_token_name) == None:
                setattr(self.tokenizer, special_token_name, self.tokenizer.convert_ids_to_tokens(len(self.tokenizer)-special_token_id_offset))
                self.config[special_token_name] = self.tokenizer.convert_ids_to_tokens(len(self.tokenizer)-special_token_id_offset)
                self.config[special_token_name+'_id'] =  len(self.tokenizer)-special_token_id_offset

        self.vocab_size = self.tokenizer.vocab_size
        opt['vocab_size'] = self.vocab_size
        self.role_size = int(opt['ROLE_SIZE'])
        vocab_dim = int(opt['VOCAB_DIM'])
        role_dim = int(opt['ROLE_DIM'])
        opt['transformer_embed_dim'] = vocab_dim
        embed = nn.Embedding(self.vocab_size, vocab_dim, padding_idx=self.tokenizer.pad_token_id)
        nn.init.normal_(embed.weight, std=0.02)
        embedder = Embedder(opt, embed)
        role_embed = nn.Embedding(self.role_size, role_dim, padding_idx=0)

        self.encoder = Encoder(opt, self.vocab_size, vocab_dim, role_dim, embedder, role_embed)
        self.decoder = Decoder(opt, vocab_dim, self.vocab_size, embedder, self.encoder.token_transformer_dim, self.encoder.sent_transformer_dim)

        if 'PYLEARN_MODEL' in self.opt:
            self.from_pretrained(os.path.join(opt['datadir'], opt['PYLEARN_MODEL']))

    def save_pretrained(self, save_dir):
        network_state = dict([(k, v) for k, v in self.state_dict().items()])
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        torch.save(params, os.path.join(save_dir, 'model.pt'))

    def from_pretrained(self, load_dir):
        checkpoint = torch.load(os.path.join(load_dir, 'model.pt'), map_location=torch.device('cuda', self.opt['local_rank']))
        state_dict = checkpoint['state_dict']
        
        self.load_state_dict(state_dict['network'])

        return self

    def get_training_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch, beam_search=False, max_sent_len=None):
        if beam_search:
            # return self.beam_search(batch, max_sent_len)
            return self.generate(batch, max_sent_len)

        outputs = self._forward(**batch)
        vocab_logprob = outputs[0]

        # assume all encoder-decoder model input has BOS and EOS
        # otherwise the loss will be ill-defined
        return vocab_logprob

    '''
        Input:
         encoders_input_ids = 1 * num_turns * x_len (word_ids)
         encoders_input_roles = 1 * num_turns (role_ids)
         encoders_input_pos = 1 * num_turns * x_len (pos_ids)
         encoders_input_ent = 1 * num_turns * x_len (ent_ids)
         decoder_input_ids = 1 * y_len (word_ids) 
        Output:
          vocab_logprob  = 1 x y_len x vocab_size
    '''
    def _forward(self, **kwargs):
        
        encoder_input_ids = kwargs.pop('encoder_input_ids')
        encoder_input_roles = kwargs.pop('encoder_input_roles')
        encoder_input_pos = kwargs.pop('encoder_input_pos')
        encoder_input_ent = kwargs.pop('encoder_input_ent')
        decoder_input_ids = kwargs.pop('decoder_input_ids')

        token_encoder_outputs, sent_encoder_outputs = self.encoder(encoder_input_ids, encoder_input_roles, encoder_input_pos, encoder_input_ent)
        vocab_logprob = self.decoder(token_encoder_outputs, sent_encoder_outputs, decoder_input_ids)
        return vocab_logprob, (token_encoder_outputs, sent_encoder_outputs)

    def generate(self, batch, max_sent_len):
        self.eval()
        self.beam_width = int(self.opt['BEAM_WIDTH'])

        input_ids = batch["encoder_input_ids"]
        input_roles = batch["encoder_input_roles"]
        input_pos = batch["encoder_input_pos"]
        input_ent = batch["encoder_input_ent"]

        batch_size = input_ids.shape[0]

        num_return_sequences = self.opt.get("NUM_RETURN_SEQUENCES", 1)
        outputs = self._generate(
            input_ids=input_ids, input_roles=input_roles,
            input_pos=input_pos, input_ent=input_ent,
            min_length=self.opt.get('MIN_GEN_LENGTH', None),
            max_length=max_sent_len,
            num_beams=self.beam_width,
            bad_words_ids=None,
            bos_token_id=self.tokenizer.bos_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.opt.get("DO_SAMPLE", False),
            top_k=self.opt.get("TOP_K", 50),
            top_p=self.opt.get("TOP_P", 1),
            repetition_penalty=self.opt.get("REPETITION_PENALTY", 1.0),
            length_penalty=self.opt.get("LENGTH_PENALTY", 1.0),
            no_repeat_ngram_size=self.opt.get("NO_REPEAT_NGRAM_SIZE", 3),
            num_return_sequences=num_return_sequences)

        sents = []
        outputs = outputs.view(outputs.shape[0], num_return_sequences, -1)

        for idx in range(batch_size):
            # TODO: use real inference scores
            candidates = [
                (self.tokenizer.convert_ids_to_tokens(outputs[idx, i, :]), 0.0)
                for i in range(num_return_sequences)]
            sents.append(candidates)

        return sents

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        return {
            "decoder_input_ids": input_ids,
            "token_encoder_outputs": encoder_outputs[0],
            "sent_encoder_outputs": encoder_outputs[1],
        }

    def prepare_scores_for_generation(self, scores, **kwargs):
        return scores

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    @torch.no_grad()
    def _generate(
        self,
        input_ids=None,
        input_roles=None,
        input_pos=None,
        input_ent=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=False,
        num_beams=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

            `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 3, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"
        

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = self.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"

        encoder_outputs = self.encoder(input_ids, input_roles, input_pos, input_ent)

        # # Expand input ids if num_beams > 1 or num_return_sequences > 1
        # if num_return_sequences > 1 or num_beams > 1:
        #     input_sent_len = input_ids.shape[2]
        #     input_word_len = input_ids.shape[3]
        #     input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_sent_len, input_word_len)
        #     attention_mask = attention_mask.unsqueeze(1).expand(
        #         batch_size, effective_batch_mult * num_beams, input_sent_len, input_word_len
        #     )

        #     input_ids = input_ids.contiguous().view(
        #         effective_batch_size * num_beams, input_sent_len, input_word_len
        #     )  # shape: (batch_size * num_return_sequences * num_beams, input_sent_len, input_word_len)
        #     attention_mask = attention_mask.contiguous().view(
        #         effective_batch_size * num_beams, input_sent_len, input_word_len
        #     )  # shape: (batch_size * num_return_sequences * num_beams, input_sent_len, input_word_len)

        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), encoder_outputs[1].index_select(0, expanded_batch_idxs))


        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask)
            
            outputs = self.decoder(**model_inputs)
            next_token_logits = outputs[:, -1, :]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            cur_len = cur_len + 1

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask)
            outputs = self.decoder(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[:, -1, :]  # (batch_size * num_beams, vocab_size)

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(
                    next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
                )

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            if do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                scores = self.prepare_scores_for_generation(scores, cur_len=cur_len, max_length=max_length)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = batch_size * num_beams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    scores[i, banned_tokens] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for i, banned_tokens in enumerate(banned_tokens):
                    scores[i, banned_tokens] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if were done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # update current length
            cur_len = cur_len + 1

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    # force one of token_ids to be generated by setting prob of all other tokens to 0.
    def _force_token_ids_generation(self, scores, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` and `mems` is at 2nd position
            reordered_layer_past = [layer_past[i, :].unsqueeze(0).clone().detach() for i in beam_idx]
            reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
            # check that shape matches
            assert reordered_layer_past.shape == layer_past.shape
            reordered_past.append(reordered_layer_past)
        past = tuple(reordered_past)
        return past


'''
  Transformer encoder
'''
class MeetingTransformerEncoder(nn.Module):
    '''
      Input:
        transformer_embed_dim: transformer dimension
    '''
    def __init__(self, opt, transformer_embed_dim):
        super(MeetingTransformerEncoder, self).__init__()
        vocab = int(opt['vocab_size'])
        n_layer = int(opt['TRANSFORMER_LAYER'])
        opt['transformer_embed_dim'] = transformer_embed_dim
        block = EncoderBlock(opt)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])

    '''
      Input:
        x: batch x len x n_state
      Output:
        h: batch x len x n_state
    '''
    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h, None)
        return h

'''
 One encoder block of transformer
'''
class MeetingDecoderBlock(nn.Module):
    def __init__(self, opt, n_state):
        super(MeetingDecoderBlock, self).__init__()
        self.opt = opt
        self.decoder_splitter = Splitter(n_state)
        self.attn = Attention(n_state, opt)
        self.token_attn = Attention(n_state, opt)
        self.sent_attn = Attention(n_state, opt)
        self.ln_1 = LayerNorm(n_state)
        self.ln_2 = LayerNorm(n_state)
        opt['transformer_embed_dim'] = n_state
        self.mlp = MLP(4 * n_state, opt)
        self.ln_3 = LayerNorm(n_state)
        self.ln_4 = LayerNorm(n_state)

    '''
     Input:
       y: batch x len x n_state (decoder part)
       token_enc_key: batch x encoder_len x n_state
       token_enc_value: batch x encoder_len x n_state
       sent_enc_key: batch x encoder_len x n_state
       sent_enc_value: batch x encoder_len x n_state
     Output:
       h: batch x len x n_state
    '''
    def forward(self, y, token_enc_key, token_enc_value, sent_enc_key, sent_enc_value):
        query, key, value = self.decoder_splitter(y)
        # batch x len x n_state

        # self-attention
        a = self.attn(query, key, value, None, one_dir_visible=True)
        # batch x len x n_state

        n = self.ln_1(y + a) # residual

        if 'NO_HIERARCHY' in self.opt:
            q = y
            r = n
        else:
            # src-tgt attention on sentences
            q = self.sent_attn(n, sent_enc_key, sent_enc_value, None)
            r = self.ln_3(n + q) # residual
            # batch x len x n_state

        # src-tgt attention on tokens
        o = self.token_attn(r, token_enc_key, token_enc_value, None)
        p = self.ln_2(r + o) # residual
        # batch x len x n_state


        m = self.mlp(p)
        h = self.ln_4(p + m)
        return h

'''
  Transformer decoder
'''
class MeetingTransformerDecoder(nn.Module):
    '''
      Input:
        embed_size: decoder transformer dimension
        token_dim: dimension of transformer from token encoder side
        sent_dim: dimension of transformer from sent encoder side
    '''
    def __init__(self, opt, embedder, embed_size, token_dim, sent_dim):
        super(MeetingTransformerDecoder, self).__init__()
        self.fp16 = 'FP16' in opt
        vocab_size = int(opt['vocab_size'])
        n_layer = int(opt['TRANSFORMER_LAYER'])
        self.encoder_splitter = Splitter(embed_size)
        block = MeetingDecoderBlock(opt, embed_size)
        self.token_linear = nn.Linear(token_dim, embed_size)
        self.sent_linear = nn.Linear(sent_dim, embed_size)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.linear = nn.Linear(embed_size, vocab_size, bias = False)
        self.linear.weight = embedder.embed.weight # share weight

    '''
      Input:
        token_encoder_outputs: 1 x (encoder_len - sent_num) x token_transformer_dim
        sent_encoder_outputs: 1 x sent_num x sent_transformer_dim
        y: batch x len x n_state
      Output:
        prob: batch x len x vocab_size (probabilities after softmax)
    '''
    def forward(self, token_encoder_inputs, sent_encoder_inputs, decoder_input_ids):
        _, token_enc_key, token_enc_value = self.encoder_splitter(self.token_linear(token_encoder_inputs))
        # token_enc_key: batch x encoder_len x n_state
        # token_enc_value: batch x encoder_len x n_state

        _, sent_enc_key, sent_enc_value = self.encoder_splitter(self.sent_linear(sent_encoder_inputs))
        # sent_enc_key: batch x encoder_len x n_state
        # sent_enc_value: batch x encoder_len x n_state

        h = decoder_input_ids
        for block in self.blocks:
            h = block(h, token_enc_key, token_enc_value, sent_enc_key, sent_enc_value)
        prob = F.softmax(self.linear(h), dim=-1)
        return prob


class Encoder(nn.Module):
    '''
     vocab_size: size of input vocabulary
     embed_size: word embedding dimension of dictionary
     role_dim: role embedding dimension
     embed: the nn.Embedding for vocab
     role_embed: the nn.Embedding for role
    '''
    def __init__(self, opt, vocab_size, embed_size, role_dim, embedder, role_embed):
        super(Encoder, self).__init__()
        self.opt = opt
        self.vocab_size = vocab_size

        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        self.embed_size = embed_size
        self.embedder = embedder
        self.role_embed = role_embed

        self.token_transformer_dim = embed_size
        if 'USE_POSENT' in opt:
            print('Use POS and ENT')
            pos_dim = opt['POS_DIM']
            ent_dim = opt['ENT_DIM']
            self.pos_embed = nn.Embedding(len(POS), pos_dim)
            self.ent_embed = nn.Embedding(len(ENT), ent_dim)
            self.token_transformer_dim += pos_dim + ent_dim

        self.sent_transformer_dim = self.token_transformer_dim
        if 'USE_ROLE' in opt:
            print("USE_ROLE")
            role_dim = opt['ROLE_DIM']
            self.sent_transformer_dim += role_dim

        self.token_encoder = MeetingTransformerEncoder(opt, self.token_transformer_dim)
        self.sent_encoder = MeetingTransformerEncoder(opt, self.sent_transformer_dim)

    '''
     x = bz * sent_num * x_len (word_ids)
     x_role = bz * sent_num (role_ids)
     x_pos = bz * sent_num * x_len (pos_ids)
     x_ent = bz * sent_num * x_len (ent_ids)
     outputs:
       token_encoder_outputs: bz x x_len_total x token_transformer_dim
       sent_encoder_outputs:  bz x sent_num x sent_transformer_dim
    '''
    def forward(self, x, x_role, x_pos, x_ent):
        batch_size = x.size(0)
        sent_num = x.size(1)
        x_len = x.size(2)

        # x contains word id >= vocab_size
        vocab_x = x.clone()
        vocab_x[vocab_x >= self.vocab_size] = 1 # UNK
        embedded = self.embedder(vocab_x.view(batch_size, -1))
        # embedded = 1 x sent_num * x_len x embed_size
        embedded = embedded.view(batch_size, sent_num, x_len, -1)
        # embedded = 1 x sent_num x x_len x embed_size

        if 'USE_ROLE' in self.opt:
            role_embed = self.role_embed(x_role) # 1 x sent_num x role_dim

        if 'USE_POSENT' in self.opt:
            embedded = torch.cat([embedded, self.pos_embed(x_pos), self.ent_embed(x_ent)], dim=3)
            # 1 x sent_num x x_len x (embed_size + pos_dim + ent_dim )

        feat_dim = embedded.size(3)

        token_transformer_output = self.token_encoder(embedded.view(-1, x_len, feat_dim))
        token_transformer_dim = token_transformer_output.size(2)
        token_transformer_output = token_transformer_output.view(batch_size, sent_num, x_len, token_transformer_dim)
        # 1 x sent_num x x_len x token_transformer_dim

        sent_encoder_inputs = token_transformer_output[:, :, 0, :] # 1 x sent_num x token_transformer_dim
        if 'USE_ROLE' in self.opt:
            sent_encoder_inputs = torch.cat([sent_encoder_inputs, role_embed], dim=2)
        sent_encoder_outputs = self.sent_encoder(sent_encoder_inputs) # 1 x sent_num x sent_transformer_dim

        token_transformer_output = token_transformer_output.view(batch_size, -1, token_transformer_dim)

        return token_transformer_output, sent_encoder_outputs

class Decoder(nn.Module):
    def __init__(self, opt, embed_size, vocab_size, embedder, token_transformer_dim, sent_transformer_dim):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedder = embedder
        self.sent_decoder = MeetingTransformerDecoder(opt, embedder, embed_size, token_transformer_dim, sent_transformer_dim)

    def forward(self, token_encoder_outputs, sent_encoder_outputs, decoder_input_ids):
        vocab_y = decoder_input_ids.clone()
        vocab_y[vocab_y >= self.vocab_size] = 1 # UNK
        embedded = self.embedder(vocab_y)

        vocab_prob = self.sent_decoder(token_encoder_outputs, sent_encoder_outputs, embedded)
        # vocab_prob: batch x y_len x vocab_size

        vocab_logprob = torch.log(vocab_prob + 1e-15)
        return vocab_logprob



    