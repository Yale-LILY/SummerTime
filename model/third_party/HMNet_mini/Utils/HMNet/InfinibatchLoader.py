# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import gzip
import numpy as np
from random import Random, shuffle, random
import torch
import math
from DataLoader import iterators
import json
import struct
from timeit import default_timer as timer

"""
Define different types of task here
"""

MONO_TASKS = ['meeting'] # tasks that takes a singe sentence and reconstruct
TRANS_TASKS = ['sum'] # tasks that transfer a source sentence to a target sentence
ALL_TASKS = MONO_TASKS + TRANS_TASKS # all valid tasks

def _bump_seed(seed):
    """
    Helper to bump a random seed if not None.
    """
    return None if seed is None else seed + 1

def HMNetBatchGen(task_args, dataset_label, model_config=None, tokenizer=None, world_size=None, rank=None, seed=None):
    '''
    This example batch generater creates simple MLM training batches
    It take paths to the dataset directories, and produce final iterator that yields tensors of a batch
    It performs file reading, shuffling, tokenization, masking, batching, collating by nesting the iterators in the DataLoader infinibatch library
    arguments:
        task_args: a dict containing parameters for the task
        dataset_label: train, dev, or test
        model_config: model architecture config
        tokenizer: a list of tokenizers
        world_size, rank: GPU world size and rank for distributed training
    Note: this batch generator does not move the batches to the GPU. The caller must do that as desired.
    '''

    role_dict_file = os.path.join(task_args['datadir'], task_args['ROLE_DICT_FILE'])
    role_dict = json.load(open(role_dict_file))
    inv_role_dict = {v: k for k, v in role_dict.items()}
    anon_roles = task_args.get('ANONYMOUS_ROLES', False) # whether to convert all speakers to speaker-0, speaker-1, ...

    dataset_file = os.path.join(task_args['datadir'], task_args['{}_FILE'.format(dataset_label.upper())])
    is_train = dataset_label == 'train'
    tokens_per_batch = task_args['MINI_BATCH'] * task_args['MAX_TRANSCRIPT_WORD']
    batch_read_ahead = task_args['BATCH_READ_AHEAD']
    doc_shuffle_buffer_size = task_args['DOC_SHUFFLE_BUF_SIZE']
    sample_shuffle_buffer_size = task_args['SAMPLE_SHUFFLE_BUFFER_SIZE']
    batch_shuffle_buffer_size = task_args['BATCH_SHUFFLE_BUFFER_SIZE']

    max_padding_ratio = task_args.get('MAX_PADDING_RATIO', 1.0)

    max_gen_length = task_args.get('MAX_GEN_LENGTH', 200)
    max_transcript_len = task_args.get('MAX_TRANSCRIPT_WORD', 8300)
    max_sentence_len = task_args.get('MAX_SENT_LEN', 30)
    max_sentence_num = task_args.get('MAX_SENT_NUM', 400)

    merge_summary_buffer_size = task_args.get('MERGE_SUMMARY_BUFFER_SIZE', 24)
    merge_summary_num = task_args.get('MERGE_SUMMARY_NUM', 1)
    merge_summary_shuffle = task_args.get('MERGE_SUMMARY_SHUFFLE', False)

    ###############################
    # set up rank-aware chunk file path iterator
    # this part can be used as is in all tasks
    ###############################
    # dataset_file is the path to a json file containing dataset information
    data_sets = json.load(open(dataset_file, encoding='utf-8'))

    # get paths to all the chunk files in the source and target dataset dirs
    datasets_chunks = []
    for i, data_set in enumerate(data_sets):
        task = data_set['task']
        dataset_name = data_set['name']
        source = data_set['source']
        # to determine if use relative path to load dataset
        if "USE_REL_DATA_PATH" in task_args:
            source['dataset'] = os.path.join(task_args['datadir'], source['dataset'])
        source_chunk_files = [x for x in os.scandir(source['dataset']) if x.name.endswith('.gz')]  # enumerate all .gz files in the given paths
        source_chunk_files.sort(key=lambda x: x.name)
        if 'target' in data_set:
            target = data_set['target']
            if "USE_REL_DATA_PATH" in task_args:
                target['dataset'] = os.path.join(task_args['datadir'], target['dataset'])

            target_chunk_files = [x for x in os.scandir(target['dataset']) if x.name.endswith('.gz')]  # enumerate all .gz files in the given paths
            target_chunk_files.sort(key=lambda x: x.name)
            assert len(source_chunk_files) == len(target_chunk_files), \
                f"Number of chunk files should be the same in source ({len(source_chunk_files)}) and target ({len(target_chunk_files)}) datasets."
            assert all([s.name == t.name for s, t in zip(source_chunk_files, target_chunk_files)])
        
            datasets_chunks.append([
                {'source': {'dataset': os.path.join(source['dataset'], s.name)},
                 'target': {'dataset': os.path.join(target['dataset'], t.name) if target['dataset'] else None},
                 'task'  : task,
                 'cid'   : i, # corpus id for corpus based metric computation during evaluation
                 'name': dataset_name,
                } for s, t in zip(source_chunk_files, target_chunk_files)
            ])
        else:
            datasets_chunks.append([
                {'source': {'dataset': os.path.join(source['dataset'], s.name)},
                 'task'  : task,
                 'cid'   : i, # corpus id for corpus based metric computation during evaluation
                 'name': dataset_name,
                } for s in source_chunk_files
            ])

    # create an iterator to iterate the chunk file paths in each dataset
    if is_train:
        for dataset_chunks in datasets_chunks:
            dataset_chunks.sort(key=lambda x: x['source']['dataset'])  # make sure file order is always the same, independent of OS
        datasets_chunks.sort(key=lambda  x: x[0]['source']['dataset'])  # make sure file order is always the same, independent of OS

        for i, dataset_chunks in enumerate(datasets_chunks):
            datasets_chunks[i] = iterators.InfinitePermutationSourceIterator(dataset_chunks, seed, shuffle=True, num_instances=world_size, instance_rank=rank)
    else:
        datasets_chunks = [[chunk for dataset_chunks in datasets_chunks for chunk in dataset_chunks]] # flatten the datasets
        datasets_chunks[0].sort(key=lambda  x: x['source']['dataset'])  # make sure file order is always the same, independent of OS
        datasets_chunks[0] = iterators.ChunkedSourceIterator(datasets_chunks[0], num_instances=world_size, instance_rank=rank) # in evaluation mode, the files are iterated once without shuffling, but still with parallelization
    ###############################

    dataset_batch_read_ahead = max(1, batch_read_ahead // len(datasets_chunks))
    dataset_doc_shuffle_buffer_size = max(1, doc_shuffle_buffer_size // len(datasets_chunks))
    dataset_sample_shuffle_buffer_size = max(1, sample_shuffle_buffer_size // len(datasets_chunks))
    dataset_batch_shuffle_buffer_size = max(1, batch_shuffle_buffer_size // len(datasets_chunks))

    ###############################
    # set up document iterator from chunk file iterator
    ###############################
    # use SelectManyIterator to split each chunk file into multiple documents
    def read_docs_from_chunk(chunk):
        # this function is provided to the SelectManyIterator constructor as a callback
        # it takes one item from the source iterator as input (one chunk in this case), and return an iterable (a list of documents), each item in the returned iterable will be yielded by the SelectManyIterator
        docs = []
        doc = []
        cid = chunk['cid']
        task = chunk['task']
        source = chunk['source']
        name = chunk['name']
        with gzip.open(source['dataset'], 'rt', encoding='utf-8') as fs:
            if 'target' in chunk:
                target = chunk['target']
                if target['dataset']:
                    with gzip.open(target['dataset'], 'rt', encoding='utf-8') as ft:
                        for line_s, line_t in zip(fs, ft):
                            line_s, line_t = line_s.strip(), line_t.strip()
                            if line_s != '':
                                # take care of multiple reference, assume line_t is splitted by " ||| "
                                if is_train:
                                    # for train, split references to multiple pairs
                                    line_t_list = line_t.split(" ||| ")
                                else:
                                    # for valid and test, not split
                                    line_t_list = [line_t]
                                    
                                for sub_line_t in line_t_list:
                                    if task == 'sum' and len(doc) >= merge_summary_buffer_size:
                                        docs.append(doc)
                                        doc = []
                                    elif (not task == 'sum') and len(doc) > 0:
                                        docs.append(doc)
                                        doc = []
                                    doc.append({'source': {'sequence'  : line_s},
                                                'target': {'sequence'  : sub_line_t},
                                                'task'  : task,
                                                'cid'   : cid,
                                                'name': name})

            else:
                for line in fs:
                    line = line.strip()
                    if len(doc) > 0:
                        docs.append(doc)
                        doc = []
                    if line != '':
                        doc.append({'source': {'sequence'  : line},
                                    'task'  : task,
                                    'cid'   : cid,
                                    'name'  : name})

        if len(doc) > 0:
            docs.append(doc)
        return docs # each doc in the docs list will be yielded by the SelectManyIterator
    datasets_doc_samples = []
    for dataset_chunks in datasets_chunks:
        datasets_doc_samples.append(iterators.SelectManyIterator(dataset_chunks, read_docs_from_chunk))
    ###############################

    ###############################
    # set up the doc randomizer
    ###############################
    # use BufferedShuffleIterator to shuffle the items from the source iterator
    # We shuffle before the next steps since at startup, shuffling needs to fill a large buffers. Doing expensive operations afterwards will reduce startup time.
    # the principle that determines a proper shuffle_buffer_size is: shuffle_buffer_size >> chunk_size
    if is_train:
        for i, doc_samples in enumerate(datasets_doc_samples):
            seed = _bump_seed(seed)
            datasets_doc_samples[i] = iterators.BufferedShuffleIterator(doc_samples, dataset_doc_shuffle_buffer_size, seed)
    ###############################

    def _parse_tags(parsed_text):
        output = {'word': [],
                  'pos_id': [],
                  'ent_id': []}

        for token in parsed_text:
            #[(token.text,token.idx) for token in parsed_sentence]
            output['word'].append(_str(token.text))
            pos = token.tag_
            output['pos_id'].append(POS[pos] if pos in POS else 0)

            ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
            output['ent_id'].append(ENT[ent] if ent in ENT else 0)

        word_idx = 0
        for sent in parsed_text.sents:
            # output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        assert len(output['word']) > 0

        return output

    def _str(s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    ###############################
    # tokenize all sentences in a doc
    ###############################
    # use SamplingRandomMapIterator because it applies one-to-one mapping (new iterator take one document from source iterator, apply transform, and output it) with checkpointed random state
    def tokenize(rand: Random, doc):
        # this function is provided to the SamplingRandomMapIterator constructor as a callback
        # it takes one item from the source iterator as input, and returns one processed item
        # use the provided Random object for all random operations in the transform, because that random object is checkpointed.
        start = timer()
        for sample in doc:
            if anon_roles:
                sample_role_dict = {}

            source = sample['source']
            if sample['task'] == 'sum':
                # make pseduo meetings
                turns = json.loads(source['sequence'])
                source['sequence'] = []
                sample['meeting'] = []
                for turn in turns:
                    turn["role"] = role_dict.get(sample['name'], 0)
                    sample['meeting'].append(turn)
                    source['sequence'].extend(turn["utt"]["word"])

                target = sample['target']
                target['sequence'] = tokenizer.tokenize(target['sequence'])
                    
            elif sample['task'] == 'meeting':
                data = json.loads(source['sequence'])
                sample['meeting'] = []
                source['sequence'] = []
                
                for turn in data['meeting']:
                    if anon_roles:
                        if turn["role"] not in sample_role_dict:
                            sample_role_dict[turn["role"]] = len(sample_role_dict)
                        turn["role"] = role_dict.get("<speaker {}>".format(sample_role_dict[turn["role"]]), 0)
                    else:
                        turn["role"] = role_dict.get(turn["role"], 0)
                    sample['meeting'].append(turn)
                    assert isinstance(turn["utt"], dict), turn["utt"]
                    source['sequence'].extend(turn["utt"]["word"])
                
                sample['target'] = {}
                summary_str = ' '.join(data['summary'])
                if anon_roles:
                    for role in sample_role_dict:
                        summary_str = summary_str.replace(role, "<speaker {}>".format(sample_role_dict[role]))
                sample['target']['sequence'] = tokenizer.tokenize(summary_str)
                
            else:
                assert False, f"Undefined Task {sample['task']}"

        doc = [sample for sample in doc if len(sample['source']['sequence']) > 0 and ('target' not in sample or sample['target']['sequence'] is None or len(sample['target']['sequence']) > 0)]
        end = timer()
        # print('Tokenize takes {:06.2f} seconds'.format(end-start))
        return doc
    for i, doc_samples in enumerate(datasets_doc_samples):
        seed = _bump_seed(seed)
        datasets_doc_samples[i] = iterators.SamplingRandomMapIterator(doc_samples, transform=tokenize, seed=seed)
    ###############################

    ###############################
    # shuffle samples from documents again
    ###############################
    if is_train:
        for i, samples in enumerate(datasets_doc_samples):
            seed = _bump_seed(seed)
            datasets_doc_samples[i] = iterators.BufferedShuffleIterator(samples, dataset_sample_shuffle_buffer_size, seed)
    ###############################

    def concat_samples_in_doc(doc):
        if len(doc) == 1:
            # return for all meeting dataset and article dataset with one article per sample
            return doc

        concat_sample = {}
        concat_sample['source'] = {'sequence':[]}
        concat_sample['target'] = {'sequence':[]}
        concat_sample['meeting'] = []

        ret_sample_list = []

        count = 0
        for sample in doc:
            for turn in sample['meeting']:
                # take the role add append '-n' for the n-th document
                turn["role"] = role_dict.get(inv_role_dict[turn["role"]]+'-{}'.format(count), 0)
                concat_sample['meeting'].append(turn)
            
            concat_sample['source']['sequence'].extend(sample['source']['sequence'])
            concat_sample['target']['sequence'].extend(sample['target']['sequence'])

            count += 1

            if count >= merge_summary_num:
                if merge_summary_shuffle and count > 1 and is_train:
                    shuffle(concat_sample['meeting']) 
                ret_sample_list.append(concat_sample)
                concat_sample = {}
                concat_sample['source'] = {'sequence':[]}
                concat_sample['target'] = {'sequence':[]}
                concat_sample['meeting'] = []
                count = 0

        return ret_sample_list

    datasets_samples = []
    for doc_samples in datasets_doc_samples:
        datasets_samples.append(iterators.SelectManyIterator(doc_samples, concat_samples_in_doc))

    ###############################
    # batching with dynamic batch size depending on the task
    ###############################
    def dynamic_batch_size(sample):
        if is_train:
            batch_size = tokens_per_batch // (len(sample['source']['sequence']) + len(sample['target']['sequence']) + 1)
        else:
            batch_size = tokens_per_batch // (len(sample['source']['sequence']) + max_gen_length + 1)
        return max(1, batch_size)
    datasets_batches = []
    for i, samples in enumerate(datasets_samples):
        seed = _bump_seed(seed)
        datasets_batches.append(iterators.BucketedReadaheadBatchIterator(samples, read_ahead=dataset_batch_read_ahead, key=lambda x : len(x['source']['sequence']), batch_size=dynamic_batch_size, shuffle=is_train, seed=seed))
    ###############################

    ###############################
    # create a zip iterator on all datasets
    ###############################
    # Use ZipIterator to zip datasets from different datasets. This is to make dataset-dependent tasks distributed evenly
    datasets_batches_zip = iterators.ZipIterator(*tuple(datasets_batches))
    ###############################

    ###############################
    # unzip batches from all datasets
    ###############################
    def unzip(datasets_batche):
        return [batche for batche in datasets_batche]
    batches = iterators.SelectManyIterator(datasets_batches_zip, unzip)
    ###############################

    ###############################
    # set up the batch randomizer
    ###############################
    seed = _bump_seed(seed)
    batches = iterators.BufferedShuffleIterator(batches, batch_shuffle_buffer_size, seed)
    ###############################

    def _pad_batch(batch):
        # padding and generate final batch
        x_sent_batch = []
        x_role_batch = []
        x_pos_batch = []
        x_ent_batch = []
        y_sent_batch = []

        encoder_tokens, decoder_tokens = [], []

        for datum in batch:
            x_sent = []
            x_role = []
            x_pos = []
            x_ent = []

            sample_input_tokens = []

            total_word_len = 0
            total_sent_len = 0

            assert len(datum['meeting']) > 0
            for m in datum['meeting']: # each m is actually a turn
                words = m['utt']['word']
                pos = m['utt']['pos_id']
                ent = m['utt']['ent_id']
                L = len(words)
                # assert L < max_transcript_len, "a turn {} is longer than max_transcript_len".format(' '.join(words))
                if L > max_transcript_len:
                    # this is rarely happpened when a turn is super long
                    # in this case we just skip it to save memory
                    continue
                if total_word_len + L > max_transcript_len or total_sent_len + 1 > max_sentence_num:
                    break
                
                sample_input_tokens.extend(words)

                for i in range(math.ceil(L/max_sentence_len)):
                    x_role.append(m['role'])
                    sub_words = words[i*max_sentence_len:min((i+1)*max_sentence_len, L)]
                    x_sent.append([tokenizer.bos_token] + sub_words + [tokenizer.eos_token])
                    x_pos.append([0] + pos[i*max_sentence_len:min((i+1)*max_sentence_len, L)] + [0])
                    x_ent.append([0] + ent[i*max_sentence_len:min((i+1)*max_sentence_len, L)] + [0])

                    total_sent_len += 1
                
                total_word_len += L

            if is_train: # training
                y_sent = [tokenizer.bos_token] + datum['target']['sequence'][:max_gen_length] + [tokenizer.eos_token]
            else:
                y_sent = [tokenizer.bos_token] + datum['target']['sequence'] + [tokenizer.eos_token]
            
            if len(x_sent) > 0:
                # this could be false when there is a single but very long turn
                x_sent_batch.append(x_sent)
                x_role_batch.append(x_role)
                x_pos_batch.append(x_pos)
                x_ent_batch.append(x_ent)
                y_sent_batch.append(y_sent)

                encoder_tokens.append(sample_input_tokens)
                decoder_tokens.append(y_sent)

        if len(x_sent_batch) == 0:
            # this could happen when there is a single but very long turn
            # leading the whole batch with all instances filtered
            return None

        # count max length
        x_max_doc_len = max([len(s) for s in x_sent_batch])
        x_max_sent_len = max([max([len(t) for t in s]) for s in x_sent_batch])
        y_max_len = max([len(s) for s in y_sent_batch])
        x_role_max_len = max([len(s) for s in x_role_batch])
        actual_size = len(x_sent_batch)

        actual_tokens_per_batch = actual_size * (x_max_doc_len*x_max_sent_len+y_max_len)

        # if the actual batch size is too larger than expected because of skewed length
        if (actual_tokens_per_batch/tokens_per_batch) > (max_padding_ratio + 1) and is_train:
            return None
 
        # create tensors
        x_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(tokenizer.pad_token_id)
        x_pos_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(0)
        x_ent_tensor = torch.LongTensor(actual_size, x_max_doc_len, x_max_sent_len).fill_(0)
        x_role_tensor = torch.LongTensor(actual_size, x_role_max_len).fill_(0)
        y_tensor = torch.LongTensor(actual_size, y_max_len).fill_(tokenizer.pad_token_id)

        for i in range(len(x_sent_batch)):
            for j in range(len(x_sent_batch[i])):
                x_tensor[i, j, :len(x_sent_batch[i][j])] = torch.LongTensor(tokenizer.convert_tokens_to_ids(x_sent_batch[i][j])) 
                y_tensor[i, :len(y_sent_batch[i])] = torch.LongTensor(tokenizer.convert_tokens_to_ids(y_sent_batch[i]))

            for j in range(len(x_pos_batch[i])):
                x_pos_tensor[i, j, :len(x_pos_batch[i][j])] = torch.LongTensor(x_pos_batch[i][j])
            for j in range(len(x_ent_batch[i])):
                x_ent_tensor[i, j, :len(x_ent_batch[i][j])] = torch.LongTensor(x_ent_batch[i][j])

            x_role_tensor[i, :len(x_role_batch[i])] = torch.LongTensor(x_role_batch[i])

        return { 
                'encoder_input_ids'  : x_tensor,
                'encoder_input_roles': x_role_tensor,
                'encoder_input_pos'  : x_pos_tensor,
                'encoder_input_ent'  : x_ent_tensor,
                'decoder_input_ids'  : y_tensor,
                'encoder_tokens': encoder_tokens,
                'decoder_tokens': decoder_tokens
                }

    ###############################
    # collate samples into padded rectangular tensors
    ###############################
    def collate(batch):
        batch = _pad_batch(batch)

        if batch is None:
            ret_batches = []
        else:
            ret_batches = [batch]

        return ret_batches

    ###############################
    # collate samples into padded rectangular tensors
    ###############################
    batches = iterators.SelectManyIterator(batches, collate)
    ###############################

    return batches

