# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import math
import re
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    '''
     Input:
       x: n_state-dim
     Output:
       o: n_state-dim
    '''
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


'''
 Convolution
 nx is the last input dim
 nf is the last output dim
'''
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)
        self.b = Parameter(torch.zeros(nf))
    '''
      Input: 
         x: batch x len x nx
      Output: 
         x: batch x len x nf
    '''
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        x = x.view(*size_out)
        return x


class PositionalEmbedding(nn.Module):
    def __init__ (self, opt, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0)/demb))
        self.pos_discount = float(opt['TRANSFORMER_POS_DISCOUNT'])
        self.register_buffer('inv_freq', inv_freq)

    '''
     Input: 
        pos_seq: len
     Output:
        pos_emb: len x demb
    '''
    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1) / self.pos_discount
        return pos_emb

'''
  Splitter
'''
class Splitter(nn.Module):
    def __init__(self, nx):
        super(Splitter, self).__init__()
        self.nx = nx
        self.augmenter = Conv1D(nx * 3, nx)

    '''
     Input: 
        x: batch x len x nx
     Output:
        query,key,value: batch x len x nx
    '''
    def forward(self, x):
        x = self.augmenter(x) 
        # x: batch x len x (3 x nx)

        query, key, value = x.split(self.nx, dim=2) 
        # query,key,value: batch x len x nx

        return query, key, value

'''
  Multi-head Attention
'''
class Attention(nn.Module):
    '''
     nx: input dimension
    '''
    def __init__(self, nx, opt):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        n_head = int(opt['TRANSFORMER_HEAD'])
        resid_pdrop = opt['TRANSFORMER_RESIDUAL_DROPOUT']
        attn_pdrop = opt['TRANSFORMER_ATTENTION_DROPOUT']
        use_cuda = opt['cuda']

        assert n_state % n_head == 0
        # if mask is needed, uncomment this
        self.maxlen = 2048 # beyond this scale 
        self.mask = Variable(torch.tril(torch.ones(self.maxlen, self.maxlen)).view(1, 1, self.maxlen, self.maxlen), requires_grad=False)
        if use_cuda:
            self.mask.cuda()
        self.n_head = n_head
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.use_cuda = use_cuda

    '''
      Input:
        q: batch x n_head x len x dim
        k: batch x n_head x dim x kv_len
        v: batch x n_head x kv_len x dim
        x_mask: batch x kv_len # key and value's mask (if not None, used for encoder's self-attention and decoder's src-tgt attention)
        one_dir_visible: only sees previous history  (used for decoder's self-attention)
        return_attn_weight: if true, also return the attention weights
      Output:
        a: batch x n_head x len x n_state x dim
        attn_weight (if return_attn_weight): attn_weight: batch x n_head x len x kv_len
    '''
    def _attn(self, q, k, v, x_mask, one_dir_visible, return_attn_weight):
        w = torch.matmul(q, k)
        # batch x n_head x len x kv_len
        w = w / math.sqrt(v.size(-1))

        mask = None
        if one_dir_visible: # mask "seeing the future"
            if w.size(-2) <= self.maxlen and w.size(-1) <= self.maxlen:
                mask = self.mask[:, :, :w.size(-2), :w.size(-1)]
                if self.use_cuda:
                    mask.cuda()
            else:
                mask = Variable(torch.tril(torch.ones(w.size(-2), w.size(-1))).view(1, 1, w.size(-2), w.size(-1)), requires_grad=False)
                if self.use_cuda:
                    mask.cuda()

        if x_mask is not None:
            mask = x_mask.unsqueeze(1).unsqueeze(1).expand_as(w).float()
            # batch x n_head x len x kv_len

        if mask is not None:
            w = w * mask + -1e9 * (1 - mask)  
            
        w_prob = nn.Softmax(dim=-1)(w)
        w_prob = self.attn_dropout(w_prob)
        if return_attn_weight:
            return torch.matmul(w_prob, v), w
        else:
            return torch.matmul(w_prob, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    '''
      Input:
        x: batch x len x dim
      Output:
        not k: batch x n_head x (dim/n_head)  x len    
        k: batch x n_head x len x (dim/n_head)
    '''
    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)
    
    '''
     Input: 
       query: batch x len x n_state
       key, value: batch x kv_len x n_state
       x_mask: batch x kv_len # key and value's mask (if not None, used for encoder's self-attention and decoder's src-tgt attention)
    one_dir_visible: only sees previous history  (used for decoder's self-attention)
    return_attn_weight: if true, also return the attention weights
     Output: 
       a: batch x len x n_state
       attn_weight (if return_attn_weight): batch x len x kv_len
    '''
    def forward(self, query, key, value, x_mask, one_dir_visible=False, return_attn_weight=False):
        query = self.split_heads(query) 
        # batch x n_head x len x (n_state/n_head)
        
        key = self.split_heads(key, k=True) 
        # batch x n_head x (n_state/n_head) x kv_len
        
        value = self.split_heads(value) 
        # batch x n_head x kv_len x (n_state/n_head)

        out = self._attn(query, key, value, x_mask, one_dir_visible, return_attn_weight) 

        if return_attn_weight:
            a, attn_weight = out
            # a: batch x n_head x len x (n_state/n_head)
            # attn_weight: batch x n_head x len x kv_len
            attn_weight = attn_weight.permute(0, 2, 3, 1).contiguous()
            # batch x len x kv_len x n_head
            attn_weight = torch.sum(attn_weight, dim=3)
            # batch x len x kv_len
        else:
            a = out
            # batch x n_head x len x (n_state/n_head)
        
        a = self.merge_heads(a) 
        # batch x len x n_state
        
        a = self.c_proj(a) 
        # batch x len x n_state
        
        a = self.resid_dropout(a) 
        # batch x len x n_state

        if return_attn_weight:
            return a, attn_weight
        else:
            return a


'''
 Two-layer network
'''
class MLP(nn.Module):
    '''
     Input: 
       n_state: intermediate dim
    '''
    def __init__(self, n_state, opt):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = int(opt['transformer_embed_dim'])
        resid_pdrop = opt['TRANSFORMER_RESIDUAL_DROPOUT']
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.dropout = nn.Dropout(resid_pdrop)

    '''
      Input: 
       x: batch x len x nx
      Output: batch x len x nx
    '''
    def forward(self, x):
        h = F.relu(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

'''
 One encoder block of transformer
'''
class EncoderBlock(nn.Module):
    def __init__(self, opt):
        super(EncoderBlock, self).__init__()
        nx = int(opt['transformer_embed_dim'])
        self.one_dir_visible = False
        if 'transformer_encoder_one_dir_visible' in opt:
            self.one_dir_visible = opt['transformer_encoder_one_dir_visible']
        self.splitter = Splitter(nx)
        self.attn = Attention(nx, opt)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, opt)
        self.ln_2 = LayerNorm(nx)

    '''
     Input: 
       x: batch x len x n_state
       x_mask: batch x len (1 means there's something)
     Output: 
       h: batch x len x n_state
    '''
    def forward(self, x, x_mask):
        query, key, value = self.splitter(x)
        if self.one_dir_visible:
            # in this case, use triangle masking, as it's one_direction
            a = self.attn(query, key, value, None, one_dir_visible=True)
        else:
            # in this case, use x_mask for attention masking
            a = self.attn(query, key, value, x_mask, one_dir_visible=False)
            
        n = self.ln_1(x + a) # residual
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h

'''
 One encoder block of transformer
'''
class DecoderBlock(nn.Module):
    def __init__(self, opt):
        super(DecoderBlock, self).__init__()
        nx = int(opt['transformer_embed_dim'])
        self.decoder_splitter = Splitter(nx)
        self.self_attn = Attention(nx, opt)
        self.cross_attn = Attention(nx, opt)
        self.ln_1 = LayerNorm(nx)
        self.ln_2 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, opt)
        self.ln_3 = LayerNorm(nx)

    '''
     Input:        
       x_mask: batch x len, mask for encoder's input
       y: batch x len x n_state (decoder part)
       enc_key: batch x encoder_len x n_state
       enc_value: batch x encoder_len x n_state
       lang_model: whether it's for language model training (no encoder part is used)
     Output: 
       h: batch x len x n_state
    '''
    def forward(self, x_mask, y, enc_key, enc_value, lang_model=False):
        query, key, value = self.decoder_splitter(y)
        # batch x len x n_state

        # self-attention
        a = self.self_attn(query, key, value, None, one_dir_visible=True)
        # batch x len x n_state

        n = self.ln_1(y + a) # residual

        # seq2seq
        if not lang_model:
            # src-tgt attention
            o = self.cross_attn(n, enc_key, enc_value, x_mask)
            p = self.ln_2(n + o) # residual
            # batch x len x n_state
        else: # language model
            p = n

        m = self.mlp(p)
        h = self.ln_3(p + m)
        return h

'''
  Embedder 
'''
class Embedder(nn.Module):
    '''
      Input: 
        vocab: size of vocabulary
    '''
    def __init__(self, opt, embed=None): 
        super(Embedder, self).__init__()
        n_state = int(opt['transformer_embed_dim']) # n_state
        embed_dropout_rate = opt['TRANSFORMER_EMBED_DROPOUT']
        if embed is None:
            self.embed = nn.Embedding(opt['vocab_size'], n_state)
            nn.init.normal_(self.embed.weight, std=0.02)
        else:
            self.embed = embed
        self.drop = nn.Dropout(embed_dropout_rate)
        self.pos_emb = PositionalEmbedding(opt, n_state)
        self.use_cuda = opt['cuda']

    '''
       Input:
        x: batch x len    (word_id)
       Output:
        h: batch x len x n_state
    '''
    def forward(self, x):
        x_emb = self.embed(x)
        batch_size = x.shape[0]
        x_len = x.shape[1]
        x_pos = self.pos_emb(torch.arange(x_len).type(torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor)) # len x n_state
        x_pos = Variable(x_pos.unsqueeze(0).repeat(batch_size, 1, 1), requires_grad=False)
        if self.use_cuda:
            x_pos.cuda()
        x_input = x_emb + x_pos
        h = self.drop(x_input)
        return h


'''
  Transformer encoder
'''
class TransformerEncoder(nn.Module):
    '''
      Input: 
        embed: (if not None) pre-computed vocab embeddings
    '''
    def __init__(self, opt, embed=None):
        super(TransformerEncoder, self).__init__()
        vocab = int(opt['vocab_size'])
        n_state = int(opt['transformer_embed_dim'])
        n_layer = int(opt['TRANSFORMER_LAYER'])
        if 'vae_z_scale_factor' in opt:
            self.vae_z_scale_factor = float(opt['vae_z_scale_factor'])

        self.embedder = Embedder(opt, embed)
        block = EncoderBlock(opt)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.use_cuda = opt['cuda']

    '''
      Input: 
        x: batch x len (word_id)
        z (optional): batch x len x n_state (for VAE)
      Output: 
        h: batch x len x n_state (word_id)  
    '''
    def forward(self, x, z=None):
        x_mask = ~x.eq(0) # 1 is PAD_id
        x_mask = x_mask.type(torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor)
            
        h = self.embedder(x)
        if z is not None:
            z *= self.vae_z_scale_factor
            h += z
            
        for block in self.blocks:
            h = block(h, x_mask)
        return h


'''
  Transformer decoder
'''
class TransformerDecoder(nn.Module):
    '''
      Input: 
        embed: (if not None) pre-computed vocab embeddings
    '''
    def __init__(self, opt, embed=None):
        super(TransformerDecoder, self).__init__()
        self.opt = opt
        vocab_size = int(opt['vocab_size'])
        n_state = int(opt['transformer_embed_dim']) # n_state
        n_layer = int(opt['TRANSFORMER_LAYER'])
        self.embedder = Embedder(opt, embed)
        self.encoder_splitter = Splitter(n_state)
        block = DecoderBlock(opt)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        if embed is None:
            self.linear = Conv1D(vocab_size, n_state)
        else:
            self.linear = nn.Linear(n_state, vocab_size, bias = False)
            if 'FINETUNE_RETRAIN_SOFTMAX' not in opt: # if FINETUNE_RETRAIN_SOFTMAX, linear needs to be seperately trained
                self.linear.weight = embed.weight # share weight
        self.use_coda = opt['cuda']

    '''
      Input: 
        x: batch x encoder_len (word id)
        x_out: batch x encoder_len x n_state
        y: batch x len (word_id)   (decoder part)
        lang_model: whether it's for language model training (no encoder part is used)
      Output: 
        prob: batch x len x vocab_size (probabilities after softmax)
    '''
    def forward(self, x, x_out, y, lang_model=False):
        # seq2seq
        if not lang_model:
            _, enc_key, enc_value = self.encoder_splitter(x_out)
            # enc_key: batch x encoder_len x n_state
            # enc_value: batch x encoder_len x n_state

            x_mask = ~x.eq(0) # 1 is PAD_id
            x_mask = x_mask.type(torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor)
        else:
            enc_key = None
            enc_value = None
            x_mask = None

        h = self.embedder(y)
        for block in self.blocks:
            h = block(x_mask, h, enc_key, enc_value, lang_model)
        prob = F.softmax(self.linear(h), dim=-1)
        return prob


class TransformerBeam():
    '''
     Input: 
      encoder: TransformerEncoder class
      decoder: TransformerDecoder class
      begin_id: word id of '<BEGIN>'
      vocab: list of words
    '''
    def __init__(self, opt, encoder, decoder, begin_id, vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt
        self.max_sent_len = int(opt['max_sent_len'])
        self.begin_id = begin_id
        self.vocab = vocab
        self.beam_width = int(opt['beam_width'])
        self.use_cuda = opt['cuda']

    # each candidate is (idx, prob, 0/1, position/wordid)
    def merge_candidates(self, cand_A, cand_B):
        C = []
        pA, lA, pB, lB = 0, len(cand_A), 0, len(cand_B)
        lC = 0
        while (pA < lA or pB < lB) and (lC < self.beam_width):
            if pA < lA and (pB >= lB or cand_A[pA][1] > cand_B[pB][1]):
                C.append(cand_A[pA])
                pA += 1
            else:
                C.append(cand_B[pB])
                pB += 1
            lC += 1
        return C    

    '''
     Input:
      x = batch * encoder_len (word_ids)   encoder's input
      k: top-k sampling
     Output:
      sents: list of words, with batch items, each one with up to beam_width (sentence, log_prob), each sentence with up to max_sent_len_word words
    '''
    def topk(self, x, k):
        batch_size = x.shape[0]
        x_len = x.shape[1]
        x_out = self.encoder(x)
        # x_out: batch x encoder_len x n_state

        # sent_ids is the words for each of the batch_size sentences
        sent_ids = []
        for i in range(batch_size):
            sent_ids.append([self.begin_id])

        topk = 1
        MIN_GEN_LENGTH = 45
        if 'MIN_GEN_LENGTH' in self.opt:
            MIN_GEN_LENGTH = int(self.opt['MIN_GEN_LENGTH'])
        for l in range(self.max_sent_len):
            y = Variable(torch.LongTensor(sent_ids)) # batch_size x l
            if self.use_cuda:
                y.cuda()
            decoder_outputs = self.decoder(x, x_out, y)
            probs = decoder_outputs[:, -1, :] # batch_size x vocab_size (only take the last output)
            for i in range(batch_size):
                topk_probs, _ = torch.topk(probs[i], k)
                threshold = float(topk_probs[-1])
                probs[i][probs[i] < threshold] = 0.

            samples = torch.multinomial(probs, 2) # sample 2 since the first one may be <END>
            for i in range(batch_size):
                if l < MIN_GEN_LENGTH and self.vocab[int(samples[i, 0])] == '<END>':
                    sent_ids[i].append(int(samples[i, 1]))
                else:
                    sent_ids[i].append(int(samples[i, 0]))

        sents = []
        for i in range(batch_size):
            utt = []
            for j in range(len(sent_ids[i])):
                w = self.vocab[sent_ids[i][j]]
                if w == '<BEGIN>':
                    continue
                if w == '<END>':
                    break
                utt.append(w)
            sents.append([(utt, 0)])

        return sents

    '''
     Input:
      x = batch * encoder_len (word_ids)   encoder's input
     Output:
      sents: list of words, with batch items, each one with up to beam_width (sentence, log_prob), each sentence with up to max_sent_len_word words
    '''
    def beam_search(self, x):
        batch_size = x.shape[0]
        x_len = x.shape[1]
        x_out = self.encoder(x)
        # x_out: batch x encoder_len x n_state

        sents = []
        topk = 1
        history_nodes = [{}]
        end_nodes = {}
        for idx in range(batch_size):
            start_node = BeamSearchNode([self.begin_id], 0, 1)
            history_nodes[0][idx] = [start_node]
            end_nodes[idx] = []
            
        for l in range(self.max_sent_len):
            last_nodes = history_nodes[-1]
            if sum([len(l) for i, l in last_nodes.items()]) == 0: # no nodes left
                break
            ys = []
            x_outs = []
            xs = []
            for idx in range(batch_size):
                ys.extend([node.word_ids for node in last_nodes[idx]])
                x_outs.extend([x_out[idx, :, :].unsqueeze(0) for node in last_nodes[idx]])
                xs.extend([x[idx, :].unsqueeze(0) for node in last_nodes[idx]])

            ys = Variable(torch.LongTensor(ys)) # N x l
            if self.use_cuda:
                ys.cuda()
            x_outs = torch.cat(x_outs, dim = 0) # N x x_len x n_state
            xs = torch.cat(xs, dim = 0) # N x x_len
            probs = self.decoder(xs, x_outs, ys)
            log_probs = torch.log(probs[:, -1, :] + 1e-15) # N x vocab_size (only take the last output)

            history_nodes.append({})
            p = 0
            for idx in range(batch_size):
                history_nodes[-1][idx] = []
                N = len(last_nodes[idx])
                if N == 0:
                    continue
                log_prob = log_probs[p:p+N]
                p += N
                # log_prob = N x extended_vocab_size

                # generate
                candidates = []
                for k in range(N):
                    logprobs, ids = torch.topk(log_prob[k], self.beam_width)
                    candidates = self.merge_candidates(candidates, [(k, p, d) for p, d in zip(logprobs, ids)])

                candidates = candidates[:self.beam_width]
                extended_nodes_in_last_nodes = set()
                for k in range(len(candidates)):
                    h, logp, next_word_id = candidates[k] # h means "the h-th node in last_nodes"
                    logp = float(logp)
                    next_word_id = int(next_word_id)                 
                    prev_node = last_nodes[idx][h]
                    next_wordids = prev_node.word_ids + [next_word_id]
                    next_word = self.vocab[next_word_id]

                    next_node = BeamSearchNode(next_wordids, prev_node.log_prob + logp, prev_node.length + 1)
                    if next_node.duplicate == False:  # no duplicate trigram generated
                        extended_nodes_in_last_nodes.add(h)
                        if next_word == '<END>' or l == self.max_sent_len - 1:
                            end_nodes[idx].append((next_node.eval(), next_node))
                        else:
                            history_nodes[-1][idx].append(next_node)

                special_words = ["<PAD>", "<UNK>", "<s>", "</s>", "<BEGIN>", "<END>"]
                for k in range(N):
                    if k not in extended_nodes_in_last_nodes:
                        node = last_nodes[idx][k]
                        effective_word_count = sum([1 for x in node.word_ids if self.vocab[x] not in special_words])
                        if effective_word_count >= 5:
                            end_nodes[idx].append((node.eval(), node))

        MIN_GEN_LENGTH = 45
        if 'MIN_GEN_LENGTH' in self.opt:
            MIN_GEN_LENGTH = int(self.opt['MIN_GEN_LENGTH'])
        for idx in range(batch_size):
            t = len([w for w in end_nodes[idx] if w[1].length > MIN_GEN_LENGTH])
            if t > 0:
                end_nodes[idx] = [w for w in end_nodes[idx] if w[1].length > MIN_GEN_LENGTH]

            end_nodes[idx].sort(key = lambda tup: tup[0], reverse=True)
            candidates = []
            for score, node in end_nodes[idx][:topk]:
                utt = [self.vocab[x] for x in node.word_ids]
                utt = [x for x in utt if x not in ["<BEGIN>", "<END>"]]
                candidates.append((utt, score))
            if len(candidates) == 0:
                candidates.append(('', 0))
            sents.append(candidates)

        return sents


class BeamSearchNode(object):
    def __init__(self, word_ids, log_prob, length):
        self.word_ids = word_ids
        self.log_prob = log_prob
        self.length = length
        
        trigram_set = set()
        self.duplicate = False

        for i in range(2, len(word_ids)):
            trigram = str(word_ids[i - 2]) + ' ' + str(word_ids[i - 1]) + ' ' + str(word_ids[i])
            if trigram in trigram_set:
                self.duplicate = True
                break
            trigram_set.add(trigram)

    def eval(self):
        return self.log_prob / float(self.length - 1.0 + 1e-6)

    def __lt__(self, other):
         return self.length < other.length


