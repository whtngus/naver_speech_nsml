#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import wavio

from collections import OrderedDict        
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev 

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq
from optimizer import build_optimizer, build_scheduler
from utils import seed_everything

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET
import sys
import warnings 
warnings.filterwarnings('ignore')

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        state_dict = state['model']
        
        try:
            model.load_state_dict(state_dict)
            if 'optimizer' in state and optimizer:
                optimizer.load_state_dict(state['optimizer'])
        except:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v

            model.load_state_dict(new_state_dict)
            if 'optimizer' in state and optimizer:
                optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state_dict = {
            'model': model.state_dict()
        }
        if optimizer:
            state_dict['optimizer'] = optimizer.state_dict()
        torch.save(state_dict, os.path.join(filename, 'model.pt'))
        print("Model Saved!")

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_spectrogram_feature(wav_path).unsqueeze(0)
        input = input.to(device)    

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) 


def main():
    
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    arg = parser.add_argument
    arg('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    arg('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    arg('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    arg('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    arg('--use_attention', action='store_true', help='use attention between encoder-decoder (default: False)')
    arg('--batch_size', type=int, default=4, help='batch size in training (default: 32)')
    arg('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    arg('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    arg('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    arg('--weight_decay', type=float, default=25e-6, help='optimizer weight decay')
    arg('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    arg('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    arg('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    arg('--seed', type=int, default=1, help='random seed (default: 1)')
    arg('--save_name', type=str, default='model', help='the name of model in nsml or local')
    arg('--mode', type=str, default='train')
    arg("--pause", type=int, default=0)
    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./hackathon.labels')
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    seed_everything(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # N_FFT: defined in loader.py
    feature_size = N_FFT / 2 + 1
    
    enc = EncoderRNN(feature_size, args.hidden_size,
                     input_dropout_p=args.dropout, dropout_p=args.dropout,
                     n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
                     SOS_token, EOS_token,
                     n_layers=args.layer_size, rnn_cell='gru', bidirectional=args.bidirectional,
                     input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)

    model = Seq2seq(enc, dec)
    model.flatten_parameters()

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    model = model.to(device)
    # model = nn.DataParallel(model).to(device)

    bind_model(model)

    if args.pause == 1: 
        nsml.paused(scope=locals())

    bTrainmode = False
    if args.mode == 'train':
        bTrainmode = True

        nsml.load(checkpoint='best_score', session='team117/sr-hack-2019-dataset/111')
        nsml.save('best')
        exit()


if __name__ == "__main__":
    main()
