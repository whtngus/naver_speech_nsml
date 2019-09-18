#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import wavio

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


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=20, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)
        
        y_hat = logit.max(-1)[1]
        if args.criterion == 'CrossEntropy':
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        elif args.criterion == 'CTC':
            loss = criterion(logit.contiguous().transpose(0,1).log_softmax(2), target.contiguous(), feat_lengths, script_lengths)

        total_loss += loss.item()
        total_num += sum(feat_lengths)
        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch:{:4d}/{:4d} loss: {:.4f} cer: {:.2f}  elapsed: {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        # elapsed,
                        epoch_elapsed, train_elapsed))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count, train_step__loss=total_loss/total_num,
                        train_step__cer=total_dist/total_length)
        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0


def evaluate(model, dataloader, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]
    
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))

            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

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

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset

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
    arg('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    arg('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    arg('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    arg('--seed', type=int, default=1, help='random seed (default: 1)')
    arg('--save_name', type=str, default='model', help='the name of model in nsml or local')
    arg('--mode', type=str, default='train')
    arg("--pause", type=int, default=0)
    arg("--print_batch", type=int, default=40)
    arg('--optimizer', type=str, default='Adam') # SGD
    arg('--scheduler', type=str, default='plateau', help='scheduler in cosine, steplr, plateau')
    arg('--criterion', type=str, default='CrossEntropy', help='choose loss function to optimize on')
    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./hackathon.labels')
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    model = nn.DataParallel(model).to(device)

    optimizer = build_optimizer(args, model, args.lr)
    scheduler = build_scheduler(args, optimizer)
    
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    elif args.criterion == 'CTC':
        criterion = nn.CTCLoss()

    bind_model(model, optimizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"

            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)

    logger.info('start')

    train_begin = time.time()

    begin_epoch = 0
    best_loss = 1e10
    best_score = 1.0
    best_score_epoch = 0

    for epoch in range(begin_epoch, args.max_epochs):
        start_time = time.time()
        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        train_loader.start()

        train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, args.print_batch, args.teacher_forcing)
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join()

        valid_queue = queue.Queue(args.workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
        valid_loader.start()

        eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        valid_loader.join()

        if eval_cer < best_score:
            best_score = eval_cer
            best_score_epoch = epoch+1
            nsml.save("best_score")

        elapsed = time.time() - start_time

        lr = [_['lr'] for _ in optimizer.param_groups]
        if args.scheduler == 'plateau':
            scheduler.step(best_score)
        else:
            scheduler.step()

        print("Epoch {}/{}  train_loss: {:.4f}  train_cer: {:.4f}  eval_loss {:.4f}  eval_cer: {:.4f}  lr: {:.6f}  best_score_epoch: {}  elapsed: {:.0f}".format(
        epoch+1, args.max_epochs, train_loss, train_cer, eval_loss, eval_cer, lr[0], best_score_epoch, elapsed))

        nsml.report(True,
            step=epoch, train_epoch__loss=round(train_loss, 4), train_epoch__cer=round(train_cer, 4),
            eval__loss=round(eval_loss,), eval__cer=round(eval_cer, 4))

if __name__ == "__main__":
    main()
