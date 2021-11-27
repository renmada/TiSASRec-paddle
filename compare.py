import os
import time

import numpy as np
import paddle
import pickle
import argparse
import torch

import pandas as pd

from model import TiSASRec
from tqdm import tqdm
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--seed', default=128, type=int)

# paddle.set_device('gpu')
args = parser.parse_args()
set_seed(args.seed)

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset

num_batch = len(user_train) // args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

try:
    relation_matrix = pickle.load(
        open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'rb'))
except:
    relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
    pickle.dump(relation_matrix,
                open('data/relation_matrix_%s_%d_%d.pickle' % (args.dataset, args.maxlen, args.time_span), 'wb'))


paddle.set_device('cpu')


def sample(user, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    time_seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    nxt = user_train[user][-1][0]

    idx = maxlen - 1
    ts = set(map(lambda x: x[0], user_train[user]))
    for i in reversed(user_train[user][:-1]):
        seq[idx] = i[0]
        time_seq[idx] = i[1]
        pos[idx] = nxt
        if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
        nxt = i[0]
        idx -= 1
        if idx == -1:
            break
    time_matrix = relation_matrix[user]
    return (user, seq, time_seq, time_matrix, pos, neg)


def compare(paddle_tensor, torch_tensor, name):
    diff = np.abs(paddle_tensor.cpu().numpy() - torch_tensor.cpu().numpy()).mean()
    print('{}的精度误差为{}'.format(name, diff))


if __name__ == '__main__':
    paddle_model = TiSASRec(usernum, itemnum, itemnum, args)

    u, seq, time_seq, time_matrix, pos, neg = sample(1, args.maxlen)
    u, seq, time_seq, time_matrix, pos, neg = [np.array([x]).repeat(2, 0) for x in [u, seq, time_seq, time_matrix, pos, neg]]
    state_dict = {}
    for k, v in paddle_model.named_parameters():
        state_dict[k] = paddle.full_like(v, 0.01)
    paddle_model.set_state_dict(state_dict)
    paddle_model.eval()
    with paddle.no_grad():
        paddle_pos_logits, paddle_neg_logits = paddle_model(seq, time_matrix, pos, neg)
        indices = paddle.to_tensor(pos != 0)
        pos_labels, neg_labels = paddle.ones(paddle_pos_logits.shape), paddle.zeros(paddle_neg_logits.shape)
        loss = paddle.nn.BCEWithLogitsLoss()(paddle.masked_select(paddle_pos_logits, indices),
                                             paddle.masked_select(pos_labels, indices))
        loss += paddle.nn.BCEWithLogitsLoss()(paddle.masked_select(paddle_neg_logits, indices),
                                              paddle.masked_select(neg_labels, indices))
        paddle_loss = loss

    from torch_model import TiSASRec

    torch_model = TiSASRec(usernum, itemnum, itemnum, args)
    for p in torch_model.parameters():
        p.data = torch.full_like(p, 0.01)
    torch_model.eval()
    with torch.no_grad():
        torch_pos_logits, torch_neg_logits = torch_model(u, seq, time_matrix, pos, neg)
    pos_labels, neg_labels = torch.ones(torch_pos_logits.shape), torch.zeros(torch_neg_logits.shape)
    indices = torch.where(torch.from_numpy(pos != 0))
    loss = torch.nn.BCEWithLogitsLoss()(torch_pos_logits[indices], pos_labels[indices])
    loss += torch.nn.BCEWithLogitsLoss()(torch_neg_logits[indices], neg_labels[indices])
    torch_loss = loss

    compare(paddle_pos_logits, torch_pos_logits, 'pos_logits')
    compare(paddle_neg_logits, torch_neg_logits, 'neg_logits')
    compare(paddle_loss, torch_loss, 'loss')


