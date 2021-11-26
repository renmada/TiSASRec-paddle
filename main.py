import os
import time
import paddle
import pickle
import argparse

import pandas as pd

from model import TiSASRec
from tqdm import tqdm
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
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

paddle.set_device('gpu')
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

sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen,
                      n_workers=3)
model = TiSASRec(usernum, itemnum, itemnum, args)
print(usernum, itemnum)
model.train()  # enable model training
epoch_start_idx = 1

bce_criterion = paddle.nn.BCEWithLogitsLoss()
adam_optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, beta1=0.9, beta2=0.98)

T = 0.0
t0 = time.time()
best = 0

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    for step in range(num_batch):
        u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()  # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
        pos_logits, neg_logits = model(seq, time_matrix, pos, neg)
        pos_labels, neg_labels = paddle.ones(pos_logits.shape), paddle.zeros(neg_logits.shape)
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        adam_optimizer.clear_grad()
        indices = paddle.to_tensor(pos != 0)
        loss = bce_criterion(paddle.masked_select(pos_logits, indices), paddle.masked_select(pos_labels, indices))
        loss += bce_criterion(paddle.masked_select(neg_logits, indices), paddle.masked_select(neg_labels, indices))
        for param in model.item_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * paddle.norm(param)
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                         loss.item()))  # expected 0.4~0.6 after init few epochs

    if epoch % 20 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
              % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
        if t_test[1] > best:
            best = t_test[1]
            paddle.save(model.state_dict(), os.path.join(args.dataset + '_' + args.train_dir, 'best_model.pdparams'))
        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()
        model.train()

f.close()
sampler.close()

print("Export model")
model.set_state_dict(paddle.load(os.path.join(args.dataset + '_' + args.train_dir, 'best_model.pdparams')))
paddle.jit.save(
    model,
    'PI',
    input_spec=[
        paddle.static.InputSpec(shape=seq.shape, dtype='int32'),
        paddle.static.InputSpec(shape=time_matrix.shape, dtype='int32'),
        paddle.static.InputSpec(shape=pos.shape, dtype='int32'),
        paddle.static.InputSpec(shape=neg.shape, dtype='int32'),
    ])
