#coding: utf8
from __future__ import print_function
import datautil

d = datautil.prepare_bilstm_data("data", 0, reg=0)
# for k,v in d.items():
#     print("{}".format(k))

term_vocab = d['term_vocab']

VOCAB = False

if VOCAB:
    id_term_vocab = dict((v, k) for k, v in term_vocab.iteritems())
    for k,v in id_term_vocab.items():
        print('{}\t{}'.format(k,v))

ITER = True

if ITER:
    test_iter = datautil.Itertool(d['test_ids_path'], batch_size=128, num_steps=100)
    cnt = 0
    for w, (term, fea, y) in test_iter:
        print('--------------------')
        print(w)
        print(len(w))
        print(term)
        print(term.shape)
        print(fea)
        print(fea.shape)
        print(y)
        print(y.shape)
        print()
        cnt += 1
        if cnt >= 1:
            break