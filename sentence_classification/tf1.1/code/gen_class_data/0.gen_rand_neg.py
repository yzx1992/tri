#!/usr/bin/env python
# encoding: utf-8
# Author: tianrong@trio.ai (Tian Rong)

import codecs
import numpy as np
import sys

infilename = sys.argv[1]
outfilename = sys.argv[2]
pos_pairs = []

with codecs.open(infilename, 'r', 'utf-8') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        if len(line) != 2:
            continue
        pos_pairs.append(line)
num_pairs = len(pos_pairs)
print num_pairs

post, replies = zip(*pos_pairs)
shuffle_indices = np.random.permutation(np.arange(num_pairs))
rand_replies = [replies[idx] for idx in shuffle_indices]
neg_pairs = zip(post, rand_replies)
print len(neg_pairs)
with codecs.open(outfilename, 'w', 'utf-8') as f:
    for p, n in zip(pos_pairs,neg_pairs):
        f.write(p[0] + '\t')
        f.write(p[1] + '\t')
        f.write(n[1] + '\n')
