#coding: utf8
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import config
import tools
import datautil
from bilstm_seq2seq import BilstmSeq2Seq
import subprocess
import argparse
import tools
import threading
import platform
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", help="the path of data", default="./data/")
parser.add_argument("-s", "--save_path", help="the path of the saved model", default="./models/")
parser.add_argument("-e", "--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-c", "--char_emb", help="the char embedding file", default="char.emb")
parser.add_argument("-w", "--word_emb", help="the word embedding file", default="term.emb.np")
parser.add_argument("-ed", "--emb_dim", help="the word embedding size", default=128)
parser.add_argument("-hd", "--hid_dim", help="the hidden size", default=128)
parser.add_argument("-g", "--gpu", help="the id of gpu, the default is 0", default=0, type=int)
parser.add_argument("-j", "--job", help="job name.", default="bilstm term-level crf", type=str)

args = parser.parse_args()

cf = config.TrainConfig(args)
if platform.system() == 'Darwin':
    cf.use_gpu = False

class TrainThread(threading.Thread):
    def __init__(self):
        train_iter = datautil.queue_iter()