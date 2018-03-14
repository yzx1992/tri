#coding: utf8
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import config
import datautil
from bilstm_seq2seq import BilstmSeq2Seq
import tools
import subprocess
import numpy as np
import argparse
import time
import sys
import os
import platform

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", help="the path of data", default="./eval/")
parser.add_argument("-s", "--save_path", help="the path of the saved model", default="./models/")
parser.add_argument("-i", "--input_path", help="input crf data", default="eval/valid.crf")
parser.add_argument("-e", "--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-c", "--char_emb", help="the char embedding file", default="char.emb")
parser.add_argument("-w", "--word_emb", help="the word embedding file", default="term.emb.np")
parser.add_argument("-ed", "--emb_dim", help="the word embedding size", default=128)
parser.add_argument("-hd", "--hid_dim", help="the hidden size", default=128)
parser.add_argument("-g", "--gpu", help="the id of gpu, the default is 0", default=0, type=int)
parser.add_argument("-j", "--job", help="job name.", default="bilstm term-level crf", type=str)

args = parser.parse_args()

cf = config.TrainConfig(args)
cf.use_gpu = False

def main():
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    d = datautil.prepare_bilstm_data(cf.data_path, max_vocab_size=100000, reg=1)

    valid_iter = datautil.Itertool(args.input_path, batch_size=cf.batch_size, seq_len=cf.seq_len)

    tag_id_to_labels = datautil.gen_label_map(d['label_vocab_path'])
    char_id_to_chars = datautil.gen_word_map(d['char_vocab_path'])
    term_emb = datautil.load_embedding_prebuilt(cf.data_path + '/' + cf.word_emb)

    with tf.Session() as sess:
        model= BilstmSeq2Seq(cf, cf.seq_len, d['char_vocab_size'], d['term_vocab_size'], d['feature_vocab_size'], cf.emb_dim, cf.hid_dim, d['label_vocab_size'], term_emb)
        print("Succeed in initializing the bidirectional LSTM model.")
        print("Begin training timestamp {}".format(time.time()))
        sys.stdout.flush()

        ckpt = tf.train.get_checkpoint_state(cf.save_path)
        if ckpt:
            print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            model.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("checkpoint not found.")

        vcost_valid = []
        ws = []
        predict_tags = []
        true_tags = []
        for w, c, fea, term, y in valid_iter:
            feed = dict(zip([model.chars, model.features, model.terms, model.targets, model.dropout_keep_prob],
                            [c, fea, term, y, 1.0]))
            out, predict, target = sess.run([model.cost, model.predict_labels, model.target_labels], feed)
            vcost_valid.append(out)
            ws.append(c)
            predict_tags.append(predict)
            true_tags.append(target)

        tools.conlleval(predict_tags, true_tags, ws, 'tmp/eval.crf', char_id_to_chars, tag_id_to_labels, cf.seq_len)
        d = tools.get_perf('tmp/eval.crf')
        print('Validation Cost: %0.6f, Precision: %0.6f, Recall: %0.6f, F1-score: %0.6f' % (
        np.sum(vcost_valid) / len(vcost_valid), d['p'], d['r'], d['f1']))
        print('Validation Details')
        print('\n'.join(d['detail']))
        sys.stdout.flush()

if __name__ == '__main__':
    main()


