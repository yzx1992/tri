# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
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
import platform
#import jieba
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_path", help="the path of data", default="./data/")
parser.add_argument("-s", "--save_path", help="the path of the saved model", default="./models/")
parser.add_argument("-b", "--batch_size", help="batch size", default=128)
parser.add_argument("-sl", "--seq_len", help="sequence length", default=100)
parser.add_argument("-m", "--mode", help="mode [train|predict]", default="predict", type=str)
parser.add_argument("-me", "--max_epoch", help="max epoch for training", default=200, type=int)
parser.add_argument("-lr", "--lr", help="learning rate for optimizer", default=0.001)
parser.add_argument("-rf", "--report_freq", help="frequency to report loss", default=300, type=int)
parser.add_argument("-vf", "--valid_freq", help="frequency to do validation", default=1000, type=int)
parser.add_argument("-sf", "--save_freq", help="frequency to do model dump", default=1000, type=int)
parser.add_argument("-k", "--top_k", help="predict output topK", default=1, type=int)
parser.add_argument("-e", "--epoch", help="the number of epoch", default=20, type=int)
parser.add_argument("-c", "--char_emb", help="the char embedding file", default="char.emb")
parser.add_argument("-w", "--word_emb", help="the word embedding file", default="term.emb.np")
parser.add_argument("-ed", "--emb_dim", help="the word embedding size", default=128)
parser.add_argument("-hd", "--hid_dim", help="the hidden size", default=128)
parser.add_argument("-g", "--gpu", help="the id of gpu, the default is 0", default=2, type=int)
parser.add_argument("-j", "--job", help="job name.", default="bilstm term-level crf", type=str)
parser.add_argument("-ug", "--use_gpu", help="if use gpu", default=True, type=bool)
parser.add_argument("-cl", "--curriculum_learing", help="if enable curriculurm learning.", default=0, type=int)
parser.add_argument("-pf", "--predict_file", help="input crf file in predict mode", default="./data/1025.test.crf", type=str)
parser.add_argument("-po", "--predict_output", help="predict output", default="./data/yzx_predict.crf", type=str)

args = parser.parse_args()

if platform.system() == 'Darwin':
    args.use_gpu = False

def SavePB(session, graph, model_dir, pb_name):
    var = {}
    for v in tf.trainable_variables():
        var[v.value().name] = session.run(v)
    g = tf.Graph()
    consts = {}
    with g.as_default(), tf.Session() as sess:
        for k in var.keys():
            consts[k] = tf.constant(var[k])
        tf.import_graph_def(graph.as_graph_def(), input_map={name:consts[name] for name in consts.keys()})
        tf.train.write_graph(sess.graph_def, model_dir, '%s.pb' % (pb_name), as_text=False)

def train():
    print("Preparing train and validation data.")
    d = datautil.prepare_bilstm_data(args.data_path, args.curriculum_learing, reg = 0)

    train_iter = datautil.Itertool(d['train_ids_path'], batch_size=args.batch_size, num_steps=args.seq_len, shuf=True)
    if args.curriculum_learing:
        train_cl_iter = datautil.Itertool(d['train_cl_ids_path'], batch_size=args.batch_size, num_steps=args.seq_len, shuf=True)

    valid_iter = datautil.Itertool(d['dev_ids_path'], batch_size=args.batch_size, num_steps=args.seq_len)
    test_iter = datautil.Itertool(d['test_ids_path'], batch_size=args.batch_size, num_steps=args.seq_len)

    term_vocab = d['term_vocab']
    label_vocab = d['label_vocab']
    id_term_vocab = dict((v,k) for k,v in term_vocab.iteritems())
    id_label_vocab = dict((v,k) for k,v in label_vocab.iteritems())
    term_emb = datautil.load_embedding_prebuilt(args.data_path + '/' + args.word_emb)

    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print("Building model.")
    g = tf.Graph()
    config= tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model= BilstmSeq2Seq(args, args.seq_len, None, d['term_vocab_size'], d['feature_vocab_size'], args.emb_dim, args.hid_dim, d['label_vocab_size'],term_emb)
        print("Succeed in initializing the bidirectional LSTM model.")
        print("Begin training timestamp {}".format(time.time()))
        sys.stdout.flush()

        checkpoint_path = os.path.join(args.save_path, "model.ckpt")
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            sess.run(tf.global_variables_initializer())
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        print("Training model.")
        total_step_num = 0
        best_eval_f1_score = 0.0
        for e in range(args.max_epoch):
            if args.curriculum_learing:
                outs = []
                start_time = time.time()
                for w, (term, fea, y) in train_cl_iter:
                    start_time = time.time()
                    feed = dict(zip([model.terms, model.features, model.targets, model.dropout_keep_prob],
                                    [term, fea, y, 0.5]))
                    cost, _ = sess.run([model.cost, model.optimizer], feed)
                    outs.append(cost)
                step_time = time.time() - start_time
                print('CL1 phrase Epoch:%d, Time: %.06f, Cost: %0.6f' % (e, step_time, 1.0 * np.average(outs)))

            outs = []
            step_time = 0
            for w, (term, fea, y) in train_iter:
                # model.task_type = "train"
                start_time = time.time()
                feed = dict(zip([model.terms, model.features, model.targets, model.dropout_keep_prob], [term, fea, y, 0.5]))
                cost, _ = sess.run([model.cost, model.optimizer], feed)
                outs.append(cost)

                total_step_num += 1
                step_time += (time.time() - start_time) / args.save_freq

                if total_step_num % args.report_freq == 0:
                    e_sub = 1.0 * total_step_num * args.batch_size / train_iter.total_size
                    print('CL2 phrase Epoch:%f, Step:%d, Step Time: %.06f, Cost: %0.6f' % (e_sub, total_step_num, step_time, 1.0 * np.sum(outs) / args.report_freq))
                    sys.stdout.flush()
                    outs = []
                    step_time = 0.

                if total_step_num % args.valid_freq == 0:
                    # model.task_type = "test"
                    vcost_valid = []
                    ws = []
                    predict_tags = []
                    true_tags = []
                    for w, (term, fea, y) in valid_iter:
                        feed = dict(zip([model.terms, model.features, model.targets, model.dropout_keep_prob], [term, fea, y, 1.0]))
                        out, predict, target = sess.run([model.test_cost, model.test_predict_labels, model.target_labels], feed)
                        vcost_valid.append(out)
                        ws.append(term)
                        predict_tags.append(predict)
                        true_tags.append(target)

                    tools.conlleval(predict_tags, true_tags, ws, 'tmp/eval.crf', id_term_vocab, id_label_vocab, args.seq_len)
                    d = tools.get_perf('tmp/eval.crf')
                    print('Validation Cost: %0.6f, Precision: %0.6f, Recall: %0.6f, F1-score: %0.6f' % (np.sum(vcost_valid) / len(vcost_valid), d['p'], d['r'], d['f1']))
                    print('Validation Details')
                    print('\n'.join(d['detail']))
                    sys.stdout.flush()
                    if d['f1'] >= best_eval_f1_score:
                        best_eval_f1_score = d['f1']
                        model.saver.save(sess, checkpoint_path, global_step=total_step_num)
                        subprocess.call(['cp', 'tmp/eval.crf',
                                         'tmp/eval.crf.%d_best_f1_%.2f' % (total_step_num, best_eval_f1_score)])
                if total_step_num % args.valid_freq == 0:
                    model.task_type = "test"
                    vcost_test = []
                    ws = []
                    predict_tags = []
                    true_tags = []

                    for w, (term, fea, y) in test_iter:

                        feed = dict(zip([model.terms, model.features, model.targets, model.dropout_keep_prob], [term, fea, y, 1.0]))
                        out, predict, target = sess.run([model.test_cost, model.test_predict_labels, model.target_labels], feed)
                        vcost_test.append(out)
                        ws.append(term)
                        predict_tags.append(predict)
                        true_tags.append(target)

                    tools.conlleval(predict_tags, true_tags, ws, 'tmp/test.crf', id_term_vocab, id_label_vocab, args.seq_len)
                    d = tools.get_perf('tmp/test.crf')
                    print('Test Cost: %0.6f, Precision: %0.6f, Recall: %0.6f, F1-score: %0.6f' % (np.sum(vcost_test) / len(vcost_test), d['p'], d['r'], d['f1']))
                    print('Test Details')
                    print('\n'.join(d['detail']))
                    sys.stdout.flush()

        f = open(args.save_path + '/_SUCCESS', 'w')
        f.writelines('_SUCCESS')
        f.close()

def predict():
 #   fw=open('yzx1026.crf','w+')

#    seg_list = jieba.cut("请问鼓浪屿在哪？", cut_all=False)
  #  str=" ".join(seg_list)
   # fw.write('<S>\tO\tO\n')
    #for i in str.strip().split():
     #   i=i.encode('utf-8')
      #  fw.write(i+'\tO\tO\t\n')
    #fw.close()
    #predict_file="/data/yzx1026.crf"
    id_path, term_vocab, fea_vocab, label_vocab = datautil.prepare_predict_data(args.data_path, args.predict_file)
    data_iter = datautil.Itertool(id_path, batch_size=args.batch_size, num_steps=args.seq_len)

    id_term_vocab = dict((v,k) for k,v in term_vocab.iteritems())
    id_label_vocab = dict((v,k) for k,v in label_vocab.iteritems())

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = BilstmSeq2Seq(args, args.seq_len, None, len(term_vocab), len(fea_vocab), args.emb_dim,
                              args.hid_dim, len(label_vocab), pretrained_emb=None)
        ckpt = tf.train.get_checkpoint_state(args.save_path)
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        fout = open(args.predict_output, 'w')

        vcost_test = []
        ws = []
        predict_tags = []
        true_tags = []

        for w, (term, fea, y) in data_iter:
            feed = dict(zip([model.terms, model.features, model.targets, model.dropout_keep_prob], [term, fea, y, 1.0]))
            out, predict, target = sess.run([model.test_cost, model.test_predict_labels, model.target_labels], feed)
            vcost_test.append(out)
            ws.append(term)
            predict_tags.append(predict)
            true_tags.append(target)
            tools.predict_dump(fout, id_term_vocab, id_label_vocab, w, predict, seq_len=args.seq_len, topK=args.top_k)

        tools.conlleval(predict_tags, true_tags, ws, 'data/predict_.crf', id_term_vocab, id_label_vocab, args.seq_len)
        d = tools.get_perf('data/predict_.crf')
        print('Predict Cost: %0.6f, Precision: %0.6f, Recall: %0.6f, F1-score: %0.6f' % (
        np.sum(vcost_test) / len(vcost_test), d['p'], d['r'], d['f1']))
        print('Predict Details')
        print('\n'.join(d['detail']))
        sys.stdout.flush()
        # for w, t in data_iter:
        #
        #
        #     term, fea = t[0], t[1]
        #     feed = dict(zip([model.terms, model.features, model.dropout_keep_prob], [term, fea, 1.0]))
        #     predict = sess.run([model.predict], feed)
        #     # predict, predict_topk = sess.run([model.predict, model.predict_topk], feed)


        fout.close()

def main():
    if args.mode == "train":
        train()
    elif args.mode == "predict":
        predict()
    else:
        raise ValueError("Unrecognized mode {}, valid includes [train|predict]".format(args.mode))

if __name__ == "__main__":
    main()
