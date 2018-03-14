# Copyright 2016 
# data pre-process
#   build_vocabulary
#   
#
#
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter
import numpy as np
import subprocess
import platform

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
__PAD = "_PAD"
PAD_ID = 0
UNK_ID = 1
LABEL_O_ID = 1
GO_ID = 2
EOS_ID = 3

_START_VOCAB = [_PAD, _UNK]
_START_VOCAB_REG = [_PAD, _UNK, '_ALPHABET', '_DIGIT']
_START_VOCAB_LABEL = [_PAD, _UNK]

def gen_label_map(label_vocab_path):
    if os.path.exists(label_vocab_path):
        label_vocab = []
        with open(label_vocab_path, mode="rb") as f:
            label_vocab.extend(f.readlines())
        label_vocab = [line.strip('\n') for line in label_vocab]
        label_map = dict(zip(range(len(label_vocab)), label_vocab))
        return label_map
    else:
        raise ValueError("Label file %s not found.", label_vocab_path)


def gen_word_map(vocab_path):
    if os.path.exists(vocab_path):
        word_vocab = []
        with open(vocab_path, mode="rb") as f:
            word_vocab.extend(f.readlines())
        word_vocab = [line.strip('\n') for line in word_vocab]
        word_map = dict(zip(range(len(word_vocab)), word_vocab))
        return word_map
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def load_vocab(vocab_path):
    rev_vocab = []
    with open(vocab_path, mode="rb") as f:
        for w in f:
            w = w.rstrip('\n')
            rev_vocab.append(w)
    vocab = dict(zip(rev_vocab, range(len(rev_vocab))))
    return vocab, rev_vocab


# Not used here.
def sentence_to_ids(sentence, vocab, eos_flag=False):
    """Convert a string to list of integers representing token-ids.
    Returns:
        a list of integers, the token-ids for the sentence.
    """
    words = sentence.strip().split()  # default space between words
    if eos_flag:
        words.append(_EOS)
    return [vocab.get(w, UNK_ID) for w in words]


def map_file_to_ids(filename, filename_ids, reg = 1, *vocabs):
    """
        Input 3 column file -> ids.
        The 3 column is char, feature(seg_pos), label.
        :param filename:
        :param filename_ids:
        :param word_vocab:
        :param feature_vocab:
        :param label_vocab:
        :param reg:
        :return:
        """
    line_no = 0
    nVocab = len(vocabs)
    print(nVocab)
    with open(filename, 'r') as f, open(filename_ids, 'w') as fw:
        l = []
        for line in f:
            line_no += 1
            line = line.rstrip('\n')
            if line:
                word = line.split(' ')
                if len(word) != nVocab:
                    raise ValueError("The line %d in file %s does not contain %d elements!" %
                            (line_no, filename, nVocab))
                if reg == 1:
                    if word[0].isalnum():
                        word[0] = "_ALPHABET"
                l.append(word)
            else:
                if len(l) > 0:
                    l = map(list, zip(*l))
                    out_l = []
                    for i, toks in enumerate(l):
                        ids = ' '.join([str(vocabs[i].get(tok, UNK_ID)) for tok in toks])
                        out_l.append(ids)
                    fw.write('\t'.join(out_l) + '\n')
                l = []
        if len(l) > 0:
            l_id = [str(vocabs[i].get(w, UNK_ID)) for w in word[i] for i in range(0, nVocab)]
            out_l = []
            for ids in l_id:
                ids = ' '.join(ids)
                out_l.append(ids)
            fw.write('\t'.join(out_l) + '\n')
    return

def prepare_bilstm_data(data_path, cl, reg=1, rebuild=True):
    """
    Prepare the train and validation data for training.
    Create the vocabularies and label dictionary.
    Convert the train and validation data to ids data.
    """
    # Get data to the specified directory.
    train_path = os.path.join(data_path, 'train.crf')
    if cl:
        train_cl_path = os.path.join(data_path, 'train.cl.crf')
    dev_path = os.path.join(data_path, 'valid.crf')  # get_wmt_enfr_dev_set(data_dir)
    test_path = os.path.join(data_path, 'test.crf')
    # train_path = train_path
    # dev_path = valid_path

    label_vocab_path = os.path.join(data_path, "output.vocab")
    term_vocab_path = os.path.join(data_path, "input1.vocab")
    feature_vocab_path = os.path.join(data_path, "input2.vocab")

    term_vocab, term_rev_vocab = load_vocab(term_vocab_path)
    feature_vocab, feature_rev_vocab = load_vocab(feature_vocab_path)
    label_vocab, label_rev_vocab = load_vocab(label_vocab_path)

    term_vocab_size = len(term_rev_vocab)
    label_size = len(label_vocab)
    feature_size = len(feature_vocab)

    # Create token ids for the training data.
    train_ids_path = train_path + ".ids.txt"
    if rebuild or not os.path.exists(train_ids_path):
        map_file_to_ids(train_path, train_ids_path, reg, term_vocab, feature_vocab, label_vocab)

    if cl:
        train_cl_ids_path = train_path + "cl.ids.txt"
        if rebuild or not os.path.exists(train_cl_ids_path):
            map_file_to_ids(train_cl_path, train_cl_ids_path, reg, term_vocab, feature_vocab, label_vocab)

    # Create token ids for the development data.
    dev_ids_path = dev_path + ".ids.txt"
    if rebuild or not os.path.exists(dev_ids_path):
        map_file_to_ids(dev_path, dev_ids_path, reg, term_vocab, feature_vocab, label_vocab)

    # Create token ids for the test data.
    test_ids_path = test_path + ".ids.txt"
    if rebuild or not os.path.exists(test_ids_path):
        map_file_to_ids(test_path, test_ids_path, reg, term_vocab, feature_vocab, label_vocab)

    out_d = {
        'train_ids_path': train_ids_path,
        'dev_ids_path': dev_ids_path,
        'test_ids_path': test_ids_path,
        'feature_vocab_path': feature_vocab_path,
        'term_vocab_path': term_vocab_path,
        'label_vocab_path': label_vocab_path,
        'feature_vocab_size': feature_size,
        'term_vocab_size': term_vocab_size,
        'label_vocab_size': label_size,
        'term_vocab' : term_vocab,
        'label_vocab' : label_vocab
    }
    if cl:
        out_d['train_cl_ids_path'] = train_cl_ids_path
    return out_d

def prepare_predict_data(data_path, predict_crf, reg=1):

    term_vocab_path = os.path.join(data_path, "input1.vocab")
    feature_vocab_path = os.path.join(data_path, "input2.vocab")
    label_vocab_path = os.path.join(data_path, "output.vocab")

    term_vocab, term_rev_vocab = load_vocab(term_vocab_path)
    feature_vocab, feature_rev_vocab = load_vocab(feature_vocab_path)
    label_vocab, label_rev_vocab = load_vocab(label_vocab_path)

    predict_id_path = predict_crf + ".ids.txt"
    map_file_to_ids(predict_crf, predict_id_path, reg, term_vocab, feature_vocab, label_vocab)

    return predict_id_path, term_vocab, feature_vocab, label_vocab

def bilstm_pad(samples, seq_len=100, dtype='int32'):
    '''
    padding 0
    '''
    n_sample = len(samples)
    n_col = len(samples[0])
    # l = map(list, zip(*samples))
    out_l = [np.zeros((n_sample, seq_len)).astype(dtype) for _ in range(0, n_col)]
    w = []
    for row, l_ids in enumerate(samples):
        w.append(l_ids[0])
        sen_len = len(l_ids[0])
        for col in range(0, n_col):
            out_l[col][row, :sen_len] = l_ids[col]
    #print(out_l[0])
    #print(out_l[1])
    #print(out_l[2])
    return w, out_l


class Itertool(object):
    '''
    data_path: ids, each line is of type "s1 \t s2", s1=sen, s2=labels.
    '''

    def __init__(self, data_path, batch_size=128, num_steps=100, shuf=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_len = num_steps
        self.shuf = shuf
        proc = subprocess.Popen(['wc', '-l', self.data_path], stdout=subprocess.PIPE)
        s = proc.stdout.read()
        toks = s.strip().split(' ')
        self.total_size = int(toks[0])
        # data back up
        if self.shuf:
            subprocess.Popen(['cp', data_path, data_path + '.bak'])

    def shuf(self):
        shuf_data_path = self.data_path + '.shuf'
        with open(shuf_data_path, 'w') as fout:
            if platform.system() == 'Darwin':
                subprocess.Popen(['gshuf', self.data_path], stdout=fout)
            else:
                subprocess.Popen(['shuf', self.data_path], stdout=fout)
        subprocess.Popen(['mv', shuf_data_path, self.data_path])

    def __iter__(self):
        # iterator
        with open(self.data_path, 'r') as f:
            sample_num = 0
            samples = []
            # words = []
            for line in f:
                sample_num += 1
                toks = line.strip().split('\t')
                toks = [s.split(' ')[:self.max_len] for s in toks]
                samples.append(toks)
                if sample_num % self.batch_size == 0:
                    w, feed_input = bilstm_pad(samples, seq_len=self.max_len)
                    yield w, feed_input
                    samples = []
            if len(samples) > 0:
                # n_batch_pad = self.batch_size - len(samples)
                # last_sample = samples[-1]
                # for _ in range(n_batch_pad):
                #     samples.append(last_sample)
                w, feed_input = bilstm_pad(samples, seq_len=self.max_len)
                yield w, feed_input
                samples = []

class HotIter(object):
    def __init__(self, data_path, mq_host = 'localhost', queue_name = 'hotlearn_nlu', batch_size=128, seq_len=40):
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(mq_host))
        self.channel = self.conn.channel()
        self.queue_name = queue_name
        # self.channel.queue_declare(queue='hotlearn_nlu', durable=True)
        self.channel.queue_declare(queue=self.queue_name)
        self.batch_size = batch_size
        self.max_len = seq_len

        # using another file to store its copy.
        # self.data_path = data_path + '.hotiter'
        # subprocess.Popen(['cp', data_path, self.data_path])

        self.replay_size = 1000
        self.l = []

    def __iter__(self):
        for method_frame, properties, body in self.channel.consume(self.queue_name):
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            # ending signal
            if body == '_END':
                start, n = 0, 0
                for _ in self.l:
                    n += 1
                    if n % self.batch_size == 0:
                        w, x, fea, term, y = bilstm_pad(self.l[start:n], seq_len=self.max_len)
                        yield w,x,fea,term,y
                        self.start = n
                if start < len(self.l):
                    w, x, fea, term, y = bilstm_pad(self.l[start:], seq_len=self.max_len)
                    yield w,x,fea,term,y
                self.l = []
            else:
                word = body.split('\t')
                sen = word[0].split(' ')
                features = word[1].split(' ')
                terms = word[2].split(' ')
                labels = word[3].split(' ')
                if len(sen) > self.max_len:
                    sen = sen[:self.max_len]
                    features = features[:self.max_len]
                    terms = terms[:self.max_len]
                    labels = labels[:self.max_len]
                self.l.append([sen,features,terms,labels])

def load_embedding(filename, extra_tokens=_START_VOCAB):
    embeddings = []
    for _ in extra_tokens:
        rnd_arr = np.random.random((1, 128)).tolist()
        embeddings.append(rnd_arr[0])
    with open(filename) as fin:
        for line in fin:
            line = line.rstrip('\n')
            if not line:
                raise ValueError("Embedding format error [{}]".format(line))
            embedding = line.split()
            embedding = [float(it) for it in embedding]
            embeddings.append(embedding)
    print("Load embedding of size: {} x {}".format(
        len(embeddings), len(embeddings[0])))
    return embeddings


def load_embedding_prebuilt(filename):
    m = np.load(filename)
    print("Load embedding of size: {} x {}".format(m.shape[0], m.shape[1]))
    return m.tolist()
