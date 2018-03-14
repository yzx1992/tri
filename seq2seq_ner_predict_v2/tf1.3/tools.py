from __future__ import print_function
import os
import subprocess
import numpy as np


# metrics function using conlleval.py
def conlleval(p, g, w, filename, word_id_to_words, tag_id_to_labels, seq_len):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''

    def extract_entities(tags, tag_id_to_labels):
        l = []
        for tag_id in tags:
            tag = tag_id_to_labels.get(tag_id, '_UNK')
            # if tag == '__PAD': # Do not truncate label pad, since predict label may be '__PAD'
            #     break
            # trans_tag = tag_to_category.get(tag, tag)
            l.append(tag)
        return l

    def extract_words(ids, word_id_to_words):
        w = []
        for id in ids:
            word = word_id_to_words[id]
            if word == '_PAD':
                break
            w.append(word)
        return w

    def extract_sen_label(l, seq_len):
        out_l = []
        for bl in l:  # Shape: [1, batch_size * seq_len]
            for i in xrange(0, len(bl), seq_len):
                out_l.append(bl[i: i + seq_len])
        return out_l

    def extract_sen_words(l):
        out_l = []
        for sl in l:  # list
            for arr in sl:  # nparray
                wl = arr.tolist()
                out_l.append(wl)
        return out_l

    out = ''
    lw = extract_sen_words(w)
    ll = extract_sen_label(g, seq_len)
    lp = extract_sen_label(p, seq_len)
    # print('------')
    # print(len(lw))
    # print(len(ll))
    # print(len(lp))
    # print(len(lw[0]))
    # print(len(ll[0]))
    # print(len(lp[0]))
    # print('------')
    for sw, sl, sp in zip(lw, ll, lp):
        sw = extract_words(sw, word_id_to_words)
        sl = extract_entities(sl, tag_id_to_labels)
        sp = extract_entities(sp, tag_id_to_labels)
        # print('######')
        # print(sw)
        # print(sl)
        # print(sp)
        # print(len(sw))
        # print(len(sl))
        # print(len(sp))
        # print('######')
        # out += 'BOS O O\n'
        # out += '<S> O O\n'
        for ww, wl, wp in zip(sw, sl, sp):  # List of seq_len
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += '\n'
        # out += 'EOS O O\n\n'
        # out += '<E> O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()
    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    out_l = []
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.py'
    proc = subprocess.Popen(["python",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        out_l.append(line)
        if 'accuracy' in line:
            out = line.split()
            # break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])
    return {'detail': out_l, 'p': precision, 'r': recall, 'f1': f1score}


def f1score(predict_tags, true_tags, tag_id_to_labels):
    """Compute the f1-score for NER.
    Args:
    predict_tags: a list of sequence predict tags, such as [[0, 1, 2], [0, 1, 2, 3]]
    true_tags: a list of sequence true tags, such as [[0, 1, 1], [0, 1, 1, 3]]
    tag_id_to_labels: a dict which map tag id to human-readable label.
    Returns:
    precion, recall, f1-score
    """

    def extract_entities(tags, tag_id_to_labels):
        tag_to_category = {
            "__PAD": "O",
            "O": "O",
            "SL": "Loc",
            "ML": "Loc",
            "EL": "Loc",
            "SO": "Org",
            "MO": "Org",
            "EO": "Org",
            "L": "L"
        }
        entities = set()
        prev_category = "O"
        entity = []
        for idx, tag_id in enumerate(tags):
            tag = tag_id_to_labels[tag_id]
            if tag in ["SL", "SO", "L", "O", "__PAD"]:
                if prev_category != "O":
                    entities.add((prev_category, tuple(entity)))
                entity = []
            entity.append(idx)
            prev_category = tag_to_category[tag]
        if prev_category != "O":
            entities.add((prev_category, tuple(entity)))
        return entities

    tp = 0
    tpfp = 0
    tpfn = 0
    for predict_tags_, true_tags_ in zip(predict_tags, true_tags):
        true_entities = extract_entities(true_tags_, tag_id_to_labels)
        predict_entities = extract_entities(predict_tags_, tag_id_to_labels)
        for entity in predict_entities:
            if entity in true_entities:
                tp += 1
        tpfp += len(predict_entities)
        tpfn += len(true_entities)

    precision = tp * 1.0 / tpfp
    recall = tp * 1.0 / tpfn
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
    return precision, recall, f1


def test_test():
    tag_id_to_labels = dict(zip(range(8), ["SL", "ML", "EL", "SO", "MO", "EO", "O", "L"]))
    predict_tags = [
        [0, 1, 2, 6, 7],
        [6, 3, 4, 5, 7, 7, 0, 1, 2]
    ]
    true_tags = [
        [0, 1, 2, 6, 6],
        [7, 3, 4, 5, 7, 6, 0, 1, 2]
    ]
    p, r, f1 = f1score(predict_tags, true_tags, tag_id_to_labels)
    true_p, true_r, true_f1 = 0.66667, 0.8, 0.72727
    assert (abs(p - true_p) < 0.0001)
    assert (abs(r - true_r) < 0.0001)
    assert (abs(f1 - true_f1) < 0.0001)
    print("Test passed..")


def test_conll():
    tag_id_to_labels = dict(zip(range(9), ["SL", "ML", "EL", "SO", "MO", "EO", "O", "L", "__PAD"]))
    word_id_to_words = dict(zip(range(10), ["_PAD", "<S>", "<E>", "b", "c", "d", "e", "f", "g", "h"]))
    words = [
        [1, 4, 3, 4, 6, 7, 2, 0, 0, 0, 0],
        [1, 6, 3, 4, 5, 7, 7, 6, 5, 4, 2]
    ]
    word_ids = [np.array([words[0], ]), np.array([words[1], ])]
    predict_tags = [
        [6, 0, 1, 2, 6, 7, 6, 8, 8, 8, 8],
        [6, 6, 3, 4, 5, 7, 7, 0, 1, 2, 6]
    ]
    true_tags = [
        [6, 0, 1, 2, 6, 6, 6, 8, 8, 8, 8],
        [6, 7, 3, 4, 5, 7, 6, 0, 1, 2, 6]
    ]
    conlleval(predict_tags, true_tags, word_ids, "tmp/eval.crf", word_id_to_words, tag_id_to_labels, 11)
    d = get_perf("tmp/eval.crf")
    print('Validation Precision: %0.6f, Recall: %0.6f, F1-score: %0.6f' % (d['p'], d['r'], d['f1']))
    print('Validation Details')
    print('\n'.join(d['detail']))

"""
@Param[IN] id_to_words: id -> word
@Param[IN] id_to_labels: id -> label
@Param[IN] l_words: batch_size length list of input ids
@Param[IN] np_predict: np matrix with shape [batch_size * max_seq_len, ] if topK = 1, [batch_size * max_seq_len, k] if topK > 1
@Param[IN] topK: if true, output topk format
                 if false, output top1 format.
"""
def predict_dump(fd, id_to_words, id_to_labels, l_words, np_predict, seq_len, topK=1):
    assert(topK >= 1)
    l_len = [len(i) for i in l_words]
    if topK == 1:
        id_predicts = np.reshape(np_predict, (-1, seq_len)).tolist() # shape: (batch_size, seq_len)
    else:
        id_predicts = np.reshape(np_predict, (-1, seq_len, topK)).tolist()
    if topK == 1:
        for id_word_sen, id_predict_sen in zip(l_words, id_predicts):
            for id_word, id_predict in zip(id_word_sen, id_predict_sen):
                word = id_to_words[int(id_word)]
                label = id_to_labels[int(id_predict)]
                fd.write('{} {}\n'.format(word, label))
            fd.write('\n')
    else:
        for id_word_sen, id_predict_mat in zip(l_words, id_predicts):
            for id_word, id_predict_vec in zip(id_word_sen, id_predict_mat):
                word = id_to_words[int(id_word)]
                labels = [id_to_labels[int(id)] for id in id_predict_vec]
                fd.write('{} {}\n'.format(word, ' '.join(labels)))
            fd.write('\n')

DEBUG = False
if DEBUG:
    test_conll()
    test_test()