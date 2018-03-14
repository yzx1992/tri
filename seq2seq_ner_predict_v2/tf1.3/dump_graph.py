#coding: utf8
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from bilstm_seq2seq import BilstmSeq2Seq
import config
import argparse
import datautil
import os
import sys

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
cf.use_gpu = False

def dump_graph_reduced(pb_path):
    output_nodes = "test_predicts"
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(cf.save_path)
        if ckpt:
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("Model load failed from {}".format(ckpt.model_checkpoint_path))
        # names = [n.name for n in sess.graph.as_graph_def().node]
        graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_nodes.split(","))
        with gfile.GFile(pb_path, "wb") as f:
            f.write(graph_def.SerializeToString())
        print("%d ops in the final graph." % len(graph_def.node))




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
        # tf.train.write_graph(sess.graph_def, model_dir, '%s.txt%s' % (pb_name, current_step))


def dump_graph_model(pb_name):
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    d = datautil.prepare_bilstm_data(cf.data_path, max_vocab_size=100000, reg=1)
    # tag_id_to_labels = datautil.gen_label_map(d['label_vocab_path'])
    # char_id_to_chars = datautil.gen_word_map(d['char_vocab_path'])
    term_emb = datautil.load_embedding_prebuilt(cf.data_path + '/' + cf.word_emb)
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        model= BilstmSeq2Seq(cf, cf.seq_len, d['char_vocab_size'], d['term_vocab_size'], d['feature_vocab_size'], cf.emb_dim, cf.hid_dim, d['label_vocab_size'], term_emb)
        print("Succeed in initializing the bidirectional LSTM model.")
        sys.stdout.flush()

        ckpt = tf.train.get_checkpoint_state(cf.save_path)
        if ckpt:
            print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            # model.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restore from {} finished".format(ckpt.model_checkpoint_path))
        else:
            raise ValueError("checkpoint not found.")

        output_node_names = "predicts"
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names.split(','))
        with gfile.GFile(os.path.join(cf.save_path, pb_name), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("{} ops in the final graph.".format(len(output_graph_def.node)))

if __name__ == '__main__':
    dump_graph_reduced("bilstm_seq2seq_attn.pb")
    #dump_graph_model("models/bilstm_seq2seq_attn.pb")
