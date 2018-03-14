from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
# from tensorflow.contrib.learn.python.learn.ops import seq2seq_ops
from tensorflow.python.ops import array_ops

class BilstmSeq2Seq(object):
    def __init__(self, config, seq_len, char_vocab_size, term_vocab_size, fea_dim, emb_size, hid_dim, num_cls,pretrained_emb=None):
        self.seq_len = seq_len
        self.char_vocab_size = char_vocab_size
        self.term_vocab_size = term_vocab_size
        self.num_cls = num_cls
        self.emb_size = emb_size
        self.hid_size = hid_dim
        self.num_cls = num_cls
        self.fea_dim = fea_dim
        self.decoder_hid_size = 2 * hid_dim
        # self.task_type="train"
        # self.task_type="train"
        # _task = tf.constant('train', dtype=tf.string)
        # print (_task)
        # self.chars = tf.placeholder(tf.int32, [None, self.seq_len], name='chars')
        # self.task_type = tf.placeholder_with_default(_task,[], name='task_type')
        # print (self.task_type)
        # assert (1==2)
        self.features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        self.terms = tf.placeholder(tf.int32, [None, self.seq_len], name='terms')
        self.targets = tf.placeholder(tf.int64, [None, self.seq_len], name='targets')
        self.targets_weight = tf.placeholder(tf.float32, shape=[None, self.seq_len], name='targets_weight')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.sequence_length = tf.reduce_sum(tf.sign(self.terms), axis=1)
        self.sequence_length = tf.cast(self.sequence_length, tf.int32)
        self.global_step = tf.Variable(0, trainable=False)

        if config.use_gpu:
            dev_info = '/gpu:{}'.format(config.gpu)
        else:
            dev_info = '/cpu:0'
        print('Device info:{}'.format(dev_info))

        with tf.device(dev_info):
            # self.char_emb = tf.Variable(tf.random_uniform([self.char_vocab_size, self.emb_size], -1.0, 1.0), name='char_emb')
            if pretrained_emb:
                self.term_emb = tf.Variable(pretrained_emb, dtype=tf.float32, name='term_emb')
            else:
                self.term_emb = tf.Variable(tf.random_uniform([self.term_vocab_size, self.emb_size], -1.0, 1.0), dtype=tf.float32, name = 'term_emb')

            self.terms_emb = tf.nn.embedding_lookup(self.term_emb, self.terms)
            self.terms_emb = tf.transpose(self.terms_emb, [1, 0, 2])
            self.terms_emb = tf.unstack(self.terms_emb, num=self.seq_len, name='terms_emb')

            self.fea_emb = tf.Variable(tf.random_uniform([self.fea_dim, self.emb_size], -1.0, 1.0), name='fea_emb')
            self.feas_emb = tf.nn.embedding_lookup(self.fea_emb, self.features)
            self.feas_emb = tf.transpose(self.feas_emb, [1, 0, 2])
            self.feas_emb = tf.unstack(self.feas_emb, num=self.seq_len, name='feas_emb')

            self.target_emb = tf.Variable(tf.random_uniform([self.num_cls, self.emb_size], -1.0, 1.0),name='target_emb')
            self.targets_emb = tf.nn.embedding_lookup(self.target_emb, self.targets)
            self.targets_emb = tf.transpose(self.targets_emb, [1, 0, 2])

            # concate words and features
            self.inputs_emb = [] # List of Tensor with shape: [batch_size, emb_size * 2 + fea_dim]
            for f_emb, t_emb in zip(self.feas_emb, self.terms_emb):
                 self.inputs_emb.append(tf.concat([t_emb, f_emb], 1))
            #for t_emb in (self.terms_emb ):
               # self.inputs_emb.append(tf.concat([t_emb], 1))

        with tf.device(dev_info):
            # bi-lstm layer
            self.decoder_cell = rnn.BasicLSTMCell(self.decoder_hid_size)
            self.encoder_cell_f = rnn.BasicLSTMCell(self.hid_size)
            self.encoder_cell_b = rnn.BasicLSTMCell(self.hid_size)
            self.encoder_cell_f = rnn.DropoutWrapper(self.encoder_cell_f, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            self.encoder_cell_b = rnn.DropoutWrapper(self.encoder_cell_b, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)

            self.encoder_outputs, self.encoder_state_fw, self.encoder_state_bw = rnn.static_bidirectional_rnn(
                self.encoder_cell_f,
                self.encoder_cell_b,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length = self.sequence_length
            )

            c_fw, h_fw = self.encoder_state_fw
            c_bw, h_bw = self.encoder_state_bw
            self.encoder_final_c = tf.concat([c_fw, c_bw], 1)
            self.encoder_final_h = tf.concat([h_fw, h_bw], 1)

            # self.encoder_state = tf.concat([tf.concat(self.encoder_state_fw, 1), tf.concat(self.encoder_state_bw, 1)], 1) # Tensor shape: [batch_size, hid_size * 2]
            self.atten_logits, _,self.test_atten_logits,_= self.attention_RNN(self.encoder_outputs, self.encoder_final_c,self.encoder_final_h, self.num_cls) # self.atten_logit Tensor shape: seq_len length list of Tensor [batch_size, num_cls]

            self.logits = tf.transpose(self.atten_logits, [1, 0, 2])  # Tensor shape: [batch_size, seq_len, num_cls]
            self.logits = tf.reshape(self.logits, [-1, self.num_cls], name='logits') # Tensor shape: [batch_size * seq_len, num_cls]
            mask_for_label = tf.reshape(tf.sign(self.targets), [-1])  # Tensor shape: [batch_size * seq_len, ]
            self.predict = tf.argmax(self.logits, axis=1) # Tensor shape: [batch_size * seq_len, ]
            self.predict_labels = self.predict * mask_for_label
            self.predicts = tf.reshape(self.predict_labels, [-1, self.seq_len], name='predicts') # Tensor shape: [batch_size, seq_len]

            self.test_logits = tf.transpose(self.test_atten_logits, [1, 0, 2])  # Tensor shape: [batch_size, seq_len, num_cls]
            self.test_logits = tf.reshape(self.test_logits, [-1, self.num_cls],
                                     name='test_logits')  # Tensor shape: [batch_size * seq_len, num_cls]
            self.test_predict = tf.argmax(self.test_logits, axis=1)  # Tensor shape: [batch_size * seq_len, ]
            self.test_predict_labels = self.test_predict * mask_for_label
            self.test_predicts = tf.reshape(self.test_predict_labels, [-1, self.seq_len],
                                       name='test_predicts')  # Tensor shape: [batch_size, seq_len]




            # self.predict_topk = tf.nn.top_k(self.logits, k=config.top_k) # Tensor shape : [batch_size * seq_len, topK]

            self.target_labels = tf.reshape(self.targets, [-1]) # Tensor shape: [batch_size * seq_len, ]
            self.labels = tf.one_hot(self.target_labels, depth=self.num_cls, on_value=1., off_value=0., name='one_hot_labels') # Tensor shape: [batch_size * seq_len, num_cls]

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='loss')
            mask = tf.reshape(tf.to_float(tf.sign(self.terms)), [-1]) # Tensor shape: [batch_size, seq_len]
            self.loss = self.loss * mask
            self.cost = tf.reduce_mean(self.loss)

            self.test_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.test_logits, labels=self.labels, name='test_loss')
            # mask = tf.reshape(tf.to_float(tf.sign(self.terms)), [-1])  # Tensor shape: [batch_size, seq_len]
            self.test_loss = self.test_loss * mask
            self.test_cost = tf.reduce_mean(self.test_loss)



            self.optimizer = tf.train.AdamOptimizer(config.lr).minimize(self.cost)

            # clip gradient
            # self.gradient = tf.gradients(self.loss, tf.trainable_variables())
            # self.clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(self.gradient, 5)
            # self.op = self.optimizer.apply_gradients(grads_and_vars = zip(self.clipped_gradients, tf.trainable_variables()), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def attention_RNN(self,
                      encoder_outputs,
                      encoder_final_c,
                      encoder_final_h,
                      vocab_size,
                      dtype=tf.float32,
                      num_heads = 1,
                      scope=None):
        attention_encoder_outputs, sequence_attention_weights = [], []
        test_attention_encoder_outputs, test_sequence_attention_weights = [], []

        with tf.variable_scope(scope or "attention_RNN"):
            output_size = encoder_outputs[0].get_shape()[1].value
            top_states = [tf.reshape(e, [-1, 1, output_size]) for e in encoder_outputs]  # List of Tensor with shape [batch_size, 1, out_size]
            attention_states = tf.concat(top_states, 1)  # Tensor shape [batch_size, seq_len, out_size]
            if not attention_states.get_shape()[1:2].is_fully_defined():
                raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                                 % attention_states.get_shape())

            batch_size = tf.shape(top_states[0])[0]  # Needed for reshaping.
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])  # Tensor shape [batch_size, seq_len, 1, out_size]
            hidden_features = []
            v = []
            attention_vec_size = attn_size  # Size of query vectors for attention.
            for i in xrange(num_heads):
                k = tf.get_variable("AttnW_%d" % i, [1, 1, attn_size, attention_vec_size])
                conv = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")  # Tensor shape [batch_size, seq_len, 1, out_size]
                hidden_features.append(conv)
                attn_vec = tf.get_variable("AttnV_0", [attn_size])
                v.append(attn_vec)

            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                weights = []
                ds = []  # Results of attention reads will be stored here.
                # if tf.nest.is_sequence(query):  # If the query is a tuple, flatten it.
                #     query_list = tf.nest.flatten(query)
                #     for q in query_list:  # Check that ndims == 2 if specified.
                #         ndims = q.get_shape().ndims
                #         if ndims:
                #             assert ndims == 2
                #     query = tf.concat(query_list, 1)
                for i in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % i):
                        y = rnn_cell._linear(query, attention_vec_size, True)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(v[i] * tf.tanh(hidden_features[i] + y), [2, 3])
                        a = tf.nn.softmax(s)
                        weights.append(a)
                        # Now calculate the attention-weighted vector d.
                        d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return weights, ds
            # batch_attn_size = tf.stack([batch_size, attn_size])
            # attns = [tf.zeros(batch_attn_size, dtype=dtype), ]
            # for a in attns:  # Ensure the second shape of attention vectors is set.
            #     a.set_shape([None, attn_size])
            #


            s_hidden_states = []
            two_states = []

            test_s_hidden_states = []
            test_two_states = []
            for i, inp in enumerate(encoder_outputs):  # list length: seq_len
                if i > 0:
                        tf.get_variable_scope().reuse_variables()
                if i == 0:
                    # with tf.variable_scope("Initial_Decoder_Attention"):
                    #     initial_state = rnn_cell._linear(encoder_state, output_size, True)
                    # weights, attens = attention(initial_state)
                    # c, h = array_ops.split(value=initial_state, num_or_size_splits=2, axis=1)
                    c, h = encoder_final_c, encoder_final_h
                    weights, attens = attention(h)
                    test_weights, test_attens =weights,attens
                    target_begin = tf.zeros_like(self.targets_emb[0])
                    s_hidden, two_state = self.decoder_cell(tf.concat([attens[0], inp, target_begin], 1), [c, h])
                    # s_hidden, two_state = self.decoder_cell(attens[0] + inp + rnn_cell._linear(target_begin,2*self.hid_size,True), [c,h])
                    s_hidden_states.append(s_hidden)
                    two_states.append(two_state)

                    test_s_hidden_states.append(s_hidden)
                    test_two_states.append(two_state)

                    output = tf.concat([attens[0], inp, target_begin, s_hidden_states[-1]], 1)
                    test_output=tf.concat([test_attens[0], inp, target_begin, test_s_hidden_states[-1]], 1)

                else:
                    predict_y_t = tf.nn.embedding_lookup(self.target_emb,tf.argmax(test_attention_encoder_outputs[-1], axis=1))

                    weights, attens = attention(s_hidden_states[-1])
                    test_weights, test_attens = attention(test_s_hidden_states[-1])

                    s_hidden, two_state = self.decoder_cell(tf.concat([attens[0], inp, self.targets_emb[i - 1]], 1),two_states[-1])
                    test_s_hidden, test_two_state = self.decoder_cell(tf.concat([test_attens[0], inp, predict_y_t], 1), test_two_states[-1])

                    s_hidden_states.append(s_hidden)
                    two_states.append(two_state)

                    test_s_hidden_states.append(test_s_hidden)
                    test_two_states.append(test_two_state)

                    output = tf.concat([attens[0], inp, self.targets_emb[i - 1], s_hidden_states[-1]], 1)
                    test_output = tf.concat([test_attens[0], inp, predict_y_t, test_s_hidden_states[-1]], 1)
                # output = tf.concat([attens[0], inp], 1)  # NOTE: here we temporarily assume num_head = 1

                def AttenOutputProject(_output,_vocab_size):
                    with tf.variable_scope("AttnRnnOutputProjection"):
                        _logit = rnn_cell._linear(_output, _vocab_size, True)  # Tensor shape: [batch_size, num_cls]
                    return _logit
                logit= AttenOutputProject(output,vocab_size)
                # test_logit=logit
                tf.get_variable_scope().reuse_variables()
                test_logit=AttenOutputProject(test_output,vocab_size)
                    # logit = rnn_cell._linear(output, vocab_size, True)  # Tensor shape: [batch_size, num_cls]
                    # test_logit = rnn_cell._linear(test_output, vocab_size, True)

                attention_encoder_outputs.append(logit)  # NOTE: here we temporarily assume num_head = 1
                sequence_attention_weights.append(weights[0])  # NOTE: here we temporarily assume num_head = 1

                test_attention_encoder_outputs.append(test_logit)  # NOTE: here we temporarily assume num_head = 1
                test_sequence_attention_weights.append(test_weights[0])





        return attention_encoder_outputs, sequence_attention_weights,test_attention_encoder_outputs,test_sequence_attention_weights
