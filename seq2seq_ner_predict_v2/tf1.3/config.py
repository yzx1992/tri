# encoding:utf-8

class TrainConfig(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.max_epoch = args.epoch
        self.word_emb = args.word_emb
        self.emb_dim = args.emb_dim
        self.use_gpu = True
        self.gpu_config = "/gpu:"+str(args.gpu)
        self.seq_len = 40 # Max length of a sentence.
        self.hid_dim = args.hid_dim
        self.batch_size = 128
        self.lr = 0.001
        self.save_freq = 50
        self.valid_freq = 500
        self.report_freq = 200
        # self.valid_freq = 10
