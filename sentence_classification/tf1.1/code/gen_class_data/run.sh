#/bin/sh
#===============================================================================
#
# Copyright (c) 2017 Trio.com, Inc. All Rights Reserved
#
#
# File: run.sh
# Author: ()
# Date: 2017/05/05 17:05:47
#
#===============================================================================
input_dir=/mnt/workspace/yezhenxu/data/termlevel/len40/train
out=/mnt/workspace/yezhenxu/data/termlevel/len40/test/
len=40

python 1.count_word.py $input_dir"/topic.question.txt.id"  $input_dir"/topic.frequence"
python 2.map_word_2_id.py $input_dir"/topic.question.txt.id" $input_dir"/topic.frequence" $input_dir"/topic.index" 
python 3.map_sentence_2_id.py $out"/topic.question.txt.id" $input_dir"/topic.index" $out"/topic.s2id" $len
python 4.shuffle_train_test_data.py $input_dir"/topic.s2id" $input_dir"/train.topic.data" $input_dir"/test.topic.data"



















# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
