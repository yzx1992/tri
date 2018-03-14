#!/usr/bin/env python
# encoding: utf-8
# Author: tianrong@trio.ai (Tian Rong)

import sys
import math
import new_seg

def GetWordFreqDict(word_frequence_file):
    word_freq_dict = {}
    with open(word_frequence_file, 'r') as file_content:
        for line in file_content:
            word_freq = line.strip('\n').split('\t')
            if len(word_freq) != 2:
                continue
            new_str=word_freq[0]+"\t"+word_freq[1]
            items=new_str.split("\t")
            if len(items)!=2:
                continue
            if word_freq[0][0]=='\^':
                continue
            word_freq_dict[word_freq[0]] = int(word_freq[1])
    return word_freq_dict

def IndexWord(data_file_path_str, word_freq_dict):
    data_file_paths = data_file_path_str.strip().split(',')
    word_index_dict = {}
    index_cursor = 1
    for path in data_file_paths:
        with open(path, 'r') as file_content:
            cur = 0
            for line in file_content:
                cur += 1
                if cur % 1000 == 0:
                    new_seg.progressbar(cur, 6906531)
                label_text = line.strip('\n').split('\t')
                if len(label_text) != 2:
                    continue
                text = label_text[1]
                words_arr = text.split(' ')
                for word in words_arr:
                    if word == "" or word not in word_freq_dict:
                        continue
                    if word not in word_index_dict:
                        word_index_dict[word] = index_cursor
                        index_cursor += 1
    return word_index_dict

def main():
    if len(sys.argv) < 4:
        print "Invalid Command. Usage: python " + sys.argv[0] + \
                " InputDataPath WordFreqFile OutputWordIndexFile"
        return
    wordseg_cnn_train_data_path = sys.argv[1]
    #wordseg_cnn_train_data_path = '/home/tianrong/mutual_info/data/raw_data_seg'

    word_frequence_file = sys.argv[2]
    word_freq_dict = GetWordFreqDict(word_frequence_file)

    word_index_dict = IndexWord(wordseg_cnn_train_data_path, word_freq_dict)
    word_index_tuple = sorted(word_index_dict.items(),
        key=lambda word_index_dict : word_index_dict[1], reverse=False)

    output_fp = open(sys.argv[3], 'w')


    output_fp.write('<UNK>\t0\n')
    for e in word_index_tuple:
        output_fp.write(e[0] + '\t' + str(e[1]) + '\n')
    print "Total words: %d" % (len(word_index_tuple))
    print "word_index file length: %d" % (len(word_index_tuple) + 1)

if  __name__ == "__main__":
    main()
