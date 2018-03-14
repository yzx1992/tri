#!/usr/bin/env python
# encoding: utf-8
# Author: tianrong@trio.ai (Tian Rong)

import sys
import math
import new_seg
def GetWordFreqDict(data_file_path_str):
    data_file_paths = data_file_path_str.strip().split(',')
    files_row_num = 0
    word_freq_dict = {}
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
                    if word == "":
                        continue
                    if word not in word_freq_dict:
                        word_freq_dict[word] = 1
                    else:
                        word_freq_dict[word] += 1
    return word_freq_dict


def main():
    if len(sys.argv) < 3:
        print "Invalid Command. Usage: python " + sys.argv[0] \
                + " InputDataPath OutputWordFrequencePath [WordFrequenceThreshold]"
        return
    wordseg_cnn_train_data_path = sys.argv[1]
    #wordseg_cnn_train_data_path = '/home/tianrong/mutual_info/data/raw_data_seg'

    word_freq_dict = GetWordFreqDict(wordseg_cnn_train_data_path)

    output_word_frequence_path = sys.argv[2]
    output_fp = open(output_word_frequence_path, 'w')

    word_freq_threshold = 0.0
    if len(sys.argv) >= 4:
        word_freq_threshold = float(sys.argv[3])

    filtered_num = 0
    for word in word_freq_dict:
        if word_freq_dict[word] < word_freq_threshold:
            filtered_num += 1
        else:
            output_fp.write(word + '\t' + str(word_freq_dict[word]) + '\n')
    print "Total words: %d" % (len(word_freq_dict))
    print "Word freq filtered words: %d" % (filtered_num)

if  __name__ == "__main__":
    main()
