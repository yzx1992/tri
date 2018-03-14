#!/usr/bin/env python
# encoding: utf-8
# Author: tianrong@trio.ai (Tian Rong)

import sys
import math
import new_seg

def GetWordIndexDict(word_index_file):
    word_index_dict = {}
    with open(word_index_file, 'r') as file_content:
        for line in file_content:
            word_index = line.strip('\n').split('\t')
            if len(word_index) != 2:
                continue
            word_index_dict[word_index[0]] = int(word_index[1])
    return word_index_dict

def MapSentence2WordID(data_file_path_str, word_index_dict, output_fp,
        max_text_length):
    data_file_paths = data_file_path_str.strip().split(',')
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
                split_field_str = ""
                write_line = ""
                text = label_text[1]
                for j in xrange(1):
                    words_arr = text.split(' ')
                    split_word_str = ""

                    # append word in post, reply+ or reply- to write_words.
                    # Right alignment.
                    write_words_curr = ""
                    write_words_curr_arr = []
                    write_words = ""
                    #for i in range(0, min(len(words_arr), max_text_length)):

                    i = 0
                    while(i < len(words_arr) and len(write_words_curr_arr) < max_text_length):
                        if words_arr[-(i+1)] == "":
                            i += 1
                            continue
                        write_words = '0'
                        if words_arr[-(i+1)] in word_index_dict:
                            write_words = str(word_index_dict[words_arr[-(i+1)]])
                        write_words_curr = write_words + split_word_str + write_words_curr
                        write_words_curr_arr.append(write_words)
                        split_word_str = " "
                        i += 1

                    # Align insufficient dimensions with 0.
                    #print len(words_arr),write_words_curr
                    while(len(write_words_curr_arr) < max_text_length):
                        write_words_curr += " 0"
                        write_words_curr_arr.append("0")

                    # append post, reply+ or reply- to write_line.
                    write_line += (split_field_str + write_words_curr)
                    split_field_str = "\t"

                output_fp.write(label_text[0]+'\t'+write_line + '\n')
    return

def main():
    if len(sys.argv) < 4:
        print "Invalid Command. Usage: python " + sys.argv[0] \
                + " InputDataPath WordIndexFile OutputWordIDTrainData [MaxTextLength]"
        return
    wordseg_cnn_train_data_path = sys.argv[1]
    #wordseg_cnn_train_data_path = '/home/tianrong/mutual_info/data/raw_data_seg'

    word_index_file = sys.argv[2]
    word_index_dict = GetWordIndexDict(word_index_file)
    output_fp = open(sys.argv[3], 'w')

    # max_text_length
    max_text_length = 25
    if len(sys.argv) >= 5:
        max_text_length = int(sys.argv[4])

    word_index_dict = MapSentence2WordID(wordseg_cnn_train_data_path,
            word_index_dict, output_fp, max_text_length)

    return

if  __name__ == "__main__":
    main()
