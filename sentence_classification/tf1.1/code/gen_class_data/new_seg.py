from __future__ import division
import re
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write('[%-50s] %s' % ( '>' * int(math.floor(cur*50 / total)), percent))
    sys.stdout.flush()

def is_ch(c_str):
    zhPattern = re.compile(ur'^[\u4e00-\u9fa5]+$')
    match = zhPattern.match(c_str.decode('utf-8', 'ignore'))
    if match:
        return True
    else:
        return False
def gen_sentence(sen):
    sen = sen.decode('utf-8')
    sen = sen.strip().split(' ')
    w_list = []
    for word in sen:
        if is_ch(word):
            for cha in word:
                w_list.append(cha)
        else:
            #print word
            w_list.append(word)
    return ' '.join(str(c) for c in w_list)

if __name__ == '__main__':

    raw_data = sys.argv[1]
    out_data = open(sys.argv[2], 'w')
    with open(raw_data, 'r') as raw_file:
        current_step = 0
        for lines in raw_file:
            current_step += 1
            if current_step % 1000 == 0:

                progressbar(current_step, 37000000)
            lines = lines.strip().split('\t')
            if len(lines) != 3:
                continue
            query = gen_sentence(str(lines[0]))
            reply_0 = gen_sentence(str(lines[1]))
            reply_1 = gen_sentence(str(lines[2]))
            write_line = query + '\t' + reply_0 + '\t' + reply_1 + '\n'
            out_data.write(write_line)
    



















