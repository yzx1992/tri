import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data
from tensorflow.contrib import learn
import sys


def get_max_index(lst):
    if len(lst)<=0:
       return (-1,-1)
    max2=lst[0]
    max_index=0
    for i in range(1,len(lst)):
       if lst[i]>max2:
          max2=lst[i]
          max_index=i
    return (max2,max_index)

def get_index(ary):
    index=-1
    for i in range(len(ary)):
       if ary[i]==1:
          index=i
          break
    return index
       
cat_dict={2:"caiwu",3:"ditu",4:"gouwu",5:"K12",6:"meishi",7:"yiliao",8:"yuedu",9:"yule",1:"other"}

cat_threshold_dict={2:0.80,3:0.9,4:0.60,5:0.40,6:0.30,7:0.50,8:0.70,9:0.80,1:0.20}

# Parameters
tf.flags.DEFINE_string("test_data", "../gen_class_data/train-0617-20/all_train.id", "")
#tf.flags.DEFINE_string("test_data", "../gen_class_data/train-0614-20/train-0613-test.seg.id", "")
tf.flags.DEFINE_string("model_dir", "./model_test0617-20/models/cnn.pb10000", "")
tf.flags.DEFINE_string("output_path","../gen_class_data/train-0617-20/all_train.id.res","")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x, y = data.LoadDevData(FLAGS.test_data)

cat_num_ary=[0,0,0,0,0,0,0,0,0]
cat_right_num_ary=[0,0,0,0,0,0,0,0,0]

cat_num_total=0
for i in range(len(y)):
    index=get_index(y[i])
    if index<0:
        continue
    cat_num_ary[index]=cat_num_ary[index]+1
    cat_num_total=cat_num_total+1
    #print y[i]
    #if i>10:
    #    break

#sys.exit()

print cat_num_total
print len(x)
print len(y)

# Evaluation
# ==================================================


fp=open(FLAGS.output_path,"w")
total_val=0
total_num=len(x)
right_num=0
res_lst=[]
graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    output_graph_def = tf.GraphDef()
    output_graph_path = FLAGS.model_dir
    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name='')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():

        tf.initialize_all_variables().run()
        input_x = sess.graph.get_tensor_by_name('import/dev_x:0')
        input_y = sess.graph.get_tensor_by_name('import/dev_y:0')
        dropout_keep_prob = sess.graph.get_tensor_by_name('import/dropout_keep_prob:0')
        scores = sess.graph.get_tensor_by_name('import/output/scores:0') 
        probs = sess.graph.get_tensor_by_name('import/output/probs:0') 

        batch_scores, batch_probs = sess.run([scores, probs], {input_x: x, input_y: y, dropout_keep_prob:1.0})
        #print len(batch_scores)
        #print len(batch_probs)

        index=0
        for s, p  in zip(batch_scores, batch_probs):
            #max_val,max_index=get_max_index(s)
            
            max_val,max_index=get_max_index(s)
            if max_index<0:
               index=index+1
               continue
            total_val=total_val+max_val
            pred_cata=cat_dict[old_index+1]
            if y[index][max_index]==1:
               right_num=right_num+1
               cat_right_num_ary[max_index]=cat_right_num_ary[max_index]+1
               if max_val<(cat_threshold_dict[max_index+1]/2):
                   pred_cata=cat_dict[1]
            else:
               if max_val>=cat_threshold_dict[max_index+1]:
                   pre_cata=cat_threshold_dict[max_index+1]
               else:
                   pred_cata=cat_dict[1]   
 
           
            old_index=get_index(y[index])
           
            lst=x[index].tolist()
            res_str=str(lst[0])
            for i in range(1,len(lst)):
                res_str=res_str+" "+ str(lst[i])
            
            #if max_val>=0.85:
            #    fp.write("%s\t%s\t%s\n"%(cat_dict[max_index+1],cat_dict[old_index+1],res_str))
            #else:
            #    fp.write("%s\t%s\t%s\n"%(cat_dict[old_index+1],cat_dict[old_index+1],res_str))
            #res_str2=pred_cata+"\t"+cat_dict[old_index+1]+"\t"+res_str
            #res_lst.append(res_str2)
            fp.write("%s\t%s\t%s\n"%(pred_cata,cat_dict[old_index+1],res_str))
            index=index+1
            if index%10000==0:
                fp.close()

                fp=open(FLAGS.output_path,"a")
                #fp.flush()
            #print s
            #print '\t'
            #print p
            #print float(s[1]) - float(s[0])
fp.close()
val=float(float(right_num)/total_num)*100
print val
print "total num:%d,right num:%d,accuracy rate:%f%s"%(total_num,right_num,val,"%")
val2=float(total_val/total_num)*100
print "total probability:%f%s"%(val2,"%")

#for i in rangel(len(res_lst)):
#    fp.write("%s\n"%(res_lst[i]))


for i in range(len(cat_right_num_ary)):
    val=float(float(cat_right_num_ary[i])/cat_num_ary[i])*100
    print "%s catalog, right_num:%d, total_num:%d,accuracy rate:%f%s"%(cat_dict[i+1],cat_right_num_ary[i],cat_num_ary[i],val,"%")
