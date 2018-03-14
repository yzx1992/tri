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
       

def get_tp_fp(lst,index):
    tp=0
    fp=0
    for i in range(len(lst)):
        if i>index:
            break
        if lst[i][0]==1:
            tp=tp+1
        if lst[i][0]==0:
            fp=fp+1
    #print "tp:%d"%(tp)
    #print "fp:%d"%(fp)

    return (tp,fp)

def get_max_f_prob(lst,true_num):
    max_dict=dict()
    max_val=dict()
    
    for i in range(len(lst)):
        tp,fp=get_tp_fp(lst,i)
        p=float(float(tp)/(tp+fp))
        r=float(float(tp)/true_num)
        if (p+r)==0:
            continue
        f1=((p*r)*2)/(p+r)
        #print f1
        
        max_dict[f1]=(p,r)
        max_val[f1]=lst[i][1]
    
    return max_dict,max_val

cat_dict={2:"caiwu",3:"ditu",4:"gouwu",5:"K12",6:"meishi",7:"yiliao",8:"yuedu",9:"yule",1:"other"}


# Parameters
#tf.flags.DEFINE_string("test_data", "../gen_class_data/test_data/0617tag-3-20.test", "")

tf.flags.DEFINE_string("test_data", "/mnt/workspace/yezhenxu/data/termlevel/len40/test/topic.s2id", "")
#tf.flags.DEFINE_string("test_data", "../gen_class_data/train-0614-20/train-0613-test.seg.id", "")
#tf.flags.DEFINE_string("model_dir", "./model_test0617-20/models/cnn.pb10000", "")
tf.flags.DEFINE_string("model_dir", "/mnt/workspace/yezhenxu/model/term_model/len40/models/cnn.pb10000", "")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x, y = data.LoadDevData(FLAGS.test_data)
print "**********************"
print x 
print "*****************************"
print y
cat_num_ary=[0,0,0,0,0,0,0,0,0]
cat_right_num_ary=[0,0,0,0,0,0,0,0,0]
cat_max_ary=[0,0,0,0,0,0,0,0,0]
cat_min_ary=[1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
cat_total_ary=[0,0,0,0,0,0,0,0,0]
cat_recall_ary=[0,0,0,0,0,0,0,0,0]

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

prob_max=0.0
prob_min=1.0
total_val=0
total_num=len(x)
right_num=0
graph = tf.Graph()
total_right_val=0
cat_prob_dict=dict()
with graph.as_default(), tf.device('/cpu:0'):
    output_graph_def = tf.GraphDef()
    output_graph_path = FLAGS.model_dir
    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name='')
    config = tf.ConfigProto(allow_soft_placement=True)
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
            
            max_val,max_index=get_max_index(p)
            if max_index<0:
               index=index+1
               continue
            print max_val
            total_val=total_val+max_val
            if y[index][max_index]==1:
               total_right_val=total_right_val+max_val
               right_num=right_num+1
               cat_right_num_ary[max_index]=cat_right_num_ary[max_index]+1
               if cat_min_ary[max_index] >max_val:
                   cat_min_ary[max_index]=max_val
               if cat_max_ary[max_index] < max_val:
                   cat_max_ary[max_index]=max_val
               cat_total_ary[max_index]=cat_total_ary[max_index]+max_val
               if cat_prob_dict.has_key(max_index):
                   cat_prob_dict[max_index].append((1,max_val))
               else:
                   lst=[]
                   lst.append((1,max_val))
                   cat_prob_dict[max_index]=lst
            else:
               if cat_prob_dict.has_key(max_index):
                   cat_prob_dict[max_index].append((0,max_val))
               else:
                   lst=[]
                   lst.append((0,max_val))
                   cat_prob_dict[max_index]=lst
            cat_recall_ary[max_index]=cat_recall_ary[max_index]+1
            index=index+1
            if max_val>prob_max:
               prob_max=max_val
            if max_val<prob_min:
               prob_min=max_val
            #print s
            #print '\t'
            #print p
            #print float(s[1]) - float(s[0])
val=float(float(right_num)/total_num)*100
print val
print "total num:%d,right num:%d,accuracy rate:%f%s"%(total_num,right_num,val,"%")
val2=float(total_val/total_num)
val3=float(total_right_val/right_num)
print "total value:%f"%(total_val)
print "total average probability:%f"%(val2)
print "total right average probability2:%f"%(val3)
print "max prob:%f min prob:%f"%(prob_max,prob_min)

for i in range(len(cat_min_ary)):
    print "catalog:%s max probability:%f min probility:%f"%(cat_dict[i+1],cat_max_ary[i],cat_min_ary[i])
    val4=float(cat_total_ary[i]/cat_num_ary[i])
    val5=float(cat_total_ary[i]/cat_right_num_ary[i])
    print "catalog:%s, total average prob:%f right average prob:%f\n"%(cat_dict[i+1],val4,val5)  
  
for i in range(len(cat_right_num_ary)):
    val=float(float(cat_right_num_ary[i])/cat_num_ary[i])*100
    vall=float(float(cat_right_num_ary[i])/cat_recall_ary[i])*100
    f11=(2*val*vall)/(val+vall)
    print "%s catalog, right_num:%d, total_num:%d,recal_num:%d accuracy rate:%f%s,recall rate:%f%s,f11:%f%s"%(cat_dict[i+1],cat_right_num_ary[i],cat_num_ary[i],cat_recall_ary[i],vall,"%",val,"%",f11,"%")

   
print "threshold:"
print len(cat_prob_dict)
for k,v in cat_prob_dict.iteritems():
    #print v
    ary=sorted(v,key=lambda x:x[1],reverse=True)
    #print ary
    #print "total size:%d"%(len(ary))
    max_dict,max_val=get_max_f_prob(ary,cat_num_ary[k]) 
    print "size:%d,%d"%(len(max_dict),len(max_val))
    sort_lst=sorted(max_dict.items(),lambda x,y:cmp(x[0],y[0]),reverse=True)
    sort_lst2=sorted(max_val.items(),lambda x,y:cmp(x[0],y[0]),reverse=True)
    #print sort_lst
    #print sort_lst2
    
    print "catalog:%s max f1:%f precesion:%f recall:%f prob values:%f"%(cat_dict[k+1],sort_lst[0][0],sort_lst[0][1][0],sort_lst[0][1][1],sort_lst2[0][1])
