#!/usr/bin/env python
# coding=utf-8 
'''
Created on 2014/10/09

@author: fanyange
'''
import json
import sys
import os

class item:
    term_num=0
    doc_num=0
    def __init__(self,tnum,dnum):
        self.term_num=tnum
        self.doc_num=dnum


def load_ok_words(path):
    fp=open(path,"r")

    word_dict=dict()
    while 1:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if line == "":
            continue
        terms = line.split("\t")
        if len(terms)!=3:
            continue
        word_dict[terms[0]]=terms[1]+"\t"+terms[2]
    return word_dict


def load_dict(dict_path):
    fp=open(dict_path,"r")

    num_dict=dict()
    index=0
    lines=fp.readlines()
    print "lines %d"%(len(lines))
    for line in open(dict_path):
        line = line.strip()
        if line == "":
            print "%d null"%(index)
            continue
        terms = line.split("\t")
        if len(terms) !=2:
            print "%d error:%s"%(index,line)            
            num_dict[index]=terms[0]+"\t"+" "
            index=index+1
            continue
        num_dict[index]=line
        index=index+1
    fp.close()
    print len(num_dict)
    return num_dict
def change_encode(str,fromencode,toencode):
    try:
        u=str.decode(fromencode)
        s=u.encode(toencode)
        return (True,s)
    except:
        return (False,str)

def get_files(path):
	lst=[]
	for dirpath,dirnames,filenames in os.walk(path):
		for file in filenames:
			fullpath=os.path.join(dirpath,file)
			lst.append(file)
			#print fullpath
	return lst


def get_catalogs(file_lst,cat_dir):
	cat_dict=dict()
	for i in range(len(file_lst)):
		names=file_lst[i].split(".")
		if len(names) != 2:
			continue
		cat_path=cat_dir+"/"+file_lst[i]

		fp=open(cat_path)
		while 1:
			line = fp.readline()
			if not line:
				break
			line = line.strip()
			if line == "":
				continue
			items=line.split("\t")
			if len(items)<2:
				print line
				continue
			cat_dict[items[0]]=items[1].decode("gbk").encode("utf-8")
		fp.close()
	return cat_dict



def treate_file(input_path,new_path,out_path):
    #file_lst=get_files(input_dir)

    old_lines=load_dict(input_path)

    print "old size:%d"%(len(old_lines))
	
    #cat_dict=get_catalogs(file_lst,cat_dir)
    #print "catalog size:%d\n"%(len(cat_dict))
    out_fp=open(out_path,"w")
	#file_dict[names[0]]=fp
    fp=open(new_path,"r")
    cat_dict={2:"caiwu",3:"ditu",4:"gouwu",5:"K12",6:"meishi",7:"yiliao",8:"yuedu",9:"yule",1:"other"}
    
    new_lines=[]
    index=0
    while 1:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if line == "":
            continue
        items=line.split("\t")
        if len(items)!=3:
            continue
        new_lines.append(line)
        index=index+1
        #if index==196256:
        #   break

        #out_fp.write("%s\t%s\n"%(cat,line))

    fp.close()

    if len(old_lines)!=len(new_lines):
        print "line size is not equal old:%d new:%d"%(len(old_lines),len(new_lines))
        return 
    for i in range(len(new_lines)):
        items=new_lines[i].split("\t")
        old_items=old_lines[i].split("\t")
        if cat_dict[int(old_items[0])]!=items[1]:
           print "%s %s is not %s\n"%(cat_dict[int(old_items[0])],old_line[i],new_lines[i])
           break
        out_fp.write("%s\t%s\t%s\n"%(items[0],items[1],old_items[1]))
    out_fp.close()


    print "success"
        


if __name__=="__main__":
    treate_file(sys.argv[1],sys.argv[2],sys.argv[3])



