#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import GeneSeg
import csv,pickle,random,json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

vec_dir="file\\word2vec.pickle"
pre_datas_train="file\\pre_datas_train.csv"
pre_datas_test="file\\pre_datas_test.csv"
process_datas_dir="file\\process_datas.pickle"

def pre_process():
    with open(vec_dir,"rb") as f :
        #print(f.readlines())
        word2vec=pickle.load(f)
        print(type(word2vec))#<class 'dict'>
        print(len(word2vec))#7
        print(word2vec)
        dictionary=word2vec["dictionary"]
        reverse_dictionary=word2vec["reverse_dictionary"]
        embeddings=word2vec["embeddings"]
    xssed_data=[]
    normal_data=[]
    with open("data\\xssedtiny.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            xssed_data.append(word)
    with open("data\\normal_less.csv","r",encoding="utf-8") as f:
        reader=csv.reader(f)
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            normal_data.append(word)
    xssed_num=len(xssed_data)
    normal_num=len(normal_data)
    xssed_labels=[1]*xssed_num   #标签
    normal_labels=[0]*normal_num
    datas=xssed_data+normal_data
    labels=xssed_labels+normal_labels
    labels=to_categorical(labels)#Converts a class vector (integers) to binary class matrix.
    def to_index(data):#去word2vec里查询
        d_index=[]
        for word in data:
            #print(word)
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        #print(d_index)
        return d_index
    datas_index=[to_index(data) for data in datas[0:]]
    for i in datas_index:
        if len(i)>100:
            print(1,len(i))
    datas_index=pad_sequences(datas_index,value=-1)
    #print(datas_index)
    rand=random.sample(range(len(datas_index)),len(datas_index))#打乱重采样
    datas=[datas_index[index] for index in rand]
    labels=[labels[index] for index in rand]
    train_datas,test_datas,train_labels,test_labels=train_test_split(datas,labels,test_size=0.3)
    #print('train data:',train_datas)
    train_size=len(train_labels)
    #print(train_size)#15381
    test_size=len(test_labels)
    #print(test_size)#6593
    input_num=len(train_datas[0])
    #print(input_num)  #258
    #print('sb',type(embeddings))
    dims_num = embeddings["UNK"]
    #print(dims_num) #128
    word2vec["train_size"]=train_size  #15381
    word2vec["test_size"]=test_size   #6593
    word2vec["input_num"]=input_num  #258
    word2vec["dims_num"]=dims_num   #128
    with open(vec_dir,"wb") as f :
        pickle.dump(word2vec,f)
    print("Saved word2vec to:",vec_dir)
    print("Write trian datas to:",pre_datas_train)
    with open(pre_datas_train,"w") as f:
        for i in range(train_size):
            data_line=str(train_datas[i].tolist())+"|"+str(train_labels[i].tolist())+"\n"
            f.write(data_line)
    print("Write test datas to:",pre_datas_test)
    with open(pre_datas_test,"w") as f:
        for i in range(test_size):
            data_line=str(test_datas[i].tolist())+"|"+str(test_labels[i].tolist())+"\n"
            f.write(data_line)
    print("Write datas over!")
def data_generator(data_dir):
    reader = tf.TextLineReader()
    queue = tf.train.string_input_producer([data_dir])
    _, value = reader.read(queue)
    # Start populating the filename queue.
    coord = tf.train.Coordinator() #创建一个协调器，管理线程  
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)#启动QueueRunner, 此时文件名队列已经进队。  
    while True:
        v = sess.run(value)
        [data, label] = v.split(b"|")
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array(json.loads(label.decode("utf-8")))
        yield (data, label)
    coord.request_stop()
    coord.join(threads)
    sess.close()
def batch_generator(datas_dir,datas_size,batch_size,embeddings,reverse_dictionary,train=True):
    batch_data = []
    batch_label = []
    generator=data_generator(datas_dir)
    n=0
    while True:
        for i in range(batch_size):
            data,label=next(generator)
            data_embed = []            
            for d in data:
                if d != -1:
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(embeddings["UNK"]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n+=1
            if not train and n==datas_size:
                break
        if not train and n == datas_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data),np.array(batch_label))
            batch_data = []
            batch_label = []
def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size=word2vec["train_size"]
    test_size=word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num =word2vec["input_num"]
    train_generator = batch_generator(pre_datas_train,train_size,batch_size,embeddings,reverse_dictionary)
    test_generator = batch_generator(pre_datas_test,test_size,batch_size,embeddings,reverse_dictionary,train=False)
    return train_generator,test_generator,train_size,test_size,input_num,dims_num
if __name__=="__main__":
    pre_process()





