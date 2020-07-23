# coding=utf-8
import os
import numpy as np
from transformer import getConfig

#获取配置信息dict
gConfig=getConfig.get_config()

#标记在未在字典中出现的词
UNK='__UNK__'
START_VOCABULART=[UNK]
UNK_ID=1

"""
词频词典的创建：
1、读取所有的文字
2、统计每个文字的出现次数
3、排序
4、取值保存

"""

def create_vocabulary(input_file,vocabulary_size,output_file):
    vocabulary={}
    k=int(vocabulary_size)
    with open(input_file,'r') as f:
        for line in f:
            tokens=[token for token in line.split()]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word]+=1
                else:
                    vocabulary[word]=1
    vocabulary_list=START_VOCABULART+sorted(vocabulary,key=vocabulary.get,reverse=True)
    # 根据配置，取vocabulary_size大小的词典
    if len(vocabulary_list)>vocabulary_size:
        vocabulary_list=vocabulary_list[:vocabulary_size]
    print(input_file+" 文件大小为:",len(vocabulary_list))

    #将生成的词典保存到文件中
    with open(output_file,'w') as f:
        for word in vocabulary_list:
            f.write(word+"\n")


# 把对话字符串转为向量形式

"""
1、遍历文件
2、找到一个字 然后在词典出来，然后做替换
3、保存文件

"""

def convert_to_vector(input_dir,vocabulary_file):
    vocab_tmp=[]
    with open(vocabulary_file,'r') as f:
        vocab_tmp.extend(f.readlines())
    vocab_tmp=[word.strip() for word in vocab_tmp]
    vocab=dict([(y,x) for (x,y) in enumerate(vocab_tmp)])

    docs=[]
    dir=os.listdir(input_dir)
    for i in range(len(dir)):
        with open(input_dir+"/"+dir[i],'r') as f:
            lines = []
            for line in f:
                for word in line.split():
                    lines.append(vocab.get(word,UNK))
            docs.append(lines)

    return docs

"""
1、获取训练和测试数据
2、并转化为向量
"""
def prepare_custom_data(working_directory,train_pos,train_neg,test_pos,test_neg,all_data,vocabulary_size):
    vocab_path=os.path.join(working_directory,'vocab%d.txt'%vocabulary_size)

    #生成字典文件
    create_vocabulary('all_data.txt',vocabulary_size,vocab_path)

    #读取训练数据并转化为向量
    _train_pos=convert_to_vector(train_pos,vocab_path)
    _train_neg=convert_to_vector(train_neg,vocab_path)

    #读取测试数据并转化为向量
    _test_pos = convert_to_vector(test_pos, vocab_path)
    _test_neg = convert_to_vector(test_neg, vocab_path)

    return  _train_pos,_train_neg,_test_pos,_test_neg

train_pos,train_neg,test_pos,test_neg=prepare_custom_data(gConfig['working_directory'],gConfig['train_pos_data'],gConfig['train_neg_data'],gConfig['test_pos_data'],gConfig['test_neg_data'],gConfig['all_data'],gConfig['vocabulary_size'])

y_train=[]
y_test=[]

for i in range(len(train_pos)):
    y_train.append(1)

for i in range(len(train_neg)):
    y_train.append(0)

for i in range(len(test_pos)):
    y_test.append(1)

for i in range(len(test_neg)):
    y_test.append(0)

x_train=np.concatenate((train_pos,train_neg),axis=0)
x_test=np.concatenate((test_pos,test_neg),axis=0)

np.savez("train_data/imdb.npz",x_train,y_train,x_test,y_test)






