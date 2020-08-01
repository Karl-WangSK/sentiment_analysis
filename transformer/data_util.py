# coding=utf-8
import os
import numpy as np
from transformer import getConfig
from gensim.models import KeyedVectors
import jieba
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

#获取配置信息dict
gConfig=getConfig.get_config()

#标记在未在字典中出现的词
UNK='__UNK__'
START_VOCABULART=[UNK]
UNK_ID=1

num_words = gConfig['vocabulary_size']
seq_len=gConfig['sentence_size']

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
    create_vocabulary(all_data,vocabulary_size,vocab_path)

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


"""
获取知乎skip-gram词向量
"""
def get_embedding(path):
    ##知乎词向量
    cn_model = KeyedVectors.load_word2vec_format(path,
                                                 binary=False)
    print(cn_model['NLP'])
    print(cn_model.most_similar(positive='卧槽'))
    return cn_model

"""
获取数据集 并转换为index
"""
def get_train_tokens(cn_model):
    pos = os.listdir('../pos')
    neg = os.listdir('../neg')
    ##样本数量
    print("负样本数量:", len(neg))
    print('len:', str(len(pos) + len(neg)))

    ##训练数据集
    train_texts_orig = []
    for i in range(len(pos)):
        with open('../pos/' + pos[i], 'r', errors='ignore') as f:
            txt = re.sub('\\s+', '', f.read())
            train_texts_orig.append(txt)
            f.close()
    for i in range(len(neg)):
        with open('../neg/' + neg[i], 'r', errors='ignore') as f:
            txt = re.sub('\\s+', '', f.read())
            train_texts_orig.append(txt)
            f.close()
    print(len(train_texts_orig))

    # 分词后的文本集
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@￥%……&*（）]+", "", text)
        cut = jieba.cut(text)

        cut_list = [i for i in cut]
        for index, word in enumerate(cut_list):
            try:
                cut_list[index] = cn_model.vocab[word].index
            except KeyError:
                cut_list[index] = 0
        train_tokens.append(cut_list)
    print(train_tokens[3])


    return train_tokens


 # index to word
def reverse_token(token,cn_model):
    text = ''
    for i in token:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ''
    return text



def padding(train_tokens):
    # padding
    train_pad = pad_sequences(train_tokens, maxlen=seq_len, padding='pre', truncating='post')
    train_pad[train_pad > num_words] = 0
    print(train_pad[33])

    return train_pad

"""
获取数据和label
词袋化过的
"""
def get_train_test(model_path='../embedding/sgns.zhihu.bigram'):
    #embedding
    cn_model=get_embedding(model_path)
    #get train data
    train_tokens=get_train_tokens(cn_model)
    #padding data
    train_pad=padding(train_tokens)
    #tar data
    train_target = np.concatenate((np.ones(2000), np.zeros(2000)))

    return train_pad,train_target
