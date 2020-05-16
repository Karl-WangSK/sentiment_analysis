#!/usr/bin/env python
#coding: utf-8

 

import jieba
import re 
import numpy as np
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")


 

##知乎词向量
cn_model=KeyedVectors.load_word2vec_format('../embedding/sgns.zhihu.bigram',
                                          binary=False)

print(cn_model['NLP'])
print(cn_model.most_similar(positive='卧槽'))

import os 
pos=os.listdir('../pos')
neg=os.listdir('../neg')
##样本数量
print("负样本数量:",len(neg))

print('len:',str(len(pos)+len(neg)))


##训练数据集
train_texts_orig=[]
for i in range(len(pos)):
    with open('../pos/'+pos[i],'r',errors='ignore') as f:
        txt=re.sub('\\s+','',f.read())
        train_texts_orig.append(txt)
        f.close()
for i in range(len(neg)):
    with open('../neg/'+neg[i],'r',errors='ignore') as f:
        txt=re.sub('\\s+','',f.read())
        train_texts_orig.append(txt)
        f.close()
print(len(train_texts_orig))


#导入依赖
from tensorflow.keras.layers import LSTM,GRU,Bidirectional,Embedding,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import tensorflow as tf


#分词后的文本集
train_tokens=[]
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@￥%……&*（）]+", "",text)
    cut=jieba.cut(text)
    
    cut_list=[i for i in cut]
    for index,word in enumerate(cut_list):
        try:
            cut_list[index]=cn_model.vocab[word].index
        except KeyError:
            cut_list[index]=0
    train_tokens.append(cut_list)
print(train_tokens[3])

##计算需要padding的长度
num_tokens=[len(token) for token in train_tokens]
num_tokens=np.array(num_tokens)
print(num_tokens.mean())
print(num_tokens.max())
print('文本的正态分布2个方差内的长度:',int(num_tokens.mean()+2*np.std(num_tokens)))
print("覆盖了多少",np.sum(num_tokens<=236)/num_tokens.size)


#去掉词向量中没有的词
def reverse_token(token):
    text=''
    for i in token:
        if i !=0:
            text=text+cn_model.index2word[i]
        else:
            text=text+''
    return text

print("去掉词向量中没有的词：",reverse_token(token=train_tokens[2]))

#加载词向量中前30000个
embedding_dim=300
num_words=30000
embedding_matrix=np.zeros((num_words,embedding_dim))

for i in range(num_words):
    embedding_matrix[i:]=cn_model[cn_model.index2word[i]]


print(embedding_matrix.shape)

np.sum(embedding_matrix[333]==cn_model[cn_model.index2word[333]])



train_pad=pad_sequences(train_tokens,maxlen=236,padding='pre',truncating='post')
train_pad[train_pad > num_words]=0
print(train_pad[33])

train_target=np.concatenate((np.ones(2000),np.zeros(2000)))


from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_pad,train_target,test_size=0.1,random_state=123)

print(reverse_token(x_train[123]))
print(y_train[123])


#开始训练

model=Sequential()
model.add(Embedding(input_dim=num_words,output_dim=300,weights=[embedding_matrix],input_length=236))
model.add(Bidirectional(LSTM(units=32, return_sequences=True,dropout=0.2)))
model.add(LSTM(units=16, return_sequences=False,dropout=0.4))
model.add(Dense(1, activation='sigmoid'))
## Adam优化器
adam=Adam(learning_rate=1e-3)
## 编译
model.compile(optimizer=adam,loss="binary_crossentropy",metrics=['accuracy'])

print(model.summary())
##checkpoint
path_checkpoint='../checkpoint/sentiment_checkpoints'
check_point=ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',save_weights_only=True,save_best_only=True)

 #
 # try:
 #     model.load_weights(path_checkpoint)
 # except Exception as e:
 #     print(e)

##callbacks
early_stop=EarlyStopping(monitor='val_loss',patience=3,verbose=1)
lr_reduce=ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_lr=1e-5)
callbacks=[check_point,early_stop,lr_reduce]

##训练模型
model.fit(x_train,y_train,batch_size=128,epochs=20,callbacks=callbacks,validation_split=0.1)

#保存模型
model.save_weights('../model/sm.model')

#evaluate
result=model.evaluate(x_test,y_test)
print(result[1])



#预测模型function
def predict_sentiment(text):
    print(text)
     #去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@￥%……&*（）]+", "",text)
    text=jieba.cut(text)
    text_list=[i for i in text]
    for index,word in enumerate(text_list):
        try:
            text_list[index]=cn_model.vocab[word].index
        except KeyError:
            text_list[index]=0
    text_pad=pad_sequences([text_list],maxlen=236,truncating='post')
    text_pad[text_pad>num_words]=0
    pred=model.predict(text_pad)
    coef=pred[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)
        

#测试数据
test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很凉，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位' ,
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵'
]
for text in test_list:
    predict_sentiment(text)

text='酒店周边环境一般酒店内设施不错感觉还挺新价格也尚可接受'
predict_sentiment(text)



pred=model.predict(x_test)
pred=[1 if i>0.5  else 0 for i in pred]
pred=np.array(pred)

cls=np.where(pred !=y_test)
cls[0]

index=27
print(reverse_token(x_test[index]))
print(pred[index])
print(y_test[index])





