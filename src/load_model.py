#导入依赖
from tensorflow.keras.layers import LSTM,Bidirectional,Embedding,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from  src.feature_extract import featur_extract
import numpy as np
import os


"""
num_words: number of embedding words
embedding_matrix: embedding_dimension
"""
def build_model(num_words,embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=300, weights=[embedding_matrix], input_length=236))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.2)))
    model.add(LSTM(units=16, return_sequences=False, dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    ## Adam优化器
    adam = Adam(learning_rate=1e-3)
    ## 编译
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])

    return model


"""
加载词向量模型
"""
def load_model(model,path):
    return model.load_weights(filepath=path)

"""
构建  词向量矩阵embedding_matrix 
"""
def embedding_matrix(embedding_dim,cn_model,path="../embedding/embedding_matrix"):
    if os.path.exists(path):
        return np.loadtxt(path)

    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i:] = cn_model[cn_model.index2word[i]]
    return embedding_matrix


if __name__ == '__main__':
    embedding_dim = 300
    num_words = 30000
    path='../model/sm.model'
    #加载词向量
    cn_model=featur_extract.embedding_words()
    #构建embedding_matrix
    embedding_matrix=embedding_matrix(embedding_dim=embedding_dim,cn_model=cn_model)
    np.savetxt(("../embedding/embedding_matrix"),embedding_matrix)
    #加载模型
    model=build_model(num_words=num_words,embedding_matrix=embedding_matrix)
    model=load_model(model,path=path)
    text="酒店周边环境一般酒店内设施不错感觉还挺新价格也尚可接受"
    re=featur_extract.predict_sentiment(text,cn_model,model)
    print(re)