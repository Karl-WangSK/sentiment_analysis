#导入依赖
from tensorflow.keras.layers import LSTM,Bidirectional,Embedding,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from  src.feature_extract import featur_extract
from tensorflow.keras.models import load_model
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
构建  词向量矩阵embedding_matrix 
"""
def embedding_matrix(embedding_dim,num_words,cn_model,path="../embedding/embedding_matrix_"):
    if os.path.exists(path):
        return np.loadtxt(path+num_words)

    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i:] = cn_model[cn_model.index2word[i]]
    np.savetxt((path+num_words), embedding_matrix)
    return embedding_matrix


if __name__ == '__main__':
    path='../model/model.h5'
    #加载词向量
    cn_model=featur_extract.embedding_words()
    #加载模型
    model = load_model(filepath=path)

    # 测试数据
    test_list = [
        '酒店设施不是新的，服务态度很不好',
        '酒店卫生条件非常不好',
        '床铺非常舒适',
        '房间很凉，不给开暖气',
        '房间很凉爽，空调冷气很足',
        '酒店环境不好，住宿体验很不好',
        '房间隔音不到位',
        '晚上回来发现没有打扫卫生',
        '因为过节所以要我临时加钱，比团购的价格贵'
    ]
    for text in test_list:
        re=featur_extract.predict_sentiment(text, cn_model, model)
