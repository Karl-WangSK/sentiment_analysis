import re
import jieba
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


class featur_extract():

    def embedding_words(path='../embedding/sgns.zhihu.bigram'):
        ##知乎词向量
        return KeyedVectors.load_word2vec_format(path,
                                                     binary=False)

    #预测模型function
    def predict_sentiment(text,cn_model,model,num_words=300):
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