# coding=utf-8
import tensorflow as tf
import numpy as np
from transformer import getConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformer import textClassiferModel as model
from tensorflow.data import Dataset
import time


#config
gConfig = getConfig.get_config()
#params
sentence_size = gConfig['sentence_size']
embedding_size = gConfig['embedding_size']
vocab_size = gConfig['vocabulary_size']
model_dir = gConfig['model_dir']
UNK_ID = 1



"""
加载训练和测试数据
"""

def read_npz(data_file):
    r = np.load(data_file, allow_pickle=True)
    return r['arr_0'], r['arr_1'], r['arr_2'], r['arr_3']


"""
对输出向量长度补齐 padding为0
"""

def pad_sequences(input):
    return pad_sequences(input, maxlen=sentence_size, padding='pre', truncating='post')



"""
构建transformer模型
"""
def create_model():
    ckpt = tf.io.gfile.listdir(model_dir)
    # 查看检查点
    if ckpt:
        print("restore model ")
        model.ckpt.restore(tf.train.latest_checkpoint(model_dir))
        return model
    else:
        return model


"""
训练数据：
    1、加载模型
    2、循环N个epoch
    3、传入每个batch的数据去训练
    4、保存checkpoint
params:
    train_data 训练数据
    ckpt_manager  模型保存
"""
def train(train_data,ckpt_manager):
    model = create_model()

    for epoch in range(gConfig.get('epoch')):
        for batch, (x, y) in enumerate(train_data.train(gConfig.get("batch_size"))):
            start = time.time()
            loss = model.step(x, y)
            print('训练集:Epoch {:} ,Batch {:} ,Loss {:.4f},Prestep {:.4f}'.format(epoch, batch, loss.numpy(),
                                                                                (time.time() - start)))

    ckpt_save_path = ckpt_manager.save()
    print('保存epoch{}模型在 {}'.format(epoch, ckpt_save_path))



"""
将文本向量转化为数字向量
"""
def convert_to_vec(input):
    vocab_tmp = []
    with open(gConfig.get('vocabulary_file'), 'r') as f:
        vocab_tmp.extend(f.readlines())
    vocab_tmp = [word.strip() for word in vocab_tmp]
    vocab = dict([(y, x) for (x, y) in enumerate(vocab_tmp)])

    txt_vec = []
    for word in input.split():
        txt_vec.append(vocab.get(word, UNK_ID))

    return txt_vec

"""
预测输入
"""
def predict(sentences):
    # 标签
    state = ['neg', 'pos']

    model = create_model()

    # 将输入转化为vec，并做padding
    txt_vec = convert_to_vec(sentences)
    txt_vec = pad_sequences([txt_vec])
    input = tf.reshape(txt_vec[0], [1, txt_vec[0]])

    # 预测结果
    prediction = model.step(input, 3, False)
    # 获取结果的索引
    pre_index = tf.math.argmax(prediction[0, :])
    pre_index = np.int32(pre_index.numpy())

    return state[pre_index]

if __name__ == '__main__':
    # 获取训练测试数据
    x_train, y_train, x_test, y_test = read_npz(gConfig.get('npz_data'))
    # padding
    x_train = pad_sequences(x_train)
    x_test = pad_sequences(x_test)

    # 把数据和标签 组成dataset [(x,y),(x,y).....]
    train_data = Dataset.from_tensor_slices((x_train, y_train)).shuffle(gConfig.get('shuffle_size'))
    test_data = Dataset.from_tensor_slices((x_test, y_test)).shuffle(gConfig.get('shuffle_size'))

    # ckpt manager
    ckpt_manager = tf.train.CheckpointManager(model.ckpt, model_dir, max_to_keep=5)

    train(train_data=train_data,ckpt_manager=ckpt_manager)


