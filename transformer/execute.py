# coding=utf-8
import tensorflow as tf
import numpy as np
from transformer.textClassiferModel import *
import time
from transformer.data_util import get_train_test,reverse_token,padding,load_embedding_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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
构建transformer模型
"""
def create_model(model,ckpt):
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    ckpt_dir = tf.io.gfile.listdir(model_dir)
    # 查看检查点
    if ckpt_dir:
        print("restore model ")
        ckpt.restore(tf.train.latest_checkpoint(model_dir))
        return model
    else:
        # 构建transformer神经网络
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
def train(train_data,embedding_matrix_weight,train_status=True):
    transformer=Transformer(gConfig['num_layers'], gConfig['num_heads'], gConfig['embedding_size'], gConfig['diff'],
                gConfig['vocabulary_size'], embedding_matrix_weight, gConfig['dropout_rate'])

    # 优化器
    optimizer = Adam(learning_rate=gConfig.get('learning_rate'))
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    #create model
    transformer = create_model(transformer,ckpt)

    # ckpt manager
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)

    #each epoch
    for epoch in range(gConfig['epochs']):

        for batch, (x, y) in enumerate(train_data.batch(gConfig['batch_size'])):

            start = time.time()
            loss = 0
            mask = create_padding_mask(input)
            if train_status:
                with tf.GradientTape() as tape:
                    predictions = transformer(x, True, mask)
                    tar = to_categorical(y, 2)

                    loss = tf.losses.categorical_crossentropy(tar, predictions)
                    loss = tf.reduce_mean(loss)

                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            else:
                # 预测
                predictions = transformer(x, False, mask)

            print('训练集:Epoch {:} ,Batch {:} ,Loss {:.4f},Prestep {:.4f}'.format(epoch, batch, loss.numpy(),
                                                                                (time.time() - start)))
        ckpt_save_path = ckpt_manager.save()
        print('保存epoch{}模型在 {}'.format(epoch, ckpt_save_path))



"""
将文本向量转化为数字向量
"""
def convert_to_vec(input):
    vocab_tmp = []
    with open(gConfig['vocabulary_file'], 'r') as f:
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
    txt_vec = padding([txt_vec])
    input = tf.reshape(txt_vec[0], [1, txt_vec[0]])

    # 预测结果
    prediction = model.step(input, 3, False)
    # 获取结果的索引
    pre_index = tf.math.argmax(prediction[0, :])
    pre_index = np.int32(pre_index.numpy())

    return state[pre_index]

if __name__ == '__main__':
    train_pad,train_target,cn_model=get_train_test()

    # 切分训练数据和测试数据

    x_train, x_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.1, random_state=123)
    print(reverse_token(x_train[123],cn_model))
    print(x_train[123])

    # 把数据和标签 组成dataset [(x,y),(x,y).....]
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(gConfig['shuffle_size'])
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(gConfig['shuffle_size'])


    print("batch/epoch: ",len(x_train)//gConfig['batch_size'])

    embedding_matrix=load_embedding_matrix(cn_model,embedding_dim=300,num_words=vocab_size)

    train(train_data=train_data,embedding_matrix_weight=embedding_matrix,train_status=True)


