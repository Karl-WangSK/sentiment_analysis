# coding=utf-8
import tensorflow as tf
import numpy as np
from transformer import getConfig
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.nn import softmax
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.losses import categorical_crossentropy



"""
FFN:前馈神经网络
2层神经网络
"""
def point_wise_feed_forward_network(diff, d_model):
    return Sequential(
        [Dense(diff, activation='relu'),
         Dense(d_model)]
    )

"""
MultiHeadAttention：
    1、对Q、K、V 拆分成多头
    2、Q、K、V scaled点积相乘  
        Return:  (batch_size,num_heads,seq_len_k,d_model)
    3、转置后合并成 三维   
        Return: (batch_size, seq_len_k, d_model)
"""
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model & self.num_heads == 0

        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)

        self.dense = Dense(self.d_model)

    # 把Q,K,V分成 multi  head
    def split_head(self, x, batch_size):
        x = tf.reshape(x, tf.shape(x)[:-1] + [self.num_heads, self.d_model // self.num_heads])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # Q K V 点积相乘
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        """
        dk为缩放因子
        当 dk较大时，向量内积的值也会容易变得很大，这时 softmax 函数的梯度会非常的小
        """
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        """
        padding之后为0的，但这样做得话，softmax计算就会被影响，e^0=1也就是有值，这样就会影响结果，这并不是我们希望看到得，
        因此在计算得时候我们需要把他们mask起来，填充一个负无穷（-1e9这样得数值），这样计算就可以为0了，等于把计算遮挡住。
        """
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_k, d_model)

        return output, attention_weights

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size,seq_len,d_model)
        k = self.wq(k)  # (batch_size,seq_len,d_model)
        v = self.wq(v)  # (batch_size,seq_len,d_model)

        q = self.split_head(q, batch_size)  # (batch_size,num_heads,seq_len_q,d_model)
        k = self.split_head(k, batch_size)  # (batch_size,num_heads,seq_len_k,d_model)
        v = self.split_head(v, batch_size)  # (batch_size,num_heads,seq_len_v,d_model)

        scaled_attention, attention_weight = self.scaled_dot_product_attention(q, k, v,
                                                                               mask)  # (batch_size,num_heads,seq_len_k,d_model)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2])  # (batch_size,seq_len_k,num_heads,d_model)
        # 合并最后2个维度
        a, b = tf.shape(scaled_attention)[-2:]
        combined_attention = tf.reshape(scaled_attention,
                                        tf.shape(scaled_attention)[-2:] + [a * b])  # (batch_size, seq_len_k, d_model)

        output = self.dense(combined_attention)

        return output, attention_weight

"""
单层TransformerBlock:
    MultiHeadAttention -> dropout -> LayerNormalization -> 
    FFN -> dropout -> LayerNormalization
"""
class TransformerBlock(Layer):

    def __init__(self, num_heads, diff, d_model, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.mth = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(diff, d_model)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask):
        attn_output, _ = self.mth(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        output1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(output1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return output2

"""
Encoder :
    1、输入向量embedding + positional_encoding
    2、dropout
    3、循环N次TransformerBlock

"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, diff, vocabulary_size, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(vocabulary_size, d_model)
        self.position_encoding = self.positional_encoding(vocabulary_size, d_model)

        self.enc_layers = [TransformerBlock(num_heads, diff, d_model) for i in range(num_layers)]

        self.dropout1 = Dropout(rate)

    def get_angles(self, pos, i, d_model):
        angle_rads = 1 / np.power(100000, (2 * i) / tf.float32(d_model))

        return pos * angle_rads

    def positional_encoding(self, position, d_model):
        # 构建一个位置编码的向量， 维度和embedding相同
        position_vec = np.zeros([position, d_model], dtype=float)

        angle_out = self.get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(0, d_model, 2)[np.newaxis, :],
                                    d_model)

        position_vec[:, 0::2] = np.sin(angle_out)
        position_vec[:, 1::2] = np.cos(angle_out)
        # 添加第0个维度
        position_vec = position_vec[np.newaxis, ...]

        return tf.cast(position_vec, tf.float32)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # 添加位置编码
        x += self.position_encoding[:, :seq_len, :]

        x = self.dropout1(x, training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

"""
Transformer:
    1、构建encoder
    2、dropout
    3、reshape [batch_size,output_shape]
    4、Dense(2,"softmax")
"""
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, d_model, diff, vocabulary_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, num_heads, d_model, diff, vocabulary_size)

        self.ffn = Dense(2, activation='softmax')

        self.dropout = Dropout(rate)

    def call(self, input, training, enc_padding_mask):
        x = self.encoder(input, training, enc_padding_mask)

        out_shape = gConfig['sentence_size'] * gConfig['embedding_size']

        out = tf.reshape(x, [-1, out_shape])

        out = self.dropout(out, training)
        ffn = self.ffn(out)

        return ffn

"""
填充长度不足的seq为-1e9
"""
def create_padding_mask(seq):
    seq=tf.cast(tf.math.equal(seq,0),tf.float32)
    return seq[:,np.newaxis,np.newaxis,:]  #(batch_size,1,1,seq_len)


"""
train
"""
def step(input,tar,train_status=True):
    mask=create_padding_mask(input)
    #是否需要训练
    if train_status:
        with tf.GradientTape() as tape:
            predictions=transformer(input,True,mask)
            tar=to_categorical(tar,2)
            loss=categorical_crossentropy(tar,predictions)
            loss=tf.reduce_mean(loss)
            print("训练损失数值为：",loss)

        gradient=tape.gradient(loss,transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradient,transformer.trainable_variables))
        return  loss
    else:
        #预测
        predictions = transformer(input, False, mask)
        return predictions


gConfig = getConfig.get_config()


#构建transformer神经网络
transformer = Transformer(gConfig['num_layers'],gConfig['embedding_size'],gConfig['diff'] ,gConfig['num_heads'],
                          gConfig['vocabulary_size'],gConfig['dropout_rate'])
# 优化器
optimizer = Adam(learning_rate=gConfig.get('learning_rate'))
ckpt=tf.train.Checkpoint(transformer,optimizer)






