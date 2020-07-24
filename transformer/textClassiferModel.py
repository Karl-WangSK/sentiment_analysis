# coding=utf-8
import tensorflow as tf
import numpy as np
from transformer import getConfig
from tensorflow.keras.layers import Layer, Dense,Embedding
from tensorflow.nn import softmax
from tensorflow.keras import Sequential

gConfig = {}
gConfig = getConfig.get_config()


def point_wise_feed_forward_network(diff, d_model):
    return Sequential(
        [Dense(diff, activation='relu'),
         Dense(d_model)]
    )


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

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

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


class EncoderLayer(Layer):

    def __init__(self, num_heads, diff, d_model, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mth = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(diff, d_model)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask):
        attn_output, _ = self.mth(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        output1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(output1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return output2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_heads,d_model,diff,vocabulary_size,rate=0.1):
        super(Encoder,self).__init__()

        self.num_layers=num_layers
        self.d_model=d_model

        self.embedding=Embedding(vocabulary_size,d_model)
        self.position_encoding=self.positional_encoding(vocabulary_size,d_model)

    def get_angles(self,pos,i,d_model):
        return


    def positional_encoding(self,position,d_model):
        angle_out=self.get_angles(np.arange(position)[:np.newaxis],
                        np.arange(d_model)[np.newaxis:d_model],
                        d_model)

        return 1













