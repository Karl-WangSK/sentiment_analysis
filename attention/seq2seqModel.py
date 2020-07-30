import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Embedding
from attention.getConfig import get_config
class Encoder(Model):
    def __init__(self, enc_hidden, batch_size, embedding_dim, vocab_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.enc_hidden = enc_hidden

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.gru = GRU(self.enc_hidden, return_sequences=True, return_state=True)

    def call(self, inputs, hidden):
        x = self.embedding(inputs)

        output, state = self.gru(x)

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros([self.batch_size, self.enc_hidden])


class BahdanauAttention(Model):
    def __init__(self, dec_hidden):
        super(BahdanauAttention, self).__init__()

        self.w1 = Dense(dec_hidden)
        self.w2 = Dense(dec_hidden)
        self.v = Dense(1)

    def call(self, query, values):
        """
        values: encoder output
            (batch_size,seq_length,hidden_size)
        query: encoder last hidden state
            (batch_size,hidden_size)
        hidden_with_time_axis:
            (batch_size,1,hidden_size)
        """
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score:
        #  (batch_size,seq_length,1)
        score = self.v(tf.nn.tanh(
            self.w1(values) + self.w2(hidden_with_time_axis)
        ))

        # attention_weight:
        #    (batch_size,seq_length,1)
        attention_weight = tf.nn.softmax(score, axis=1)

        """
        context_vector:
            before reduce_sum :  (batch_size,seq_length,hidden_size)
            after: (batch_size,hidden_size)
        """
        context_vector = attention_weight * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weight


class Decoder(Model):
    def __init__(self, embedding_dim, vocab_size, dec_hidden):
        super(Decoder, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.gru = GRU(dec_hidden, return_state=True, return_sequences=True)
        self.dense = Dense(vocab_size)

        self.attention = BahdanauAttention(dec_hidden)

    def call(self, inputs, enc_output, enc_state):
        # context_vec:
        #   (batch_size,hidden_size)
        context_vec, attention_weight = self.attention(enc_state, enc_output)

        # (batch_size,1,embedding_size)
        x = self.embedding(inputs)

        # (batch_size,1,hidden_size+embedding_size)
        x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)

        # (batch_size,1,hidden_size+embedding_size)
        output, state = self.gru(x)

        # (batch_size,hidden_size+embedding_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # (batch_size,vocab_size)
        predictions = self.dense(output)

        return predictions, state, attention_weight


gConfig = get_config()

vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
batch_size = gConfig['batch_size']

encoder = Encoder(units, batch_size, embedding_dim, vocab_inp_size)

decoder = Decoder(embedding_dim, vocab_inp_size, units)

optimizer = tf.keras.optimizers.Adam(lr=0.001)

sparse_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


def train_step(inp, targ, targ_lang, enc_hidden):
    """

    :param inp:
    :param targ: (batch_size,output_seq_length)
    :param targ_lang:
    :param enc_hidden:
    :return:
    """
    loss = 0

    with tf.GradientTape() as tape:
        output, enc_hidden = encoder(inputs=inp, enc_hidden=enc_hidden)

        dec_hidden = enc_hidden

        # (batch_size, 1)
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * batch_size, 1)

        for i in range(targ.shape[1]):
            predictions, state, attention_weight = decoder(dec_input, output, dec_hidden)
            # targ[:, t]  (batch_size)
            # predictions   (batch_size,vocab_size)
            loss += loss_function(targ[:, i], predictions)

            dec_hidden = state

            dec_input = tf.expand_dims(targ[:, i], axis=1)

    batch_loss = (loss / int(targ.shape[0]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def loss_function(real, pred):
    loss = sparse_loss(real, pred)

    return loss.numpy()
