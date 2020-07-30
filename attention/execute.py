# -*- coding:utf-8 -*-
import os
import sys
import time
import tensorflow as tf
from attention import seq2seqModel
from attention import getConfig
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

gConfig=getConfig.get_config()

vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim=gConfig['embedding_dim']
units=gConfig['layer_size']
BATCH_SIZE=gConfig['batch_size']
max_length_inp,max_length_tar=20,20



def tokenize(lang,vocab_size):

    tokenizer=Tokenizer(num_words=vocab_size,oov_token='UNK')
    tokenizer.fit_on_texts(lang)

    tensor=tokenizer.texts_to_sequences(lang)
    tensor=pad_sequences(tensor,max_length_inp,padding='pre',truncating='pre')

    return  tensor,tokenizer

def preprocess_sentence(w):
    return 'start '+w+' end'

def create_dataset(data_path,num_examples):
    lines=io.open(data_path,encoding='UTF-8').read().strip().split("\n")
    word_pairs=[[preprocess_sentence(w) for w in line.split("\t")] for line in lines[:num_examples]]

    return zip(*word_pairs)

def read_data(data_path,num_examples):
    input_lang,target_lang=create_dataset(data_path,num_examples)

    input_tensor,input_tokens=tokenize(input_lang,vocab_inp_size)
    tar_tensor,tar_tokens=tokenize(target_lang,vocab_tar_size)

    return input_tensor,input_tokens,tar_tensor,tar_tokens



input_tensor,input_token,target_tensor,target_token= read_data(gConfig['seq_data'], gConfig['max_train_data_size'])



def train():

    steps_per_epoch=len(input_tensor)//BATCH_SIZE
    print("epoch size is %d" %steps_per_epoch)

    model_dir=gConfig.get('model_data')
    ckpt=tf.io.gfile.listdir(model_dir)
    if ckpt:
        print("reload from %s" %model_dir)
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(model_dir))

    BUFFER_SIZE=len(input_tensor)
    dataset=tf.data.Dataset.from_tensor_slices((input_tensor,target_tensor)).shuffle(BUFFER_SIZE)
    dataset=dataset.batch(BATCH_SIZE,drop_remainder=True)

    ckptManager=tf.train.CheckpointManager(seq2seqModel.checkpoint,model_dir,max_to_keep=5)

    start_time = time.time()

    for epoch in range(gConfig['epoch_size']):
        start_time_epoch=time.time()
        total_loss=0
        epoch+=1

        for (batch,(inp,tar)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss=seq2seqModel.train_step(inp,tar,target_token,seq2seqModel.Encoder.initialize_hidden_state())
            total_loss+=batch_loss
            print(total_loss.numpy())

        step_time_epoch=(time.time()-start_time_epoch)/steps_per_epoch
        step_loss=total_loss/steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('第{}次epoch：训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(epoch,current_steps, step_time_total, step_time_epoch,step_loss))

        ckptManager.save()



def predict(sentence):
    model_dir = gConfig.get('model_data')
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(model_dir))


    sentence=preprocess_sentence(sentence)

    input=[input_token.word_index.get(i,'UNK') for i in sentence.split(' ')]
    input=pad_sequences([input],maxlen=max_length_inp,padding='pre',truncating='pre')


    hidden=tf.zeros((1,units))
    output,enc_hiddden=seq2seqModel.encoder(input,hidden)

    dec_hidden=enc_hiddden
    dec_input=tf.expand_dims([target_token.word_index['start']],0)

    result=''
    for i in range(max_length_tar):
        prediction,state,_=seq2seqModel.decoder(dec_input,dec_hidden,output)

        predition_id=tf.argmax(prediction[0]).numpy()

        if target_token.index_word[predition_id]=='end':
            break
        result+=target_token.index_word[predition_id]

        dec_input=tf.expand_dims([predition_id],0)

    return result

if __name__ == '__main__':

    mode=gConfig['mode']

    if mode=='train':
        train()