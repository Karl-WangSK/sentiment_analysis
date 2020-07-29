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

    tokenizer=Tokenizer(num_words=vocab_size,oov_token=3)
    tokenizer.fit_on_texts(lang)

    tensor=tokenizer.texts_to_sequences(lang)
    tensor=pad_sequences(tensor,max_length_inp,padding='pre',truncating='pre')

    return  tensor,tokenizer

def preprocess_sentence(w):
    return 'START '+w+' END'

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
