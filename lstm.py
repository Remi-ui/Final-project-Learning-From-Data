#!/usr/bin/env python

import os
import json

import numpy as np
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


def read_data(directory):
    i = 0
    documents = []
    labels = []
    os.chdir(directory)
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name.endswith('filt3.sub.json'):
                file = open(os.path.join(root, name))
                text = json.load(file)
                for article in text['articles']:
                    i += 1
                    documents.append(article['body'])
                    labels.append(article['newspaper'])
    return documents, labels


def tokenize(train, dev):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(train), maxlen=30)
    x_dev = pad_sequences(tokenizer.texts_to_sequences(dev), maxlen=30)

    return vocab_size, word_index, x_train, x_dev


def word_embeddings(vocab_size, word_index):
    os.chdir('/glove')
    glove_emb = 'glove.6B.300d.txt'
    embedding_dim = 300
    lr = 1e-3
    batch_size = 1024
    epochs = 10
    model_path = '../glove/best_model.hdf5'

    embeddings_index = {}
    file = open(glove_emb)
    for line in file:
        values = line.split()
        word = value = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()
    print(len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=30,
        trainable=False
    )
    return embedding_layer


def model(embedding_layer):
    sequence_input = Input(shape=(30,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Conv1D(64, 5, activation='relu')(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(sequence_input, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=['accuracy'])
    reducelr = ReduceLROnPlateau(factor=0.1,
                                 min_lr=0.01,
                                 monitor='val_loss',
                                 verbose=1)
    return model


def main():
    X_train, Y_train = read_data('/content/drive/MyDrive/AS5/test')
    # print(X_train[0])
    X_dev, Y_dev = read_data('/content/drive/MyDrive/AS5/dev')
    # print(X_dev[0])

    vocab_size, word_index, x_train, x_dev = tokenize(X_train, X_dev)
    embedding_layer = word_embeddings(vocab_size, word_index)
    model(embedding_layer)


if __name__ == '__main__':
    main()