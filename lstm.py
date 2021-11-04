import os
import json
import random as python_random
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout
from keras.layers.core import Dense
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

np.random.seed(11)
tf.random.set_seed(11)
python_random.seed(11)


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


def read_data2(directory):
    labels = []
    documents = []
    f = open(directory)
    data = json.load(f)
    for i in data:
        labels.append(i['Newspaper'])
        documents.append(i['Content'])

    return documents, labels


def read_embeddings(embeddings_file):
    embeddings = open(embeddings_file, 'r', encoding='utf-8')
    embeddings_list = [line.split() for line in embeddings]
    embeddings.close()
    return {line[0]: np.array(line[1:-1]) for line in embeddings_list}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    embedding_dim = len(emb["the"])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model(Y_train, emb_matrix):
    '''Create the Keras model to use'''
    learning_rate = 0.01
    loss_function = 'categorical_crossentropy'
    # optim = SGD(learning_rate=learning_rate)
    optim = Adam(learning_rate=learning_rate)
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = 9

    # Simple LSTM
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=True))
    model.add(LSTM(units=128, input_dim=embedding_dim))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    # Bidirectional LSTM with 2 LSTM layers
    # model = Sequential()
    # model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=True))
    # model.add(Bidirectional(LSTM(units=128, input_dim=embedding_dim, return_sequences=True)))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(units=128, input_dim=embedding_dim)))
    # model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here. Note the different settings you can experiment with!'''
    verbose = 1
    batch_size = 16
    epochs = 10
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    t0 = time.time()
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs,
              callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    train_time = time.time() - t0
    print("Training time: ", round(train_time, 1), "seconds.")

    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))
    print(classification_report(Y_test, Y_pred))


def main():
    # On this data I get an accuracy of ~0.7
    X_train, Y_train = read_data('/content/drive/MyDrive/Project/train')
    X_dev, Y_dev = read_data('/content/drive/MyDrive/Project/dev')
    X_train, Y_train = X_train[:4956], Y_train[:4956]
    X_test, Y_test = read_data('/content/drive/MyDrive/Project/test')
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train + X_dev + X_test, Y_train + Y_dev + Y_test,
                                                      test_size=0.2, random_state=0, shuffle=True)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_dev, Y_dev, test_size=0.5,
                                                    random_state=0, shuffle=True)

    # On this data I get an accuracy of 0.111 (random guess)
    # X_train, Y_train = read_data2('/content/drive/MyDrive/Project/newspapers_157_upsampled.json')
    # X_dev, Y_dev = read_data2('/content/drive/MyDrive/Project/newspapers_157_upsampled_dev.json')
    # X_test, Y_test = read_data2('/content/drive/MyDrive/Project/newspapers_157_upsampled_test.json')
    print(Counter(Y_train))
    print(Counter(Y_dev))
    print(Counter(Y_test))

    embeddings = read_embeddings('/content/drive/MyDrive/glove/glove.6B.100d.txt')
    vectorizer = TextVectorization(standardize=None, output_sequence_length=200)

    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)

    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # encoder = LabelEncoder()
    # encoder.fit(Y_train + Y_dev)
    # Y_train_bin = encoder.transform(Y_train)
    # Y_dev_bin = encoder.transform(Y_dev)

    model = create_model(Y_train, emb_matrix)

    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    Y_test_bin = encoder.fit_transform(Y_test)
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
    test_set_predict(model, X_test_vect, Y_test_bin, "test")


if __name__ == '__main__':
    main()