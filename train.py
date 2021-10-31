import json
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

import evaluate
import predict
from collections import Counter

import numpy as np
import tensorflow_hub as hub
import tokenization
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder  
from keras.utils import np_utils
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)


def read_data():
    i = 0
    documents = []
    labels = []
    os.chdir('/content/gdrive/MyDrive/AS5/all_set')
    counter = Counter()
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name.endswith('filt3.sub.json'):
                file = open(os.path.join(root, name))
                text = json.load(file)    
                for article in text['articles']:
                    if article['newspaper'] not in counter.keys():
                        documents.append(article['body'])
                        labels.append(article['newspaper'])
                        counter = Counter(labels)
                    elif article['newspaper'] in counter.keys() and counter.get(article['newspaper']) < 91:
                        documents.append(article['body'])
                        labels.append(article['newspaper'])
                        counter = Counter(labels)
    print(len(labels))
    print(len(documents))
    if os.path.exists("newspapers_91.json"):
        os.remove("newspapers_91.json")
    jsonList= []
    i=-1
    with open('newspapers_91.json', 'w') as file:
        for item in documents:
            i+=1
            jsonList.append({"Newspaper" : labels[i], "Content" : item})
        json.dump(jsonList, file)

    return documents, labels


def train_naive_bayes(X_train, Y_train):
    vec = TfidfVectorizer()
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)
    return naive_classifier


def train_svm_optimized2(X_train, Y_train):
    vec = TfidfVectorizer(ngram_range=(1,3))
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('svc', SVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier


def train_svm(X_train, Y_train):
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('svc', SVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier


def train_svm_optimized(X_train, Y_train):
    '''Trains and conducts hyperparameter optimization on a linear SVM.
    Param grid dictionary can be expanded with additional parameters.'''
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('linearsvc', LinearSVC(random_state=0))])
    svm_classifier = svm_classifier.fit(X_train, Y_train)
    f1 = evaluate.model_report(svm_classifier)
    param_grid = {
        'linearsvc__C': [0.1, 1, 10],
    }

    best_score = f1
    for g in ParameterGrid(param_grid):
        svm_classifier.set_params(**g)
        svm_classifier.fit(X_train, Y_train)
        current_score = evaluate.model_report(svm_classifier)
        if current_score > best_score:
            best_score = current_score
            best_grid = g
    print(best_grid)
    print(best_score)
    return svm_classifier


def train_lstm(X_train, Y_train):
    print('Still some code needed!')
    return


def bert_encode(texts, tokenizer, max_len=512):

    #texts = np.array(texts)[indices.astype(int)]

    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def train_bert(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    print("/nwith lstm now@!/n")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    x = tf.expand_dims(clf_output, axis=-1)
    x = LSTM(128)(x)
    x = Dropout(0.25)(x)
    x = Dense(9, activation='softmax')(x)
    opt = SGD(learning_rate=0.01)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=x)
    model.compile(optimizer=opt , loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model):
    X_train, Y_train = read_data()

    if model == 'naive_bayes':
        naive_classifier = train_naive_bayes(X_train, Y_train)
        return naive_classifier
    elif model == 'svm':
        svm_classifier = train_svm(X_train, Y_train)
        return svm_classifier
    elif model == 'svm_optimized':
        svm_classifier = train_svm_optimized(X_train, Y_train)
        return svm_classifier
    elif model == 'lstm':
        lstm_model = train_lstm(X_train, Y_train)
    elif model == 'bert':
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        le = LabelEncoder()
        Y_train_vec = np_utils.to_categorical(le.fit_transform(Y_train))
        
        X_train, X_dev, Y_train_vec, Y_dev = train_test_split(X_train, Y_train_vec, test_size=0.2, shuffle=True)
        
        #X_dev, Y_dev = predict.read_data()
        train_input = bert_encode(X_train, tokenizer, max_len=200)
        dev_input = bert_encode(X_dev, tokenizer, max_len=200)
        
        train_labels = Y_train_vec
        dev_labels = Y_dev

        bert_model = train_bert(bert_layer, max_len=200)
        print("done!")

        train_history = bert_model.fit(
            train_input, train_labels,
            validation_data=(dev_input, dev_labels),
            epochs=100,
            batch_size=32
        )
    else:
        print('Something went wrong, please execute this program again and type --help after.')


if __name__ == '__main__':
    main()