import json
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import evaluate
import predict

def read_data():
    i = 0
    documents = []
    labels = []
    os.chdir('train')
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

def train_naive_bayes(X_train, Y_train):
    vec = TfidfVectorizer()
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)
    return naive_classifier

def train_svm_optimized(X_train, Y_train):
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

def train_lstm(X_train, Y_train):
    print('Still some code needed!')
    return

def train_bert(X_train, Y_train):
    print('Still some code needed!')
    return

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
        bert_model = train_bert(X_train, Y_train)
    else:
        print('Something went wrong, please execute this program again and type --help after.')


if __name__ == '__main__':
    main()