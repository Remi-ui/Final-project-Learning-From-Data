import json
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid

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