import json
from sklearn.metrics import classification_report as report
from sklearn.naive_bayes import MultinomialNB
import evaluate
import os

def read_data():
    labels_dev = []
    documents_dev = []
    args = evaluate.create_arg_parser()
    f = open(args.eval)
    data = json.load(f)
    for i in data:
        labels_dev.append(i['Newspaper'])
        documents_dev.append(i['Content'])
    
    return documents_dev, labels_dev


def predict_model(model):
    X_dev, Y_dev = read_data()
    predictions = model.predict(X_dev)
    return Y_dev, predictions


if __name__ == '__main__':
    main()