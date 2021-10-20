import json
from sklearn.metrics import classification_report as report
from sklearn.naive_bayes import MultinomialNB
import os

def read_data():
    i = 0
    documents_dev = []
    labels_dev = []
    os.chdir('../dev')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name.endswith('filt3.sub.json'):
                file = open(os.path.join(root, name))
                text = json.load(file)
                for article in text['articles']:
                    i += 1
                    documents_dev.append(article['body'])
                    labels_dev.append(article['newspaper'])
    return documents_dev, labels_dev

def predict_model(model):
    X_dev, Y_dev = read_data()
    predictions = model.predict(X_dev)
    return Y_dev, predictions


if __name__ == '__main__':
    main()