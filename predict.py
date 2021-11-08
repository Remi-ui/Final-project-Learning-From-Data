import json
from sklearn.metrics import classification_report as report
from sklearn.naive_bayes import MultinomialNB
import evaluate
import os


def read_data():
    '''Reads the evaluation data, depending on the test/dev set that is provided
    as an argument using evaluate.py --eval. Depending on the file that is supplied
    the documents and labels are returned.'''
    labels_dev = []
    documents_dev = []
    args = evaluate.create_arg_parser()
    if args.eval == 'newspapers_157_upsampled_test.json' or args.eval == 'newspapers_157_upsampled_dev.json':
        f = open(args.eval)
        data = json.load(f)
        for i in data:
            labels_dev.append(i['Newspaper'])
            documents_dev.append(i['Content'])
    else:
        f = open(args.eval)
        data = json.load(f)
        for article in data['articles']:
            documents_dev.append(article['body'])
            labels_dev.append(article['newspaper'])
    return documents_dev, labels_dev


def predict_model(model):
    '''Reads the evaluation data and predicts on that data with the
    model that is trained by the user. Returns the gold labels and
    the labels that the model predicted.'''
    X_dev, Y_dev = read_data()
    predictions = model.predict(X_dev)
    return Y_dev, predictions


if __name__ == '__main__':
    main()