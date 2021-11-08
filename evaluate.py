import predict

from sklearn.metrics import classification_report as report
from sklearn.metrics import f1_score

import argparse
import train


def create_arg_parser():
    ''' Creates argparses for all different models. These will be run through both
    train.py and predict.py '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='svm', choices=['naive_bayes', 'svm', 'svm_optimized', 'lstm', 'bert'])
    parser.add_argument('--eval', default = 'newspapers_157_upsampled_test.json', type=str)
    parser.add_argument('--saved_model', default = '', type=str)
    args = parser.parse_args()
    return args

def read_file_bert(file):
    ''' Reads the prediction vs gold standard file and prints a classification report
    for bert generated files '''
    j = 0
    complete = ""
    new_list = []
    gold = []
    predict = []
    with open(file) as f:
        for line in f:
            if line[0] == '[' and j == 0:
                complete = line
            if line[0] == '[':
                new_list.append(complete)
                complete = line
            else:
                complete = complete + line
                complete = complete.replace('\n', '')
            j+=1
        new_list.append(complete)

    for item in new_list[1:]:
        item = item.split(' - ')
        item[0] = item[0].replace('[', '')
        item[0] = item[0].replace(']', '')
        item[1] = item[1].replace('[', '')
        item[1] = item[1].replace(']', '')
        item[0] = item[0].split('.')
        item[1] = item[1].split(' ')
        i = 0
        for score in item[0]:
            score = score.replace(' ', '')
            if score == '1':
                gold.append(i)
            i+=1

        highest_score = 0
        highest = 0
        j=0
        for score in item[1]:
            if score != '':
                score = float(score)
                if score > highest_score:
                    highest_score = score
                    highest = j

                j+=1
        predict.append(highest)
    
    class_report = report(gold, predict, digits=3)
    print(class_report)

def model_report_bert(bert_model, dev_input, dev_labels):
    ''' Prints a classification report for bert models '''
    predictions = bert_model.predict(dev_input)
    i=0
    with open('experiments/prediction_vs_gold.txt', 'w') as f:
        for item in dev_labels:
            f.write("{} - {}\n".format(item, predictions[i]))
            i+=1
    f.close()
        
    read_file_bert('experiments/prediction_vs_gold.txt')

def model_report(model):
    ''' prints out a classification report of the selected model. '''
    Y_dev, predictions = predict.predict_model(model)
    i=0
    with open('experiments/prediction_vs_gold_11.txt', 'w') as f:
        for item in Y_dev:
            f.write("{} - {}\n".format(item, predictions[i]))
            i+=1
    f.close()
    class_report = report(Y_dev, predictions, digits=3)
    print(class_report)


def main():
    args = create_arg_parser()
    model = train.train_model(args.model)
    if args.model == 'bert':
        bert_model, dev_input, dev_labels = train.train_model(args.model)
        model_report_bert(bert_model, dev_input, dev_labels)
    elif args.model != 'lstm':
        model = train.train_model(args.model)
        model_report(model)


if __name__ == '__main__':
    main()