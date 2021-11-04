import predict

from sklearn.metrics import classification_report as report
from sklearn.metrics import f1_score

import argparse
import train


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='svm', choices=['naive_bayes', 'svm', 'svm_optimized', 'lstm', 'bert'])
    parser.add_argument('--eval', default = 'newspapers_157_upsampled_test.json', type=str)
    args = parser.parse_args()
    return args


def model_report(model):
    Y_dev, predictions = predict.predict_model(model)
    i=0
    with open('experiments/svm/prediction_vs_gold.txt', 'w') as f:
        for item in Y_dev:
            f.write("{} - {}\n".format(item, predictions[i]))
            i+=1
    f.close()

    class_report = report(Y_dev, predictions, digits=3)
    print(class_report)
    return f1_score(Y_dev, predictions, average='weighted')


def main():
    args = create_arg_parser()
    model = train.train_model(args.model)
    model_report(model)


if __name__ == '__main__':
    main()