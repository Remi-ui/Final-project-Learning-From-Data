import predict
from sklearn.metrics import classification_report as report
import argparse
import train

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='naive_bayes', choices=['naive_bayes', 'svm', 'svm_optimized'])

    args = parser.parse_args()
    return args

def model_report(model):
    Y_dev, predictions = predict.predict_model(model)
    print(report(Y_dev, predictions, digits=3))

def main():
    args = create_arg_parser()
    model = train.train_model(args.model)
    
    model_report(model)

if __name__ == '__main__':
    main()