from sklearn.metrics import classification_report as report
import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = 'experiments/prediction_vs_gold.txt', type=str)
    args = parser.parse_args()
    return args

gold_standard = []
prediction = []
def read_file(path):
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.split(' - ')
            gold_standard.append(line[0])
            prediction.append(line[1])
    class_report = report(gold_standard, prediction, digits=3)
    print(class_report)


def main():
    args = create_arg_parser()
    read_file(args.path)


if __name__ == '__main__':
    main()