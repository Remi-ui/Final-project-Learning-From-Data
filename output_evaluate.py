from sklearn.metrics import classification_report as report
import argparse


def create_arg_parser():
    ''' Creates argparses for this script '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = 'experiments/prediction_vs_gold_11.txt', type=str)
    parser.add_argument('--bert', default = False)
    args = parser.parse_args()
    return args

gold_standard = []
prediction = []


def read_file(path):
    ''' Reads the prediction vs gold standard file and prints a classification report '''
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.split(' - ')
            gold_standard.append(line[0])
            prediction.append(line[1])
    class_report = report(gold_standard, prediction, digits=3)
    print(class_report)


def read_file_bert(path):
    ''' Reads the prediction vs gold standard file and prints a classification report
    for bert generated files '''
    j = 0
    complete = ""
    new_list = []
    gold = []
    predict = []
    with open(path) as f:
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


def main():
    args = create_arg_parser()
    if args.bert == False:
        read_file(args.path)
    else:
        read_file_bert(args.path)


if __name__ == '__main__':
    main()