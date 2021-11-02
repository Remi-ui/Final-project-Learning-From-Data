import random
import json
import os
from collections import Counter
import nltk

from collections import Counter

def read_data():
    labels = []
    documents = []
    f = open('newspapers_91.json')
    data = json.load(f)
    y = []
    x = 0
    for i in data:
        doc = i['Content']
        x += 1
        while len(doc) > 440:

            first_word = doc.split()[0]
            doc = doc[len(first_word):]
            documents.append(doc[:220])
            labels.append(i['Newspaper'])
            y.append(x)

            doc = doc[220:]

        labels.append(i['Newspaper'])
        documents.append(doc)
        y.append(x)

    #times o india, the hindu

    return documents, labels, y

def main():
    documents, labels, y = read_data()

    # Make a balanced set of 1155 labels per class
    counter = Counter()
    i = -1
    labels_balanced = []
    documents_balanced = []
    y_balanced = []

    for item in labels:
        i+=1
        if item not in counter.keys():
            labels_balanced.append(item)
            documents_balanced.append(documents[i])
            y_balanced.append(y[i])
            counter = Counter(labels_balanced)
        elif item in counter.keys() and counter.get(item) < 1155:
            labels_balanced.append(item)
            documents_balanced.append(documents[i])
            y_balanced.append(y[i])
            counter = Counter(labels_balanced)

    # Out of the balanced set make another balanced set for the dev and test set
    counter = Counter()
    counter_test = Counter()
    labels_dev = []
    documents_dev = []
    y_dev = []

    labels_test = []
    documents_test = []
    i = -1
    for item in labels_balanced:
        i+=1
        if item not in counter.keys():
            labels_dev.append(item)
            documents_dev.append(documents_balanced[i])
            y_dev.append(y_balanced[i])
            counter = Counter(labels_dev)
        elif item in counter.keys() and counter.get(item) < 120:
            labels_dev.append(item)
            documents_dev.append(documents_balanced[i])
            y_dev.append(y_balanced[i])
            counter = Counter(labels_dev)

        elif counter.get(item) == 120:
            if item not in counter_test.keys():
                labels_test.append(item)
                documents_test.append(documents_balanced[i])
            
                y_dev.append(y_balanced[i])
                counter_test = Counter(labels_test)
            elif item in counter_test.keys() and counter_test.get(item) < 120:
                labels_test.append(item)
                documents_test.append(documents_balanced[i])
                
                y_dev.append(y_balanced[i])
                counter_test = Counter(labels_test)

    
    print("Dev set: {}\n".format(Counter(labels_dev)))
    # Send the balanced dev set to a JSON file
    if os.path.exists("newspapers_91_upsampled_dev.json"):
        os.remove("newspapers_91_upsampled_dev.json")
    
    jsonList= []
    i=-1
    with open('newspapers_91_upsampled_dev.json', 'w') as file:
        for item in documents_dev:
            i+=1
            jsonList.append({"Newspaper" : labels_dev[i], "Content" : item})
        json.dump(jsonList, file)

    print("Test set: {}\n".format(Counter(labels_test)))
    jsonList= []
    i=-1
    with open('newspapers_91_upsampled_test.json', 'w') as file:
        for item in documents_test:
            i+=1
            jsonList.append({"Newspaper" : labels_test[i], "Content" : item})
        json.dump(jsonList, file)

    # Remove all items that appear in the dev/test set from the balanced set,
    # but ALSO remove all entries in the balanced set of which a part of it
    # appears in the dev/test set (so that parts of articles don't appear in both the train and dev/test set.)
    i = -1
    remove_indecies = []
    for item in y_balanced:
        i+=1
        if item in y_dev:
            remove_indecies.append(i)

    for index in sorted(remove_indecies, reverse=True):
        del documents_balanced[index]
        del labels_balanced[index]
    
    print("Train set: {}\n".format(Counter(labels_balanced)))
    # Send the final version of the balanced set to a json file.
    if os.path.exists("newspapers_91_upsampled.json"):
        os.remove("newspapers_91_upsampled.json")
    
    jsonList= []
    i=-1
    with open('newspapers_91_upsampled.json', 'w') as file:
        for item in documents_balanced:
            i+=1
            jsonList.append({"Newspaper" : labels_balanced[i], "Content" : item})
        json.dump(jsonList, file)

if __name__ == '__main__':
    main()