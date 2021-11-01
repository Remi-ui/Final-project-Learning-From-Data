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
    for i in data:
        doc = i['Content']
        while len(doc) > 440:
            documents.append(doc[:220])
            labels.append(i['Newspaper'])
            doc = doc[220:]

        labels.append(i['Newspaper'])
        documents.append(doc)

    #times o india, the hindu

    return documents, labels

def main():
    documents, labels = read_data()

    print(len(documents))
    print(len(labels))

    print(Counter(labels))

    if os.path.exists("newspapers_91_upsampled.json"):
        os.remove("newspapers_91_upsampled.json")
    
    jsonList= []
    i=-1
    with open('newspapers_91_upsampled.json', 'w') as file:
        for item in documents:
            i+=1
            jsonList.append({"Newspaper" : labels[i], "Content" : item})
        json.dump(jsonList, file)

if __name__ == '__main__':
    main()