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
        sent_text = nltk.sent_tokenize(i['Content'])
        
        while len(sent_text) >= 24:
            first_sentences = sent_text[:12]
            sent_text = sent_text[13:]

            labels.append(i['Newspaper'])
            documents.append(first_sentences)

        labels.append(i['Newspaper'])
        documents.append(sent_text)

    #print(len(documents))
    #print(len(labels))
    
    #print("")
    #print(Counter(labels).keys()) # equals to list(set(words))
    print(Counter(labels).values()) # counts the elements' frequency

    #times o india, the hindu

    return documents, labels

def main():
    read_data()


if __name__ == '__main__':
    main()