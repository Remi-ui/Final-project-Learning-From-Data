#!/usr/bin/env python

import os
import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

def read_data(directory, amount):
    documents = []
    labels = []
    os.chdir(directory)
    counter = Counter()
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name.endswith('filt3.sub.json'):
                file = open(os.path.join(root, name))
                text = json.load(file)
                for article in text['articles']:
                    if amount == 'all':
                        documents.append(article['body'])
                        labels.append(article['newspaper'])
                    elif amount == 'some':
                        if article['newspaper'] not in counter.keys():
                            documents.append(article['body'])
                            labels.append(article['newspaper'])
                            counter = Counter(labels)
                        elif article['newspaper'] in counter.keys() and counter.get(article['newspaper']) < 156:
                            documents.append(article['body'])
                            labels.append(article['newspaper'])
                            counter = Counter(labels)
    # length = 0
    # for i in documents:
    #    length += len(i)
    # print(length/len(documents))

    if amount == 'all':
        print(Counter(labels))
    elif amount == 'some':
        print(counter)

    return documents, labels


def plot_distribution(df):
    val_count = df.Labels.value_counts()
    plt.figure(figsize=(8,4))
    plt.bar(val_count.index, val_count.values)
    plt.xticks(rotation=90)
    plt.title("Label Data Distribution")
    plt.show()


def main():
    documents, labels = read_data('../Final-project-Learning-From-Data/all_data', 'some')
    print(documents[0])
    print(labels[0])

    df = pd.DataFrame(list(zip(labels, documents)), columns=['Labels', 'Documents'])

    print(df.head())

    plot_distribution(df)

if __name__ == '__main__':
    main()