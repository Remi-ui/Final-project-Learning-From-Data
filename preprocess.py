#!/usr/bin/env python

import os
import json
from collections import Counter
import re

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
import pandas as pd
from wordcloud import WordCloud

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')

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
    plt.show(block=True)



def preprocess(text, stem=False):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def wordcloud(df):
    labels = ['Sydney Morning Herald (Australia)', 'The New York Times', 'The Age (Melbourne, Australia)', 'The Washington Post']
    for label in labels:
        plt.figure(figsize=(20, 20))
        wc = WordCloud(max_words = 100 , width = 1600 , height = 800).generate(" ".join(df[df.Labels == label].Documents))
        plt.imshow(wc, interpolation='bilinear')
        plt.show()


def main():

    documents, labels = read_data('../Final-project-Learning-From-Data/all_data', 'all')
    #print(documents[0])
    #print(labels[0])

    # Create pandas dataframe
    df = pd.DataFrame(list(zip(labels, documents)), columns=['Labels', 'Documents'])
    print(df.head())

    # Plot distribution
    plot_distribution(df)

    # Preprocess data and wordcloud
    df.Documents = df.Documents.apply(lambda x: preprocess(x))
    wordcloud(df)




if __name__ == '__main__':
    main()