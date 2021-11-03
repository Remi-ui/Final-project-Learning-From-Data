#!/usr/bin/env python

import os
import json
from collections import Counter
import re
import random

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')


def read_data(directory, amount):
    '''Reads the overall data of the project and returns either all or the first 157
    of the entire dataset. The amount (all or 157) is selected via a parameter.'''
    documents = []
    labels = []
    both = []
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
                        both.append([article['newspaper'], article['body']])
                    elif amount == 'some':
                        if article['newspaper'] not in counter.keys():
                            documents.append(article['body'])
                            labels.append(article['newspaper'])
                            both.append([article['newspaper'], article['body']])
                            counter = Counter(labels)
                        elif article['newspaper'] in counter.keys() and counter.get(article['newspaper']) < 157:
                            documents.append(article['body'])
                            labels.append(article['newspaper'])
                            both.append([article['newspaper'], article['body']])
                            counter = Counter(labels)
    if amount == 'all':
        print(Counter(labels))
    elif amount == 'some':
        print(counter)
    return documents, labels, both


def shuffle_data(both):
    '''Shuffles the lists of labels and documents in the total data set. Then
    takes 157 of these randomly shuffled labels and documents and returns those.'''
    random.shuffle(both)
    shuffled_both = []
    labels = []
    counter = Counter()
    for i in both:
        if i[0] not in counter.keys():
            shuffled_both.append(i)
            labels.append(i[0])
            counter = Counter(labels)
        elif i[0] in counter.keys() and counter.get(i[0]) < 157:
            shuffled_both.append(i)
            labels.append(i[0])
            counter = Counter(labels)
    return shuffled_both


def write_json(shuffled_both):
    '''Takes the 157 randomly selected newspapers and writes them into a .json file.'''
    os.chdir('..')
    if os.path.exists("newspapers_157.json"):
        os.remove("newspapers_157.json")
    i = -1
    jsonList = []
    with open('newspapers_157.json', 'w') as file:
        for item in shuffled_both:
            i += 1
            jsonList.append({"Newspaper": shuffled_both[i][0], "Content": shuffled_both[i][1]})
        json.dump(jsonList, file)


def preprocess_json():
    '''Removes url's and emails from the upsampled dataset to prevent overfitting to
    certain texts.'''
    url = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    email = r'\S*@\S*\s?'
    os.chdir('..')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name.startswith('newspapers_157_upsampled'):
                file = open(os.path.join(root, name))
                data = json.load(file)
                jsonList = []
                for line in data:
                    content = re.sub(url, '', line['Content'], flags=re.MULTILINE)
                    content = re.sub(email, '', line['Content'], flags=re.MULTILINE)
                    jsonList.append({"Newspaper": line['Newspaper'], "Content": content})
                file.close()
                os.remove(name)
                with open(name, 'w') as wfile:
                    json.dump(jsonList, wfile)


def plot_distribution(df):
    '''Plots the distribution of the overall dataset.'''
    val_count = df.Labels.value_counts()
    plt.figure(figsize=(8,4))
    plt.bar(val_count.index, val_count.values)
    plt.xticks(rotation=90)
    plt.title("Label Data Distribution")
    plt.show(block=True)


def preprocess(text, stem=False):
    '''Helper function that preprocesses text by removing url's, stems words,
    and removes stopwords.'''
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
    '''Creates a wordcloud for some specified labels.'''
    labels = ['Sydney Morning Herald (Australia)', 'The New York Times', 'The Age (Melbourne, Australia)', 'The Washington Post']
    for label in labels:
        plt.figure(figsize=(20, 20))
        wc = WordCloud(max_words = 100 , width = 1600 , height = 800).generate(" ".join(df[df.Labels == label].Documents))
        plt.imshow(wc, interpolation='bilinear')
        plt.show()


def main():
    # Read the data and preprocess it by shuffling it.
    documents, labels, both = read_data('../Final-project-Learning-From-Data/all_data', 'all')
    shuffled_both = shuffle_data(both)

    # Create .json with data (this will overwrite the current 157 used newspapers.
    #write_json(shuffled_both)

    # Preprocess .json by removing url's and emails (overwrites the current upsampled newspapers data)
    preprocess_json()

    # Preprocess data for plotting
    df = pd.DataFrame(list(zip(labels, documents)), columns=['Labels', 'Documents'])
    df.Documents = df.Documents.apply(lambda x: preprocess(x))

    # Data analysis by plotting the distribution and possible wordclouds
    #plot_distribution(df)
    #wordcloud(df)


if __name__ == '__main__':
    main()