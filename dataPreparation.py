import json
import string
from copy import deepcopy
from itertools import chain
import pandas as pd
from pandas import DataFrame
import nltk
from constants import *


def loadData(dataType='') -> DataFrame:
    return pd.read_json(DATA_FIE_PATH + DATA_FILE_NAME + dataType + '.json')


def cleanData(saveFileSuffix=None) -> DataFrame:
    books = pd.read_csv(DATA_FIE_PATH + RAW_DATA_FILE_NAME +
                        RAW_DATA_FILE_EXTENSION, sep='\t')
    books.columns = ('wId', 'fId', 'title', 'author',
                     'date', 'genres', 'summary')
    books = books.drop(
        columns=['wId', 'fId', 'title', 'author', 'date']).dropna()
    books['genres'] = books['genres'].map(
        lambda genres: list(json.loads(genres).values()))

    books['genres'] = books['genres'].map(
        lambda genres: list(set(SELECTED_GENRES).intersection(genres)))
    books['genres'] = books['genres'].map(
        lambda genres: genres[0] if len(genres) == 1 else None)
    books = books.dropna()
    books.columns = ('genre', 'summary')

    # print(countElements(books['genre']))
    # books.info()
    # print('\n',books,'\n')

    books['summary'] = books['summary'].map(cleanSummary)
    books = books.dropna()
    # books.info()
    # print('\n',books,'\n')
    vocabulary = {word: 0 for word in chain.from_iterable(books['summary'])}
    books['summary'] = books['summary'].map(
        lambda summary: calcPrevalance(summary, vocabulary))
    # books.info()
    # print('\n',books,'\n')

    if saveFileSuffix is not None:
        books.to_json(DATA_FIE_PATH + DATA_FILE_NAME +
                      saveFileSuffix + '.json')
    return books


def countElements(elements, top=None):
    count = {}
    for e in elements:
        if e in count:
            count[e] += 1
        else:
            count[e] = 1

    count = list(count.items())
    count.sort(key=lambda gc: gc[1], reverse=True)
    if top is not None:
        count = count[0:top]
    return count


def cleanSummary(summary):
    cSummary = ''
    if CLEAN_SUMMARY_MANUAL:
        for c in summary:
            if c.isalpha():
                cSummary += c.lower()
            elif c in string.whitespace and cSummary != '' and cSummary[-1] != ' ':
                cSummary += ' '
        cSummary = cSummary.split()
    else:
        cSummary = nltk.word_tokenize(summary)
    return cSummary if SUMMARY_MIN_LENGTH <= len(cSummary) <= SUMMARY_MAX_LENGTH else None


def calcPrevalance(summary, vocabulary):
    vocabulary = deepcopy(vocabulary)
    for word in summary:
        vocabulary[word] += 1
    return list(map(lambda wc: wc/len(summary), vocabulary.values()))
