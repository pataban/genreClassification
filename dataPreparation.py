import os
# suppress warnings from tensorFlow
# show only: {'0':info, '1':warning, '2':error, '3':fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import chain
from keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler

from constants import *
from support import *


def loadData(dataType=''):
    return pd.read_json(DATA_FIE_PATH + DATA_FILE_NAME + dataType + '.json')


def loadRawData(verbose=False):
    books = pd.read_csv(DATA_FIE_PATH + RAW_DATA_FILE_NAME +
                        RAW_DATA_FILE_EXTENSION, sep='\t')
    books.columns = ('wId', 'fId', 'title',
                     'author', 'date', 'genres', 'summary')

    books = books.drop(
        columns=['wId', 'fId', 'title', 'author', 'author', 'date']).dropna()

    books['genres'] = books['genres'].map(
        lambda genres: list(json.loads(genres).values()))

    if verbose > 1:
        print('raw data:\n', books)
        print('\nraw data info:')
        books.info()
    return books


def printLen(summaries):
    lens = list(map(len, summaries))
    print('Min = %d Avg = %.2f Max = %d' %
          (min(lens), sum(lens)/len(lens), max(lens)))


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


def select(books, selection=None, unique=False):
    genres = []
    summaries = []
    for (genre, summary) in zip(books['genres'], books['summary']):
        if (((not unique) or (len(genre) == 1)) and
                ((selection is None) or (len(set(genre).intersection(selection)) > 0))):
            genres.append(genre[0]if unique else list(
                set(genre).intersection(selection)))
            summaries.append(summary)
    return {'genres': genres, 'summary': summaries}


def selectData(books, verbose=False):
    if verbose > 1:
        print('\nfullGenreCount:\n', countElements(books['genres'].sum(), 10))

    books = select(books, selection=GENRE_INDEX.keys())
    if verbose > 1:
        print('\nselectedGenreCount: All =', len(books['genres']), '\n',
              countElements(chain.from_iterable(books['genres'])))

    books = select(books, unique=True)
    if verbose > 1:
        print('\nuniqueGenreCount: All =', len(books['genres']), '\n',
              countElements(books['genres']))
        print('\ncharacters:')
        printLen(books['summary'])

    return (books['genres'], books['summary'])


def cleanSummaryManual(summary):
    cSummary = ''
    for c in summary:
        if c.isalpha():
            cSummary += c.lower()
        elif c in string.whitespace and cSummary != '' and cSummary[-1] != ' ':
            cSummary += ' '
    cSummary = cSummary.split()
    return cSummary


def cleanSummaries(summaries, verbose=False):
    if CLEAN_SUMMARY_MANUAL:
        summaries = list(map(lambda s: cleanSummaryManual(s)
                         [0:SUMMARY_LENGTH_MAX], summaries))
    else:
        summaries = list(map(lambda s: nltk.word_tokenize(s)
                         [0:SUMMARY_LENGTH_MAX], summaries))
    if verbose > 1:
        print('\nwords:')
        printLen(summaries)
    return summaries


def getWordIndex(summaries, verbose=False):    # TODO [DROP_TOP:KEEP_BOTTOM]
    wordIndex = {word: 0 for word in chain.from_iterable(summaries)}
    wordIndex = dict(zip(wordIndex.keys(), range(1, len(wordIndex)+1)))
    if verbose > 1:
        print('\nwordIndex:\n', wordIndex)
    return wordIndex


def calcPrevalance(summary, wordIndex):
    mappedSummary = np.zeros((len(wordIndex)))
    for word in summary:
        if word in wordIndex:
            mappedSummary[wordIndex[word]-1] += 1
    mappedSummary /= len(summary)
    return mappedSummary


def saveData(saveFileSuffix, booksData):
    # books.to_json(DATA_FIE_PATH + DATA_FILE_NAME + saveFileSuffix + '.json')
    pass


def cleanData(saveFileSuffix=None, verbose=0):
    if verbose > 0:
        print('--------------------------------------------------------------------------')
        print("CleanData")
        print('--------------------------------------------------------------------------')

    books = loadRawData(verbose)
    genres, summaries = selectData(books, verbose=verbose)
    genres = genres[:TRAIN_SIZE+TEST_SIZE]
    summaries = summaries[:TRAIN_SIZE+TEST_SIZE]
    summaries = cleanSummaries(summaries, verbose=verbose)
    wordIndex = getWordIndex(summaries, verbose)

    shuffeledIndexes = tf.random.shuffle(
        tf.range(start=0, limit=len(genres), dtype=tf.int32))

    genres = np.array(list(map(GENRE_INDEX.get, genres)))
    genres = tf.gather(genres, shuffeledIndexes).numpy()
    if verbose > 1:
        print('\nfinal genres:\n', genres)

    summariesWP = np.array(list(map(
        lambda summary: calcPrevalance(summary, wordIndex), summaries)))
    scaler = StandardScaler().fit(summariesWP)
    summariesWP = scaler.transform(summariesWP)
    summariesWP = tf.gather(summariesWP, shuffeledIndexes).numpy()
    if verbose > 1:
        print('\nfinal summariesWP:\n', summariesWP)

    summariesIndexed = list(map(lambda summary: list(map(
        lambda word: wordIndex[word] if word in wordIndex else 0, summary)), summaries))
    summariesIndexed = pad_sequences(
        summariesIndexed, padding='post', dtype=int, value=0)
    summariesIndexed = tf.gather(summariesIndexed, shuffeledIndexes).numpy()
    if verbose > 1:
        print('\nfinal summariesIndexed:\n', summariesIndexed)

    booksData = {
        'genres': (genres[:TRAIN_SIZE],
                   genres[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]),
        'summaries': (summariesIndexed[:TRAIN_SIZE],
                      summariesIndexed[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]),
        'summariesWP': (summariesWP[:TRAIN_SIZE],
                        summariesWP[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]), # WordPrevalances
        'wordIndex': wordIndex
    }

    if saveFileSuffix is not None:
        saveData(saveFileSuffix, booksData)

    return booksData
