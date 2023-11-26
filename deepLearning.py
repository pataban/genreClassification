import os
# suppress warnings from tensorFlow
# show only: {'0':info, '1':warning, '2':error, '3':fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from bidict import bidict
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import keras
import nltk
import json



GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, 'Fantasy': 3})
            #({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, "Children's literature": 3, 'Fantasy': 4})
            #'Young adult literature': 5, 'Historical novel': 6, 'Speculative fiction': 7, 'Crime Fiction': 8, 'Non-fiction': 9})
DROP_TOP = 10
KEEP_BOTTOM = 20000
SUMMARY_LENGTH_MIN = 50
SUMMARY_LENGTH_MAX = 200



def loadData():
    books = pd.read_csv('data/booksummaries.txt', sep='\t')
    books.columns = ('wId', 'fId', 'title',
                     'author', 'date', 'genre', 'summary')

    books = books.drop(
        columns=['wId', 'fId', 'title', 'author', 'date', 'author']).dropna()

    books['genre'] = books['genre'].map(
        lambda genre: list(json.loads(genre).values()))
    return books


def countElements(elements, top=None):
    count = {}
    for e in elements:
        if e in count:
            count[e] += 1
        else:
            count[e] = 1

    count = [(count, genre) for (genre, count) in count.items()]
    count.sort(reverse=True)
    if top is not None:
        count = count[0:top]
    return count


def printLen(summaries):
    lens = list(map(len, summaries))
    print('Min = %d Avg = %.2f Max = %d' %
          (min(lens), sum(lens)/len(lens), max(lens)))


def select(books, selection=None, unique=False):
    genres = []
    summaries = []
    for (genre, summary) in zip(books['genre'], books['summary']):
        if ((not unique) or (len(genre) == 1)) and ((selection is None) or (len(set(genre).intersection(selection))>0)):
            genres.append(genre[0]if unique else list(set(genre).intersection(selection)))
            summaries.append(summary)
    return {'genre': genres, 'summary': summaries}


def selectData(books, log=False):
    if log:
        print('\nfullGenreCount:', countElements(books['genre'].sum(), 10))

    books = select(books, selection=GENRE_INDEX.keys())
    if log:
        print('\nselectedGenreCount: All =', len(
            books['genre']), countElements(itertools.chain.from_iterable(books['genre'])))

    books = select(books, unique=True)
    if log:
        print('\nuniqueGenreCount: All =', len(
            books['genre']), countElements(books['genre']))
        print('\ncharacters:')
        printLen(books['summary'])

    return (books['genre'], books['summary'])


def mapSummaries(summaries):
    words = itertools.chain.from_iterable(summaries)
    wordCount = countElements(words)
    wordIndex = {k: v for (k, v) in zip(
        map(lambda wc: wc[1], wordCount[0:KEEP_BOTTOM]), range(1, len(wordCount)+1))}#[DROP_TOP:KEEP_BOTTOM]       ###VERY VERY slow?
    indexedSummaries = list(
        map(lambda s: list(map(lambda w:wordIndex[w] if w in wordIndex else 0, s)), summaries))
    return (indexedSummaries, wordIndex)


def convertData(genres, summaries):
    genres = np.array(list(map(GENRE_INDEX.get, genres)))

    summaries = list(map(lambda s:nltk.word_tokenize(s)[0:SUMMARY_LENGTH_MAX], summaries))
    print('\nwords:')
    printLen(summaries)
    (summaries, wordIndex) = mapSummaries(summaries)
    summaries = keras.utils.pad_sequences(
        summaries, padding='post', dtype=int, value=0)

    shuffeledIndexes = tf.random.shuffle(
        tf.range(start=0, limit=summaries.shape[0], dtype=tf.int32))
    summaries = tf.gather(summaries, shuffeledIndexes).numpy()
    genres = tf.gather(genres, shuffeledIndexes).numpy()

    return (genres, summaries, wordIndex)


if __name__ == '__main__':

    print('\navailable GPU:', tf.config.list_physical_devices('GPU'))

    books = loadData()
    genres, summaries = selectData(books, log=True)
    genres, summaries, wordIndex = convertData(genres, summaries)

    vocabularySize = max(wordIndex.values())+1
    print('vocabulary size = ',vocabularySize)
    print('summaries.shape =', summaries.shape)
    xTrain, XTest, yTrain, yTest = train_test_split(
        summaries, genres, test_size=0.1, random_state=1)


    model = keras.models.Sequential([
        keras.layers.Embedding(
            vocabularySize, 32, input_length=summaries.shape[1], mask_zero=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(4, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    model.summary()


    print("\ntraining model")
    model.fit(xTrain, yTrain, epochs=30, validation_split=0.1, verbose=1, batch_size=128,
            callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True))

    print("\nevaluating model")
    res = model.evaluate(XTest, yTest, verbose=1, batch_size=16)
    print(model.metrics_names[0], res[0])
    print(model.metrics_names[1], res[1], '\n')
