import os
# suppress warnings from tensorFlow
# show only: {'0':info, '1':warning, '2':error, '3':fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

from constants import *
from dataPreparation import *


def clasify(genres, summaries, wordIndex, verbose):
    print('\n--------------------------------------------------------------------------')
    print('LSTM')
    print('--------------------------------------------------------------------------')

    print('available GPU:', tf.config.list_physical_devices('GPU'))

    vocabularySize = max(wordIndex.values())+1
    if 0 < verbose < 3:
        print('vocabulary size = ', vocabularySize)
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
    if 0 < verbose < 3:
        model.summary()

    if verbose > 0:
        print("training")
    model.fit(xTrain, yTrain, epochs=30, validation_split=0.1, verbose=verbose, batch_size=128,
              callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      patience=3, restore_best_weights=True))
    if 0 < verbose < 4:
        print('')

    print("evaluating")
    res = model.evaluate(XTest, yTest, verbose=verbose, batch_size=16)
    print(model.metrics_names[0], res[0])
    print(model.metrics_names[1], res[1])
