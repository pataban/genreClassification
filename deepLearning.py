import keras
import tensorflow as tf

from constants import *
from dataPreparation import *


def classify(genres, summaries, wordIndex, verbose):
    print('\n--------------------------------------------------------------------------')
    print('LSTM')
    print('--------------------------------------------------------------------------')

    vocabularySize = max(wordIndex.values())+1
    if 0 < verbose < 3:
        print('vocabulary size = ', vocabularySize)
        print('summaries.shape =', summaries.shape)

    xTrain, XTest = summaries
    yTrain, yTest = genres

    model = keras.models.Sequential([
        keras.layers.Embedding(
            vocabularySize, 32, input_length=xTrain.shape[1], mask_zero=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(4, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    if 0 < verbose < 3:
        model.summary()

    if verbose > 0:
        print("training...")
    model.fit(xTrain, yTrain, epochs=30, validation_split=0.1, verbose=verbose, batch_size=128,
              callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      patience=3, restore_best_weights=True))
    if 0 < verbose < 4:
        print('')

    print("evaluating...")
    res = model.evaluate(XTest, yTest, verbose=verbose, batch_size=128)

    print('loss: %.4f' % (res[0]))
    print('accuracy: %.4f' % (res[1]))
