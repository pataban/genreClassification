import os
# suppress warnings from tensorFlow
# show only: {'0':info, '1':warning, '2':error, '3':fatal}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import dataPreparation as dp
import shallowLearning as sl
import deepLearning as dl
from support import *


if __name__ == "__main__":
    print('available GPU:', tf.config.list_physical_devices('GPU'))

    with Timer('dataLoad'):
        booksData = dp.cleanData(verbose=1)

    with Timer('naiveBayes'):
        sl.classify(sl.NB, booksData, verbose=1)

    with Timer('SVM'):
        sl.classify(sl.SVM, booksData, verbose=1)

    with Timer('LSTM'):
        dl.classify(dl.LSTM, booksData, verbose=3)

    with Timer('MLP'):
        dl.classify(dl.MLP, booksData, verbose=3)

    with Timer('CNN'):
        dl.classify(dl.CNN, booksData, verbose=3)

    with Timer('MNN'):
        dl.classify(dl.MNN, booksData, verbose=3)

    print('\n--------------------------------------------------------------------------')
    Timer.prtTimes()
