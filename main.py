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

    start('dataLoad')
    booksData = dp.cleanData(saveFileSuffix='', verbose=1)
    stop('dataLoad')
    # booksData = dp.loadData('')
    # print('\n', booksData, '\n')

    start('naiveBayes')
    sl.classifyNB(booksData['genres'], booksData['summariesWP'], verbose=1)
    stop('naiveBayes')

    start('SVM')
    sl.classifySVM(booksData['genres'], booksData['summariesWP'], verbose=1)
    stop('SVM')

    start('LSTM')
    dl.classify(booksData['genres'], booksData['summaries'], booksData['wordIndex'], verbose=3)
    stop('LSTM')

    print('\n--------------------------------------------------------------------------')
    prtTimes()
