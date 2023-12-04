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
        booksData = dp.cleanData(saveFileSuffix='', verbose=1)

    with Timer('naiveBayes'):
        sl.classifyNB(booksData['genres'], booksData['summariesWP'], verbose=1)

    with Timer('SVM'):
        sl.classifySVM(booksData['genres'],
                       booksData['summariesWP'], verbose=1)

    with Timer('LSTM'):
        dl.classifyLSTM(booksData['genres'], booksData['summaries'],
                        booksData['wordIndex'], booksData['embeddingMatrix'], verbose=3)

    with Timer('MLP'):
        dl.classifyMLP(booksData['genres'], booksData['summariesWP'],
                       booksData['wordIndex'], verbose=3)

    with Timer('CNN'):
        dl.classifyCNN(booksData['genres'], booksData['summaries'],
                        booksData['wordIndex'], booksData['embeddingMatrix'], verbose=1)

    print('\n--------------------------------------------------------------------------')
    Timer.prtTimes()
