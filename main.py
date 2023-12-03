import svm
import naiveBayes as nb
import deepLearning as dl
import dataPreparation as dp
from support import *


if __name__ == "__main__":
    start('dataLoad')
    booksData = dp.cleanData(saveFileSuffix='', verbose=1)
    stop('dataLoad')
    # booksData = dp.loadData('')
    # print('\n', booksData, '\n')

    start('naiveBayes')
    nb.clasify(booksData['genres'], booksData['summariesWP'], verbose=1)
    stop('naiveBayes')

    start('SVM')
    svm.clasify(booksData['genres'], booksData['summariesWP'], verbose=1)
    stop('SVM')

    start('LSTM')
    dl.clasify(booksData['genres'], booksData['summaries'],  booksData['wordIndex'], verbose=3)
    stop('LSTM')

    print('\n--------------------------------------------------------------------------')
    prtTimes()
