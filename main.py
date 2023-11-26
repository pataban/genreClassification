import svm
import naiveBayes as nb
import deepLearning as dl
import dataPreparation as dp


if __name__ == "__main__":
    booksData = dp.cleanData(saveFileSuffix='')
    # booksData = dp.loadData('')
    # booksData.info()
    # print('\n', booksData, '\n')
    nb.clasify(booksData['SL'])
    svm.clasify(booksData['SL'])
    dl.clasify(booksData['DL'])
