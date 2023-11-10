import svm
import naiveBayes as nb
import dataPreparation as dp


if __name__ == "__main__":
    books = dp.cleanData(saveFileSuffix='')
    # books = dp.loadData('')
    books.info()
    print('\n', books, '\n')
    nb.clasify(books)
    svm.clasify(books)
