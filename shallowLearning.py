from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from constants import *


NB = 'naiveBayes'
SVM = 'SVM'


def buildNB():
    return GaussianNB()


def buildSVM():
    return SVC(C=1.0, coef0=0.0, decision_function_shape='ovr',
               kernel='sigmoid', tol=0.0001, cache_size=4000)


def classify(modelName, booksData, verbose=0):
    print('\n--------------------------------------------------------------------------')
    print(modelName)
    print('--------------------------------------------------------------------------')

    xTrain, xTest = booksData['summariesWP']
    yTrain, yTest = booksData['genres']

    if modelName == NB:
        model = buildNB()
    elif modelName == SVM:
        model = buildSVM()
    else:
        return

    if verbose > 0:
        print('training...')
    model.fit(xTrain, yTrain)

    print("evaluating...")
    predictedLabels = model.predict(xTest)
    if verbose > 1:
        print('predictedLabels:\n', predictedLabels)

    print('accuracy: %.4f' % (accuracy_score(yTest, predictedLabels)))
