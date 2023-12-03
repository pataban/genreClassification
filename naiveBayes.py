from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from constants import *


def clasify(labels, features, verbose):
    print('\n--------------------------------------------------------------------------')
    print('naiveBayes')
    print('--------------------------------------------------------------------------')

    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
        features, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE)

    gnb = GaussianNB()

    if verbose > 0:
        print('training')
    gnb.fit(featuresTrain, labelsTrain)

    predictedLabels = gnb.predict(featuresTest)
    if verbose > 1:
        print('predictedLabels:\n', predictedLabels)

    print('accuracy: %2f' % (accuracy_score(labelsTest, predictedLabels)))
