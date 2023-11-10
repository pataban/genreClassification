from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from constants import *


def clasify(data):
    labels = data['genre']
    features = list(data['summary'])

    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
        features, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE)

    gnb = GaussianNB()

    print('training NB')
    gnb.fit(featuresTrain, labelsTrain)

    predictedLabels = gnb.predict(featuresTest)
    print('predictedLabels:\n', predictedLabels)

    print('accuracy: %2f\n' % (accuracy_score(labelsTest, predictedLabels)))
