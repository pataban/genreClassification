from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from constants import *


def clasify(labels, features, verbose):
    print('\n--------------------------------------------------------------------------')
    print('SVM')
    print('--------------------------------------------------------------------------')

    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
        features, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(featuresTrain)
    featuresTrain = scaler.transform(featuresTrain)
    featuresTest = scaler.transform(featuresTest)

    svm = SVC(C=1.0, coef0=0.0, decision_function_shape='ovr',
              degree=5, kernel='sigmoid', tol=0.0001)

    if verbose>0:
        print('training')
    svm.fit(featuresTrain, labelsTrain)

    predictedLabels = svm.predict(featuresTest)
    if verbose>1:
        print('predictedLabels:\n', predictedLabels)

    print('accuracy: %2f' % (accuracy_score(labelsTest, predictedLabels)))
