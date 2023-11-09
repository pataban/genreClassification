import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from constants import *


def clasify(data):
    labels=list(data['genre'].map(lambda genre:SELECTED_GENRES.index(genre)))
    features = list(data['summary'])

    train, test, train_lables, test_labels = train_test_split(features,labels, test_size = 0.33, random_state=42)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    svc_model = SVC(C=1.0, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, 
        kernel='linear',tol=0.001)
    svc_model.fit(train, train_lables)

    y_predict = svc_model.predict(test)
    print(y_predict)
    print(accuracy_score(test_labels, y_predict))





