import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from constants import *


def clasify(data):
    
    # Organize data:
    label_names = SELECTED_GENRES
    labels = list(data['genre'].map(lambda genre:SELECTED_GENRES.index(genre)))
    feature_names = ['summary']#list(map(str,range(0,SUMMARY_MIN_LENGTH)))
    features = list(data['summary'])

    # Print data:
    #print(label_names)
    #print('Class label = ', labels[0])
    #print(feature_names)
    #print(features[0])

    # Split dataset into random train and test subsets:
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
    #print(test_labels)
    #print(test)

    # Initialize classifier:
    gnb = GaussianNB()

    # Train the classifier:
    model = gnb.fit(train, train_labels)
    # Make predictions with the classifier:
    predictive_labels = gnb.predict(test)
    print(predictive_labels)

    # Evaluate label (subsets) accuracy:
    print(accuracy_score(test_labels, predictive_labels))