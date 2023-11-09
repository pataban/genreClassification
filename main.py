import pandas as pd
from constants import *
import dataPreparation as dp
import seaborn  as sns
import matplotlib.pyplot as plt
import naiveBayes as nb
import svm


if __name__ == "__main__":
    #books=dp.cleanData()
    books=dp.loadData('')
    #books.info()
    #print(books)
    nb.clasify(books)
    svm.clasify(books)
