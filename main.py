import pandas as pd
from constants import *
import dataPreparation as dp

if __name__ == "__main__":
    books1=dp.cleanData()
    books1.info()
    print(books1)
    books2=dp.loadData()
    books2.info()
    print(books2)
