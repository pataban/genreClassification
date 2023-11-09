from constants import *
import pandas as pd
from pandas import DataFrame
import json
import copy
import string
import copy
import csv


def loadData(type)->DataFrame:
    books=pd.read_json(DATA_FIE_PATH+DATA_FILE_NAME+type+'.json')
    return books


def cleanData()->DataFrame:
    books=pd.read_csv(DATA_FIE_PATH+RAW_DATA_FILE_NAME,sep='\t')
    books.columns=('wId','fId','title','author','date','genres','summary')

    books=books.drop(columns=['fId','date','author']).dropna()

    books['genres']=books['genres'].map(lambda genres:list(json.loads(genres).values()))


    genreCount={}
    for book in books['genres']:
        for genre in copy.copy(book):
            if genre not in SELECTED_GENRES:
                book.remove(genre)
            else:
                if genre in genreCount:
                    genreCount[genre]+=1
                else:
                    genreCount[genre]=1
    genreCount=list(map(lambda k,v:(k,v),genreCount.keys(),genreCount.values()))
    genreCount.sort(key=lambda t:t[1],reverse=True)
    #print(genreCount)


    books['genres']=books['genres'].map(lambda genres:genres[0] if len(genres)==1 else None)
    books=books.dropna()

    books.columns=('id','title','genre','summary')
    #books.info()
    #print(books)


    books['summary']=books['summary'].map(unifySummary)
    books=books.dropna()
    #print(books)

    vocabulary=getVocabulary(books)
    books['summary']=books['summary'].map(lambda sum,v=vocabulary:countWords(sum,v))

    books.to_json(DATA_FIE_PATH+DATA_FILE_NAME+'.json')
    return books


def unifySummary(bookSummary):
    summary=""
    for c in bookSummary:
        if c.isalpha():
            summary+=c.lower()
        elif c in string.whitespace and summary!='' and summary[-1]!=' ':
            summary+=' '
    summary=summary.split()
    return summary if len(summary)>=SUMMARY_MIN_LENGTH else None


def getVocabulary(books):
    vocabulary={}
    for summary in books['summary']:
        for word in summary:
            vocabulary[word]=0
    return vocabulary


def countWords(bookSummary,vocabulary):
    vocabulary=copy.deepcopy(vocabulary)
    for word in bookSummary:
        vocabulary[word]+=1
    return list(vocabulary.values())