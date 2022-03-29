from __future__ import division, unicode_literals
import math

from main import VectorSpace

n_contain = []

def n_containing(word, docs_list):
    '''calculate the number of docs with this word.'''

    sum = 0
    for doc in docs_list:
        if doc[word] != 0: 
            sum += 1
    return sum

def idf(word, docs_list):
    return math.log(len(docs_list)/ (1 + n_contain[word]))

def tfidf(tfVec, idfVec):   
    tfIdfVec = [tfVec[i]*idfVec[i] for i in range(len(tfVec))]
    return  tfIdfVec

########################################################33
