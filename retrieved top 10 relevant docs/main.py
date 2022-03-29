import encodings
from functools import reduce
from itertools import count
from math import log2
from pprint import pprint
from unicodedata import name
from xml.dom.minidom import Document

from textblob import WordList
from Parser import Parser
import util
import argparse
import os
import tfidf
from collections import Counter
import string
import nltk
import jieba

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Collection of idf
    idfVector = []
    
    #Collection of document term vectors with idf
    tfidfDocumentVectors = []

    #Tidies terms
    parser=None

    #methods name
    methodsName = ['TF-Weighting + Cosine Similarity:', 'TF-Weighting + Euclidean Distance:', 'TF-IDF Weighting + Cosine Similarity:' ,'TF-IDF Weighting + Euclidean Distance:','Feedback Queries + TF-IDF Weighting + Cosine Similarity:']
    
    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.dic = []
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]
        #print(self.vectorKeywordIndex)
        # print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
       
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector

    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query

    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings[0:10]


    def search1(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        #  tfidf 
        for word in self.vectorKeywordIndex.values():
            tfidf.n_contain.append(tfidf.n_containing(word, self.documentVectors))
        
        self.idfVector = [tfidf.idf(word, self.documentVectors)  for word in self.vectorKeywordIndex.values()]
        self.tfidfDocumentVectors = [tfidf.tfidf(docV, self.idfVector)for docV in self.documentVectors]
        tfidfQuery = [self.idfVector[i]*queryVector[i] for i in range(len(queryVector))]

        rating1 = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        rating2 = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        rating3 = [util.cosine(tfidfQuery, tfidfVector) for tfidfVector in self.tfidfDocumentVectors]
        rating4 = [util.euclidean(tfidfQuery, tfidfVector) for tfidfVector in self.tfidfDocumentVectors]
        ratingLists = [rating1, rating2, rating3, rating4]

        for i in range(len(ratingLists)):
            ratingsWithName = list(zip(names, ratingLists[i]))
            if(i%2 == 0):
                ratingsWithName.sort(key = lambda x :x[1], reverse=True)
            else:
                ratingsWithName.sort(key = lambda x :x[1], reverse=False)
            ratingLists[i] = ratingsWithName[0:10]
        
        return ratingLists

    def search2(self, searchList, ratingList):
        queryVector = self.buildQueryVector(searchList)
        newRatingList = self.doNLTK(ratingList[2][0][0],queryVector)
        return newRatingList

    def getNewQuery(self,queryVector, doc):
        tokens = nltk.word_tokenize(doc)
        tagged = nltk.pos_tag(tokens)
        newtokens = []
        for tag in tagged:
            if (tag[1] == 'VB') | (tag[1] == 'NN'):
                newtokens.append(tag[0])     
        newStr = ' '.join(newtokens)
    
        newDocVector = self.makeVector(newStr)
        
        newQueryVector = [queryVector[i] + 0.5* newDocVector[i] for i in range(len(queryVector))]
        # print(len(newQueryVector))
        return newQueryVector
    
    def doNLTK(self, fileName, originalQuery):
        path = './EnglishNews/'+ fileName + '.txt'
        with open(path, 'r', encoding="utf-8") as testF:
            testF1 = testF.read().translate(str.maketrans('', '', string.punctuation))
            testF1 = testF1.replace('\n', ' ')
            newQueryVector = self.getNewQuery(originalQuery,testF1)
        tfidfNewQuery = [self.idfVector[i]*newQueryVector[i] for i in range(len(newQueryVector))]
        rating = [util.cosine(tfidfNewQuery, tfidfVector) for tfidfVector in self.tfidfDocumentVectors]
        ratingsWithName = list(zip(names, rating))
        ratingsWithName.sort(key = lambda x :x[1], reverse=True)

        return ratingsWithName[0:10]

names = []

if __name__ == '__main__':
    
    documents = []
    # query
    parser = argparse.ArgumentParser(description= 'Input a string.')
    parser.add_argument('--query',dest= 'sentence', nargs= '+')
    parser.add_argument('--part',dest= 'version')
    args = parser.parse_args()
   
    # deal with vector space
    for filename in os.listdir("./EnglishNews"):
        path = os.path.join("./EnglishNews", filename)
        with open(path, 'r',encoding= 'utf-8') as f:
            f1 = f.read().translate(str.maketrans('', '', string.punctuation))
            f1 = f1.replace('\n', ' ')
            documents.append(f1)
            names.append(filename[0:10])
            util.dictionary[filename[0:10]] = os.stat(path).st_size

    # documents = ["The cat in the hat disabled",
    #              "A cat is a fine pet ponies.",
    #              "Dogs and cats make good pets.",
    #              "I haven't got a hat.",
    #              "cat cat cat"]
 
   

    vectorSpace = VectorSpace(documents)
    if(args.version == '1'):
        ratingList = vectorSpace.search1(args.sentence)
        for i in range(len(ratingList)):
            util.printTemplate(ratingList[i], vectorSpace.methodsName[i])
            # print(util.check(i, ratingList[i]))
    elif(args.version == '2'):    
        ratingList = vectorSpace.search1(args.sentence)
        newRatingList = vectorSpace.search2(args.sentence, ratingList)
        util.printTemplate(newRatingList,vectorSpace.methodsName[4])
        # print(util.check(2,newRatingList))
    
    
   



###################################################
