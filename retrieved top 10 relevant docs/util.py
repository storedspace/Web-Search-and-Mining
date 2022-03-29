import sys
import numpy
import tfidf

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

dictionary = {}


def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return  float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))


def euclidean(vector1, vector2):
    return norm(numpy.asarray(vector1)-numpy.asarray(vector2))


def calDocSize(dictionary, list):
	sum = 0
	for i in range (len(list)):
		sum += dictionary[list[i][0]]
	return sum

def printTemplate(list, label):
    print('---------------------------')
    print(label)
    print('NewsID'+'     '+'Score')
    print('------'+'     '+'------')
    for i in range(len(list)):
        print(list[i][0], end= '  ')
        print('%.6f'%(list[i][1]))
    print('Data Size: ' + str(calDocSize(dictionary, list)))

def check(num,tuple_list):
    my_list = []
    count = 0
    if num == 0:
        my_list = ["News123256","News119356","News111959","News115859","News120265","News119746","News101763","News108578","News107163","News122750"]
    if num == 1:
        my_list = ["News107883","News108482","News109808","News110033","News110141","News110871","News108024","News108653","News108964","News110211"]
    if num == 2:
        my_list = ["News108813","News104913","News116613","News103134","News116634","News103728","News110804","News121995","News118108","News103767"]
    if num == 3:
        my_list = ["News107883","News110329","News110871","News108482","News105142","News110514","News110033","News110804","News110141","News111579"]
    for newsid in tuple_list:
        if newsid[0] in my_list:
            count+=1
    return count