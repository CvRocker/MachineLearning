import numpy
import os
import random
from functools import reduce

PRIOR = 'prior'
VECT = 'vect'
DENOM = 'demon'

#return the splited documents and corresponding label
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

#create a list that contain the all distinctive words in total documents
def createVocabularyList(dataSet):
    vocaSet = set()
    for document in dataSet:
        vocaSet = vocaSet | set(document)
    return list(vocaSet)

#return the input document's binary vector of vocabulary that the position is 1 if the word occur or 0 
def setOfWords2Vec(vocaList, inputSet):
    returnVec = [0] * len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] = 1
        else:
            print('the word {0} is not in my vocabulary !'.format(word))
    return returnVec

def bagOfWords2Vec(vocaList, inputSet):
    returnVec = [0] * len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] += 1
        else:
            print('the word {0} is not in my vocabulary !'.format(word))
    return returnVec
#return a dict that contain the each class which is also a dict contain three elements 
#are class's Vector, denominator of class and prior probability
def trainNB0(trainMatrix, trainCategory):
    priors = {}
    numTrainDocs = len(trainCategory)
    numwords = len(trainMatrix[0])

    for i in range(numTrainDocs):
        label = trainCategory[i]
        priors[label] = priors.get(label, 0) + 1
        #priors[label] = 1 if label in priors else priors[label] = priors.get(label) + 1
    for key in priors:
        priors[key] = priors.get(key, 0) / numTrainDocs
        cateI = {}
        cateI[PRIOR] = priors[key]
        priors[key] = cateI

    #print('before initiate:\n', priors)
    for i in range(numTrainDocs):
        key = trainCategory[i]
        label = priors.get(key, None)
        #print('before compute label:\n', label)
        if not label: continue
        label[VECT] = (label.get(VECT, numpy.ones(numwords)) + trainMatrix[i])
        label[DENOM] = (label.get(DENOM, 2) + sum(trainMatrix[i]))
        priors[key] = label
        #print('after compute label', label)
    #print('after initiate:\n', priors)
    for key in priors:
        label = priors[key]
        #avoid the misstake of float or out of down-brim
        label[VECT] = numpy.log(label[VECT] / label[DENOM])
        priors[key] = label
    #print('after compute probability:\n', priors)
    return priors

#the arguments are binary target vector and necessary probabilities.
def classifyNB(targetVec, priors):
    #print('targetVec: ', targetVec)
    labels = {}
    for key in priors:
        label = priors[key]
        probability = sum(targetVec * label[VECT]) + numpy.log(label[PRIOR])
        #print(key, ' probability: ', probability)
        labels[key] = probability
    maxPro = reduce(max, labels.values())
    #print('maxPro: ', maxPro)
    resultLabel = ''
    for (key, value) in labels.items():
        if value == maxPro:
            resultLabel = key
            break
    return resultLabel

def testingNB():
    postingList, classVec = loadDataSet()
    vocabulary = createVocabularyList(postingList)
    trainMatrix = []
    for posting in postingList:
        trainMatrix.append(setOfWords2Vec(vocabulary, posting))
    priors = trainNB0(numpy.array(trainMatrix), numpy.array(classVec))
    testEntry = ['love', 'my', 'dalmation']
    thisDocVec = numpy.array(setOfWords2Vec(vocabulary, testEntry))
    print(testEntry, ' classfied as:\n', classifyNB(thisDocVec, priors))
    testEntry = ['stupid', 'garbage']
    thisDocVec = numpy.array(setOfWords2Vec(vocabulary, testEntry))
    print(testEntry, ' classfied as:\n', classifyNB(thisDocVec, priors))
    

def textParse(bigString):
    import re
    listOfToken = re.split(r'[^\w*]', bigString)
    return [token.lower() for token in listOfToken if len(token) > 2]

def spamTest(datapath):
    filelist = os.listdir(datapath)
    classList =[]; doclist = []; fullText = []
    i = 0
    for filename in filelist:
        #print(i)
        datafiles = os.listdir(datapath+'/'+filename)
        for datafile in datafiles:
            #print('filename: ', datapath+'/'+filename+'/'+datafile)
            wordlist = textParse(open(datapath+'/'+filename+'/'+datafile, 'rt').read())
            doclist.append(wordlist)
            fullText.extend(wordlist)
            classList.append(i)
        i += 1
    print('classes: ', classList)
    #print(doclist)
    vocabulary = createVocabularyList(doclist)
    trainSet = list(range(len(doclist)))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMatrix = []; trainClass = []
    for trainIndex in trainSet:
        trainMatrix.append(setOfWords2Vec(vocabulary, doclist[trainIndex]))
        trainClass.append(classList[trainIndex])
    priors = trainNB0(numpy.array(trainMatrix), numpy.array(trainClass))
    errorcount = 0
    for testIndex in testSet:
        wordVect = setOfWords2Vec(vocabulary, doclist[testIndex])
        if classifyNB(numpy.array(wordVect), priors) != classList[testIndex]:
            errorcount += 1
    print('the error rate is: ', errorcount / len(testSet))
