import numpy
import math
import operator

def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels

#axis is the target column that will be extracted!
def splitData(dataset, axis):
    attrLists = {}
    for featureV in dataset:
        attr = featureV[axis]
        extractV = featureV[:axis]
        extractV.extend(featureV[axis+1:])
        if not attr in attrLists.keys():
            attrLists[attr] = []
            attrLists[attr].append(extractV)
        else:
            attrLists[attr].append(extractV)
    return attrLists

#create a decision tree recursively
def createDecisionTree(dataset, labels):
    print('-' * 80)
    classList = [record[-1] for record in dataset]
    print('current dataset: ', dataset)
    print('current labels: ', labels)
    print('current classList: ', classList)
    # if All elements in the current dataset has concurrent class, this will be a leave node!
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #the the element of feature in dataset only contains two([attr1, target])
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataset)
    print('best feature is: ', bestFeature)
    bestFeatureLabel = labels[bestFeature]
    #After choosing the best feature, constructing a node
    MyTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    bestFeaSubset = splitData(dataset, bestFeature)
    for key in bestFeaSubset:
        print(key, ' => ', bestFeaSubset[key])
    for key in bestFeaSubset.keys():
        subset = bestFeaSubset[key]
        subLabels = labels[:]
        MyTree[bestFeatureLabel][key] = createDecisionTree(subset, subLabels)
    
    return MyTree

#compute all attributes' Gain in the current dataset
#the compute methond is information gain, using entropy!
def chooseBestFeature(dataset):
    #calculate the currently dataset's entropy 
    entropyDS = calculateEntropy(dataset)
    #print("The entropy of dataset", entropyDS)
    bestFeature = 0; bestGain = 0
    datasetLen = len(dataset)
    featureLen = len(dataset[0]) - 1
    #print('The feature\'len is: {0} and dataset\'len is: {1}'.format(featureLen, datasetLen))
    for i in range(featureLen):
        i_datas = splitData(dataset, i)
        #calculate the entropy that use the feature i as the spliting feature
        entropys_i = 0.0
        for key in i_datas.keys():
            value = i_datas[key]
            valueLen = len(value)
            weight = valueLen/datasetLen
            k_entropy = calculateEntropy(value)
            entropys_i += weight * k_entropy
        #calculate the information gain of feature i
        newGain = entropyDS - entropys_i
        if bestGain < newGain:
           bestGain = newGain
           bestFeature = i
        #print('Currently the best Feature is: {0} and best Gain is: {1}'.format(bestFeature, bestGain))
    return bestFeature

#compute the main class of the current node
def majorityCnt(classlist):
    classCount = {}
    for key in classlist:
        classCount[key] = classCount.get(key, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

#shannon entropy
#the format fo dataset like '[attrA1, attrB1, classification] '
def calculateEntropy(dataset):
    datasetLen = len(dataset)
    labels = {}
    for featureVect in dataset:
        label = featureVect[-1]
        if not label in labels.keys():
            labels[label] = 1
        else :
            labels[label] += 1
    entropy = 0
    for key in labels.keys():
        probK = labels[key] / datasetLen
        entropy -= probK * math.log(probK, 2)
    return entropy

def classify(myTree, featurelabel, testVect):
    print('-' * 80)
    print(featurelabel)
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    featIndex = featurelabel.index(firstStr)
    
    for key in secondDict.keys():
        if testVect[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], featurelabel, testVect)
            else:
                classLabel = secondDict[key]
    return classLabel
