import numpy
import random
from fractions import Fraction

def loadDataSet():
    dataMatrix = []; labelMatrix = []
    
    for line in open('testSet.txt'):
        lineArr = line.strip().split()
        dataMatrix.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMatrix.append(int(lineArr[2]))
    return dataMatrix, labelMatrix

def sigmod(inX):
    return 1 / (1 + numpy.exp(-inX))

#the arguments of this method is the data matrix that each element is the training sample and label Matrix corresponding to the data Matrix
def gradAscent(dataMatrix, classLabels):
    dataMatrix = numpy.mat(dataMatrix)
    labelMatrix = numpy.mat(classLabels).transpose()
    row, col = numpy.shape(dataMatrix)
    print('row: {0}, col:{1}'.format(row, col))
    alpha = 0.001
    maxCycles = 500
    weights = numpy.ones((col,1))
    for i in range(maxCycles):
        h = sigmod(dataMatrix * weights)
        #print('{0}th iterationh, h is:'.format(i), h)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m, n = numpy.shape(dataMatrix)
    alpha = 0.01
    weights = numpy.ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMat, classLabels, numIter = 150):
    m, n = numpy.shape(dataMat)
    weights = numpy.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #the errorRate is 36% when constant is 0.01, it's 33% when constant is 0.02, it's 39 when constant is 0.025
            alpha = 4 / (1 + j + i) + 0.02
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

def multiTest():
    numTest = 10; errorSum = 0
    for i in range(numTest):
        errorSum += colicTest()
    print('after {0} iterations the average error rate is {1}'.format(numTest, errorSum / numTest))

def colicTest():
  
    trainMat = []; trainLabel = []; testMat = []; testLabel = []
    for line in open('horseColicTraining.txt'):
        lineArr = line.strip().split()
        currentline = []
        for i in range(len(lineArr) - 1):
            currentline.append(float(lineArr[i]))
        trainMat.append(currentline)
        trainLabel.append(float(lineArr[-1]))
    for line in open('horseColicTest.txt'):
        lineArr = line.strip().split()
        currentline = []
        for i in range(len(lineArr) - 1):
            currentline.append(float(lineArr[i]))
        testMat.append(currentline)
        testLabel.append(float(lineArr[-1]))
    #print('training data:\n', len(trainMat[0]), '\n', trainLabel[0], '\n')
    #print('test data:\n', testMat, '\n', testLabel, '\n')
    trainweights = stocGradAscent1(numpy.array(trainMat), trainLabel, 450)
    errorcount = 0; numTestVec = 0
    numtestcase = len(testMat)
    for testIndex in range(numtestcase):
        if classifyVector(numpy.array(testMat[testIndex]), trainweights) != testLabel[testIndex]:
            errorcount += 1
    errorRate = Fraction(errorcount, numtestcase)
    print('the error rate of this test is {0}, {1}'.format(errorcount/numtestcase, errorRate))
    return trainweights,errorRate

def classifyVector(inX, weight):
    prob = sigmod(sum(inX * weight))
    if prob > 0.5: return 1
    else: return 0

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataMat = []; labelMat = []
    for line in open('horseColicTraining.txt'):
        lineArr = line.strip().split()
        currentline = []
        for i in range(len(lineArr) - 1):
            currentline.append(float(lineArr[i]))
        dataMat.append(currentline)  
        labelMat.append(float(lineArr[-1]))
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcode1 = []; ycode1 = []
    xcode2 = []; ycode2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcode1.append(dataArr[i, 1]); ycode1.append(dataArr[i, 2])
        else:
            xcode2.append(dataArr[i, 1]); ycode2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcode1, ycode1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcode2, ycode2, s = 30, c = 'green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
