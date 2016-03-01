import numpy as np
import matplotlib.pyplot as plt

def loadsimpleData():
    data = np.matrix([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]], dtype=float)
    marker = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, marker

def plotdata(data,label):
    plt.figure(1)
    ax1 = plt.subplot(111)
    data = data.tolist()
    #print(data)
    class1 = []; class2 = []
    length = len(label)
    for i in range(length):
        if label[i] == 1.0:
            class1.append(data[i])
        else:
            class2.append(data[i])
    #print(type(class1), '\n', class1)
    class1 = np.mat(class1)
    #print(type(class2), '\n', class2)
    class2 = np.mat(class2)
    plt.scatter(class1[:,0], class1[:,1], c='green')
    plt.scatter(class2[:,0], class2[:,1], marker='x', c='red')
    plt.title('simple data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def stumpClassify(data, dim, threshVal, threshIneq):
    result = np.ones((np.shape(data)[0],1))
    #print('result:\n', result)
    if threshIneq == 'lt':
        result[data[:,dim] <= threshVal] = -1.0
    else:
        result[data[:,dim] > threshVal] = -1.0
    return result

def constructStump(data, label, D):
    data = np.mat(data); label = np.mat(label).T
    m,n = np.shape(data)
    numsteps = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        colMin = data[:,i].min(); colMax = data[:,i].max()
        stepsize = (colMax - colMin)/numsteps
        for j in range(-1, int(numsteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (colMin + float(j) * stepsize)
                predictions = stumpClassify(data, i, threshVal, inequal)
                errors = np.mat(np.ones((m,1)))
                #print('label: \n', label, '\npredictions:\n', predictions)
                errors[predictions == label] = 0
                weightedError = D.T * errors
                #print('split: dim:{0}, thresh:{1}, thresh inequal: {2}, weighted error: {3}'.format(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictions.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, bestClassEst, minError

#此处使用的若分类器是单层决策树，还可以使用其的若分类器
def adaboostTrain(data, label, numIter=40):
    weekClassifiers = []
    m = np.shape(data)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIter):
        bestStump,classEst,error = constructStump(data, label, D)
        print('D: ', D.T, '\nerror: ', error)
        alpha = 0.5 * np.log((1-error)/max(error,1e-16))[0,0]
        bestStump['alpha'] = alpha
        weekClassifiers.append(bestStump)
        #print('classEst:', classEst)
        #print('label:', -1*alpha*np.mat(label).T)
        expon = np.exp(np.multiply(-1*alpha*np.mat(label).T, classEst))
        z = D.T * expon
        D = np.multiply(D,expon)/z
        aggClassEst += alpha*classEst
        print('aggClassEst: ', aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(label).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m 
        print('total train error: {0}\n'.format(errorRate))
        if errorRate == 0: break
    return weekClassifiers

def adaboostClassify(weekClassifiers, data):
    data = np.mat(data)
    m = np.shape(data)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for classifier in weekClassifiers:
        aggClassEst += classifier['alpha'] * stumpClassify(data, classifier['dim'], classifier['thresh'], classifier['ineq'])
    #errorRate = np.mat(np.zeros((m,1)))
    return np.sign(aggClassEst)

def testErrorRate(trainfile, testfile, numIter=40):
    traindata,trainlabel = loaddataset(trainfile)
    testdata,testlabel = loaddataset(testfile)
    weekClassifiers = adaboostTrain(traindata,trainlabel,numIter)
    predictions = adaboostClassify(weekClassifiers,testdata)
    error = np.mat(np.ones((len(testlabel),1)))
    error[predictions == np.mat(testlabel).T] = 0
    return error.sum()/len(testlabel)

def loaddataset(filename, separator='\t'):
    data = []; label = []
    for line in open(filename):
        content = line.strip().split(separator)
        data.append([float(elem) for elem in content[:-1]])
        label.append(float(content[-1]))
    return data,label
