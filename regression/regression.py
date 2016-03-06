import numpy as np
import matplotlib.pyplot as plt

def loaddataset(filename, separator='\t'):
    X = []; y = []
    for line in open(filename, 'rt'):
        content = line.strip().split(separator)
        X.append([float(elem) for elem in content[:-1]])
        y.append(float(content[-1]))
    return X, y

def normalequation(X, y):
    X = np.mat(X); y = np.mat(y).T
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print('this matrix is singular, cannot do inverse!')
        return
    theta = xTx.I * X.T * y
    return theta

#局部加权线性回归(Local weighted Linear Regression)
def lwlr(testpoint, X, y, k=1.0):
    X = np.mat(X); y = np.mat(y).T
    m = X.shape[0]
    W = np.mat(np.eye(m))
    for j in range(m):
        diff = testpoint - X[j,:]
        W[j,j] = np.exp(diff*diff.T/(-2*(k**2)))
    xTx = X.T * W * X
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse.')
        return
    w = xTx.I * (X.T * W * y)
    return testpoint * w

def testLWLR(test, X, y, k=1.0):
    m = np.shape(test)[0]
    pred = np.zeros(m)
    for i in range(m):
        pred[i] = lwlr(test[i], X, y, k)
    return pred

def stageWise(X, y, eps=0.01, numIter=100):
    X = featureScaling(X)
    y = featureScaling(y)
    #ymean = np.mean(y,0)
    #y = y - ymean
    #y = np.mat(y).T
    m,n = np.shape(X)
    returnW = np.zeros((numIter,n))
    theta = np.zeros((n,1)); thetaTest = theta.copy(); thetaMax = theta.copy()
    #print(theta[0])
    for i in range(numIter):
        print(theta.T)
        lowestErr = np.inf
        for j in range(n):
            for sign in [-1,1]:
               thetaTest = theta.copy()
               thetaTest[j] += eps * sign
               pred = X * thetaTest
               err = squareError(y.A, pred.A)
               if err < lowestErr:
                  lowestErr = err
                  thetaMax = thetaTest
        theta = thetaMax.copy()
        returnW[i,:] = theta.T
    return returnW

def squareError(y, pred):
    m = len(y)
    #y = np.mat(y).T
    #print((y-pred)**2)
    return np.sum((y-pred)**2)/(2*m)

def featureScaling(data):
    data = np.mat(data)
    if np.shape(data)[0] == 1:
        data = data.T
    mean = np.mean(data,0)#data is a matrix, this mean is conputed by data's column
    var = np.var(data,0)#each column's variance of data
    #print('data: {0}\nmean: {1}\nvar: {2}\n'.format(data,mean,var))
    data = (data - mean)/var
    return data

def plotdata_curve(X, y, pred):
    X = np.mat(X); y = np.mat(y).T
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(X[:,1], y[:,0], c='red')
    #pred = X * theta
    plotLWLRCurve(ax,X,pred)
    plt.show()

def plotLWLRCurve(ax, X, pred):
    
    #if flag:
    #    print('Enter if!')
        srtInd = X[:,1].argsort(0)
        xSort = X[srtInd][:,0,:]
        predSort = pred[srtInd]
        ax.plot(xSort[:,1], predSort[:,0])
    #else:
    #    print('Enter else!')
    #    ax.plot(X[:,1], np.mat(pred).T[:,0])

from time import sleep
import json
import urllib

def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)

    

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
