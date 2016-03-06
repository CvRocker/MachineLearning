import numpy as np

def loadsimpledata(filename, separator='\t'):
    X = []; y = []
    for line in open(filename, 'rt'):
        content = line.strip().split(separator)
        X.extend([float(elem) for elem in content[:-1]])
        y.append(float(content[-1]))
    return X, y

def CART_regression(X,y):
    print('Enter!')
    print('X: {0}\ty:{1}'.format(X,y))
    tree = dict()
    xMat = np.mat(X).T; yMat = np.mat(y).T
    s,j = bestSplit(xMat, yMat)
    print('ind: ', j)
    
    R1, R2, c1, c2 = computeParams(xMat, yMat, s)
    y1 = R1; y2 = R2
    tree['node'] = dict(s=s)
    tree['left'] = dict(c1=c1)
    tree['right'] = dict(c2=c2)
    #print('R1: {0}\tR2: {1}'.format(R1, R2))
 
    position = j + 1; R1 = xMat[xMat <= s]; R2 = xMat[xMat > s]
    #print('ind: {0} y: {1}'.format(position, yMat))
    #print('y1: {0}\ny2: {1}'.format(yMat[:position,0], yMat[position:,0]))
    if R1.tolist() and len(R1.T) != 1:
        tree['left']['node'] = CART_regression(R1, y1)
    print('***************************')
    if R2.tolist() and len(R2.T) != 1:
        tree['right']['node'] = CART_regression(R2, y2)
    return tree
       

def bestSplit(X,y):
    m = len(X)
    mins = np.inf; s = 0
    for j in range(m):
        st = X[j,:]
        R1, R2, c1, c2 = computeParams(X, y, st)
        
        mintemp = optimaLeastSquare(R1.A, c1, R2.A, c2)
        if mintemp < mins:
            mins = mintemp
            s = st;
            positionJ = j
        print('s: {0}, j: {1}, min: {2}'.format(st, j, mintemp))
    return s, positionJ

def computeParams(X, y, s):
    R2 = y[X > s]
    R1 = y[X <= s]
    c1 = 0; c2 = 0
    if R1.tolist():
        c1 = np.mean(R1)
    if R2.tolist():
        c2 = np.mean(R2)
    return R1, R2, c1, c2

def optimaLeastSquare(y1, c1, y2, c2):
    #print('y1: {0}\ny2: {1}'.format(y1, y2))
    J = np.sum((y1 - c1)**2) + np.sum((y2 - c2)**2)
    return J
