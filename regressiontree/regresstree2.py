import numpy as np
import matplotlib.pyplot as plt

def loaddataset(filename, separator='\t'):
    data = []
    for line in open(filename, 'rt'):
        content = line.strip().split(separator)
        content = map(float, content)
        data.append(content)
    return data

def splitdataset(dataset, feature, value):
    gtmat = dataset[np.nonzero(dataset[:,feature] > value)[0],:][0]
    ltmat = dataset[np.nonzero(dataset[:,feature] <= value)[0],:][0]
    return gtmat, ltmat

def createTree(data, leafType):
    return
