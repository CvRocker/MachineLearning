import trees

datapath = 'lenses.txt'

def testLens(datapath):
    fw = open(datapath)
    labels = fw.readline().strip().split('\t')
    del(labels[len(labels) - 1])
    lensedata = [featVect.strip().split('\t') for featVect in fw.readlines() ]
    print(labels)
    print(lensedata)
    lenseTree = trees.createDecisionTree(lensedata, labels)
    print(lenseTree)
    return lenseTree, labels
