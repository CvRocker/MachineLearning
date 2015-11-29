import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
    xytext=centerPt, textcoords='axes fraction', va='center',\
    bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getLeafNumber(myTree)
    depth = getTreeDepth(myTree) - 1
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + numLeafs)/2/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD

 
def getLeafNumber(DecisionTree):
    print('-' * 80)
    leafcount = 0
    #root = DecisionTree.keys()[0]
    #secondNode = DecisionTree[root]
    for key in DecisionTree.keys():
        values = DecisionTree[key]
        #print(key, ' type: ', type(values).__name__)
        if type(values).__name__  == 'dict':
            leafcount += getLeafNumber(values)
        else:
            leafcount += 1
    return leafcount

def getTreeDepth(DecisionTree):
    print('-' * 80)
    maxDepth = 0
    for key in DecisionTree.keys():
        values = DecisionTree[key]
        #print(key, ' type: ', type(values).__name__)
        if type(values) == dict:
            thisDepth = 1 + getTreeDepth(values)
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth   

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getLeafNumber(inTree))
    plotTree.totalD = float(getTreeDepth(inTree)) - 1
    print('Total depth:' ,plotTree.totalD)
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    #plotNode('decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    #plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
