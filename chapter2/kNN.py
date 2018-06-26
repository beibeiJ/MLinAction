from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


# inX is input vector; dataSet is training set; k is the number of selected nearest neighbors
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]     # .shape return a vector and get the first one. it is the number of row

	# get the distance (each row of dataSet is a class)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile([0,0], (5,1)) means repeat [0,0] 5 times in row and 1 time in column

    sqDiffMat = diffMat**2 # **2 square of each element
    sqDistances = sqDiffMat.sum(axis=1)    # axis＝0 works on each column (summarize all row elements)，axis＝1 works on each row (summarize all column elements)
    distances = sqDistances**0.5  # **0.5 square root of each element
    sortedDistIndicies = distances.argsort()   # argsort returns the indices that would sort an array
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



