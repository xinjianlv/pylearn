# -*- coding: utf-8 -*-
'''
Created on 2016年2月29日

@author: nocml
'''
from numpy import *


def loadDataSet(filename):
    dataArr = []
    labelArr = []
    with open(filename) as ifile:
        for line in ifile:
            arr = line.split()
            dataArr.append([1.0 , float(arr[0]) , float(arr[1])])
            labelArr.append(float(arr[2]))
    return dataArr , labelArr

def sigmod(intX):
    ex = exp(-intX)
    return 1.0 / (1 + ex)

def logistic(dataArr , labelArr , iter):
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m , n = shape(dataMat)
    weights = ones((n , 1))
    alpha = 0.001
    ndx = 0
    while ndx < iter :
        iterMat =  dataMat * weights
        h = sigmod(iterMat)
        error = (labelMat - h)
        print sum(error)
        weights = weights + alpha * dataMat.transpose() * error
        ndx += 1
    return weights


def plotBestFit(weights , filename):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet(filename)
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


root = "..\\..\\logistic\\"
dataArr , labelArr = loadDataSet(root + "testSet.txt")
weights = logistic(dataArr, labelArr, 500)
print weights
plotBestFit(array(weights) , root + "testSet.txt")
