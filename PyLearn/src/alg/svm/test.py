# -*- coding: utf-8 -*-
'''
Created on 2016年2月29日

@author: nocml
'''
import svmMLiA

root = "..\\..\\svm\\"

dataArr , labelArr = svmMLiA.loadDataSet(root + "testSet.txt")

b , alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
# b , alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

print "b" , ":" , b
print "=================="
 
for i in range(100):
    if alphas[i] > 0.0 :
        print dataArr[i] , labelArr[i]


# svmMLiA.testRbf();