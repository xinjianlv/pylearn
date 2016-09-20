'''
Created on 2016年9月19日

@author: nocml
'''
import codecs
import copy
import re  
from numpy.core.defchararray import lstrip
import numpy as np


def load(filepath):
    dic = {}
    f =  codecs.open(filepath, encoding='GBK')
    while True:  
        line = f.readline()  
        if line:  
            line = line[21:]
            # Using regular expressions  
            line =  re.sub("/[a-z]+", "", line)
            terms= str(line).split(" ")
            for t in terms:
                c = 1
                if  t in dic.keys():
                    c = dic[t] + 1
                dic[t] = c
        else:  
            break
    f.close()
    return dic


def process (dic):
#     maxv = max(dic.values())
    keys = dic.keys()
    lst = []
    for k in keys:
        v = dic[k]
        if v > 10:
            lst.append(k)
    return lst
 
def getdic(filepath):
    dic = load(filepath)
    lst = process(dic)
    dic2 = {}
    ndx = 0
    for k in lst:
        dic2[k] = ndx
        ndx += 1
    return dic2

def loadData(filepath):
    f = codecs.open(filepath, encoding='GBK')
    lst = []
    while True:
        line = f.readline()
        if line:
            line = line[21:]
            line = re.sub("/[a-z]+", "", line)
            lines = re.split('|，|。|；|！|', line)
#             lines = re.split('。', line)
            for line in lines:
                terms= line.split(" ")
                lstTemp = []
                for t in terms:
                    lstTemp.append(t.strip())
                lst2 = list(filter(lambda x :len(x) > 0 , lstTemp))
                lst.append(lst2)
        else:
            break
    return lst

def int2OneHot(size):
    return enc.transform([[58]]).toarray()
    
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()
numberic = []
for i in range(100):
    numberic.append([i])
enc.fit(numberic)

# print(enc.transform([1]).toarray())

arr = np.zeros(100)
arr[0] = 1

# loadData('/Users/nocml/Documents/workspace/python/Tensorflow/com/mylearn/myrnn/data/people.txt')


