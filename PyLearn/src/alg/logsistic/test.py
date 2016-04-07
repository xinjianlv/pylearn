# -*- coding: utf-8 -*-
'''
Created on 2016年2月26日

@author: nocml
'''
from numpy import *

import logRegres

dataArr , labelMat =  logRegres.loadDataSet()
print dataArr
# weights = logRegres.gradAscent(dataArr, labelMat)
weights = logRegres.stocGradAscent1(array(dataArr), labelMat , 150)
# print "weights:"
print weights
# weights = [9.90028796735921,1.4181704748685218,    -1.3358509819647089     ]
# weights = [10.373488441795256,    0.7810704644295239 ,   -1.5443579566870218   ]
logRegres.plotBestFit(array(weights))

