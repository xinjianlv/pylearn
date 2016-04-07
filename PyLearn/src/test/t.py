#! coding: cp936
from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

fracs = [45, 30, 25]             #ÿһ��ռ�ñ������ܺ�Ϊ100
explode=(0, 0, 0.08)             #�뿪����ľ��룬��Ч��
labels = 'Hogs', 'Dogs', 'Logs'  #��Ӧÿһ��ı�־

pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90, colors = ("g", "r", "y"))
                                 # startangle�ǿ�ʼ�ĽǶȣ�Ĭ��Ϊ0�������￪ʼ����ʱ�뷽������չ��

title('Raining Hogs and Dogs')   #����

show()