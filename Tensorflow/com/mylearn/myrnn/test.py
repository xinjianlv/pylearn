'''
Created on 2016年9月19日

@author: nocml
'''
lst = []
dic = {}
for i in range(100):
    lst.append(i)
    dic[i] = i * 100
    
lst2 = filter(lambda n: n > 90, lst)
dic2 = dict(filter(lambda k,v: v > 9000 , dic))

# for i in lst2:
#     print(i)
    
for t in dic:
    print(t)