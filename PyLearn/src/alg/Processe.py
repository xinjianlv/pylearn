# -*- coding: utf-8 -*- 
'''
Created on 

@author: nocml
'''
import re

class Process():
    def data2csv(self , infile):
        data = []
        with open(infile) as ifile:
            for line in ifile:
                data.append(self.line2csv(line))
                data.append("\n")
    def line2csv(self , line):
        strlist = []
        dictlist = []
        for term in line.split(","):
            str = "".join(term)
            terminfo = self.split(str)
            dictlist.append(terminfo)
        for dt in dictlist:
            for key in dt.keys():
                strlist.append(key)
                strlist.append(",")
                strlist.append(dt[key])
                strlist.append(",")
        return "".join(strlist).strip(",")

    '''
    '''
    def split(self , term):
        str = ""
        str = str.join(re.findall(r"[0-9]+斤", term, 0))
        tokens = {}
        tokens[term.replace(str , "")] = str
        return tokens

# print re.findall(r"[0-9]+��", "����320��   ", 0)
inst = Process()
s = inst.line2csv("哈哈32斤,没有3456斤")
print s