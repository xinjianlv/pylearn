#-*- coding:utf-8 -*-
'''
Created on 2016年2月23日

@author: nocml
'''
class MyTrie(object):
    def __init__(self):
        self.data = {}
        self.tag = []
    def init(self , info):
        pass
    def addWord(self , word):
        if word == None :
            return ;
        if len(word) < 1:
            return ;
        begin = word[0]
        suffix = "".join(word[1:len(word)])
        subtrie = MyTrie()
        if begin in self.data:
            subtrie = self.data.get(begin)
            if len(suffix) > 0 :
                subtrie.addWord(suffix)
                self.data[begin] = subtrie
            else :
                self.tag.append(begin)
             
        else :
            if len(suffix) > 0:
                subtrie.addWord(suffix)
            else :
                self.tag.append(begin)
            self.data[begin] = subtrie

    def detect(self , line):
        ndx = 0
        termlist = []
        while len(line) > 0:
            ch = line[0]
#             print ch
            if ch in self.data:
                ndx = self.detectTerm(line)
                if ndx > 0 :
                    termlist.append(line[0:ndx])
                if ndx < len(line):
                    line = line[ndx : len(line)]
                    #当使用多字节表示汉字时，可能会匹配到汉字的第一个字节，所以42行的判断会成立，
                    #但是找不到匹配项，因此返回的ndx为0，这里要判断这种情况，当出现这种情况时，ndx自加1
                    #否则会进入死循环
                    if ndx == 0:
                        line = line[1 : len(line)]
                else:
                    break;
            else:
                if 0 < len(line):
                    line = line[1 : len(line)]
        return termlist
    def detectTerm(self , line):
        end = 0
        count = 0;
        trie = self
        for ch in line :
            if ch in trie.data :
                count += 1
                if ch in trie.tag : 
                    end = count
                trie = trie.data[ch]
            else :
                return end
        return end
    def printtrie(self):
        terms = []
        if  len(self.data) > 0 :
            termlist = []
            for ch in self.data:
                termlist.append(ch)
                trie = self.data[ch];
                if len(trie.data) > 0 :
                    l = trie.printtrie()
                    for t in l:
                        termlist.append(t)
                    
            for t in termlist:
                terms.append(t)
        return terms
trie = MyTrie()
trie.addWord("中国");
trie.addWord("哈哈");
trie.addWord("哈哈中国人民中");
termlist = trie.detect("哈哈中国人民中国中国哈哈")
for term in termlist:
    print term