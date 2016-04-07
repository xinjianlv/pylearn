# -*- coding: utf-8 -*- 
'''
Created on 2016年2月25日

@author: nocml
'''
print "hello"

class BinTree(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None
    def add(self , word):
        if word == None:
            return ;
        if self.data == None:
            self.data = word
        if self.data == word :
            return ;
        if self.data > word :
            if self.left == None :
                self.left = BinTree()
            self.left.add(word)
        else :
            if self.right == None :
                self.right = BinTree()
            self.right.add(word)
    def preorder(self):
        if self.left != None :
            self.left.preorder()
        print self.data
        if self.right != None :
            self.right.preorder()
    
    def prebackorder(self):
        nodes = []
        nodes.append(self)
        while len(nodes) > 0 :
            for nd in nodes :
                print nd.data
            nodesTemp = []
            for nd in nodes :
                if nd.left != None : 
                    nodesTemp.append(nd.left)
                if nd.right != None :
                    nodesTemp.append(nd.right)
            nodes = []
            nodes.extend(nodesTemp)
            nodesTemp = []
                
    def prebackpreorder(self):
        nodes = []
        nodes.append(self)
        mark = True
        while len(nodes) > 0 :
            for nd in nodes :
                print nd.data
            nodestemp = []
            for nd in nodes :
                if nd.left != None :
                    nodestemp.append(nd.left)
                if nd.right != None :
                    nodestemp.append(nd.right)
            if mark == False :
                nodestemp.reverse()
            mark = not mark
            nodes = []
            nodes[len(nodes):] = nodestemp
            nodestemp = []
            
            
            
                
        
        
    
bintree = BinTree()
bintree.add(1)
bintree.add(3)
bintree.add(9)
bintree.add(0)
bintree.add(2)

# bintree.preorder()
bintree.prebackorder()
print "==============="
bintree.prebackpreorder()