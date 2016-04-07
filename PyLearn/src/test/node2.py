'''
Created on 

@author: nocml
'''
class nodetest(object):
    data = None
    left = None
    right = None
    
    def insert(self,data):
        if self.data == None:
            self.data = data
        else:
            if self.data < data:
                if self.left == None:
                    self.left = nodetest()
                self.left.insert(data)
            if self.data > data:
                if self.right == None:
                    self.right = nodetest()
                self.right.insert(data)
    
    def preOrder(self):
        if self.left != None:
            self.left.preOrder()
        if self.data != None:
            print(self.data)
        if self.right != None:
            self.right.preOrder()
            
t = nodetest()
t.insert(5)
t.insert(6)
t.insert(7)
t.insert(100)
t.insert(3)

t.preOrder()
        
    