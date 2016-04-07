
class Node(object):
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None
 
    def insert(self , data):
        if self.data == None :
            self.data = data
        else:
            if self.data > data :
                if self.left == None:
                    self.left =  Node()
                self.left.insert(data)
            else :
                if self.right == None:
                    self.right = Node()
                self.right.insert(data)
    
    def printElement(self):
        if self.left != None:
            self.left.printElement()
       
        print self.data,
        
        if self.right != None:
            self.right.printElement()
            
    def HierarchyTraversal(self):
  
        lst = list()
        listTemp = list()
        lst.append(self)
        while len(lst) > 0 :
            for ele in lst :
                if ele != None :
                    if ele.data != None : 
                        print ele.data,
                    if ele.left != None:
                        listTemp.append(ele.left)
                    if ele.right != None:
                        listTemp.append(ele.right)
            lst = []
            lst = listTemp
            listTemp = []
            
               
        
        
        
                

    
    