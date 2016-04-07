'''
Created on 2015-12-8

@author: nocml
'''

from node import Node

tree = Node()

print tree.data
tree.insert(3)
tree.insert(5)
tree.insert(12)
tree.insert(1)
tree.insert(0)

tree.printElement()
print 
tree.HierarchyTraversal()
    