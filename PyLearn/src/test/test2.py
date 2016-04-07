class Base(object):
    elem1 = 1
    elem2 = 1
    def function(self , dx , dy):
        t1 = 1
        t2 = 1
        t1 +=  dx
        t2 += dy
        return  t1 , t2,3
        
base =  Base();
value =  base.function(10, 11)
print value[0]
print True.__and__(False)
# root = "D:\\data\\wanda\\"
# f = open(root + "test.txt","r")
# lines = f.readlines()
# for (index , line) in  enumerate(lines):
#     print index,line
    