import sys;
print 'hh'

print('Hello World!')
# s=(1,2,3)
# print(max(s))
#  
# for a in [3,4.4,'life']:
#     print a
# i=0
# while i < 10:
#     print i
#     i = i + 1
#      
# class Bird(object):
#     have_feather = True
#     way_of_reproduction  = 'egg'
#      
# summer = Bird()
# print summer.way_of_reproduction
# 
class Bird(object):
    have_feather = True
    way_of_reproduction = 'egg'
    def move(self, dx , dy):
        position = [1,2]
        position[0] = position[0] + dx
        position[1] = position[1] + dy
        return position

summer = Bird()
print 'after move:',summer.move(5,8)

class Chicken(Bird):
    way_of_move = 'walk'
    possible_in_KFC = True

class Oriole(Bird):
    way_of_move = 'fly'
    possible_in_KFC = False

summer = Chicken()
print summer.have_feather
print summer.possible_in_KFC
print summer.move(5,8)