import torch.nn as nn
import torch
from torch.autograd import Variable
# rnn = nn.RNN(256, 256, 1)
# input = torch.randn(1,1,256)
# h0 = torch.randn(1, 1, 256)
# print(input.size())
# output, hn = rnn(input)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(16, 2,100,1, requires_grad=True)
# target = torch.empty(16,100,1, dtype=torch.long).random_(2)
# output = loss(input, target)
# print(input)
# print(target)
# print(output)
#
# output.backward()

# 构造RNN网络，x的维度5，隐层的维度10,网络的层数2
rnn_seq = nn.RNN(5, 10,2)
# 构造一个输入序列，长为 6，batch 是 3， 特征是 5
x = Variable(torch.randn(6, 3, 5))
out,ht = rnn_seq(x) # h0可以指定或者不指定
# q1:这里out、ht的size是多少呢？ out:6*3*10, ht:2*3*5
# q2:out[-1]和ht[-1]是否相等？  相等！
print(out.size())
print(ht.size())