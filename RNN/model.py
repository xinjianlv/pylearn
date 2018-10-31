import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pdb

class RNN(nn.Module):

    def __init__(self,input_dim , hidden_dim):
        super(RNN,self).__init__()
        self._rnn = nn.RNN(input_size = input_dim , hidden_size= hidden_dim ,num_layers=1)
        self.linear = nn.Linear(hidden_dim , 1)
        self.relu = nn.ReLU()

    def forward(self , _in ):
        layer1 , h = self._rnn(_in)
        layer2 = self.relu(self.linear(self.relu(layer1)))
        return layer2 , h

    def init_weight(self):
        nn.init.normal_(self.linear.weight.data  , 0 , np.sqrt(2 / 16))
        nn.init.uniform_(self.linear.bias, 0, 0)
        # pass
def getBinDict(bit_size = 16):
    max = pow(2,bit_size)
    bin_dict = {}
    for i in range(max):
        s = '{:016b}'.format(i)
        arr = np.array(list(s))
        arr = arr.astype(int)
        bin_dict[i] = arr
    return bin_dict

binary_dim = 16
int2binary = getBinDict(binary_dim)

def getBatch( batch_size):
    x = np.random.randint(0,256,[batch_size , 2])
    x_arr = np.zeros([binary_dim , batch_size , 2 ] , dtype=int)
    y_arr = np.zeros([binary_dim,batch_size,1] , dtype=int)
    for i in range(0 , binary_dim):
        batch_x_arr = np.zeros([batch_size,2] , dtype=int)
        batch_y_arr = np.zeros([batch_size,1] , dtype=int)
        for j in range(len(x)):
            batch_x_arr[j] =[int2binary[int(x[j][0])][i] , int2binary[int(x[j][1])][i]]
            batch_y_arr[j] =[int2binary[ int(x[j][0]) + int(x[j][1])][i]]

        #此处要翻转，rnn处理时是从下标为0处开始，所以要把二进制的高低位翻转
        y_arr[binary_dim - i - 1] = batch_y_arr
        x_arr[binary_dim - i - 1] = batch_x_arr
    return x_arr , y_arr , x

def getBatch_bf( batch_size ):
    x = np.random.randint(0,256,[batch_size , 2])
    x_arr = np.zeros([batch_size , binary_dim , 2 ] , dtype=int)
    y_arr = np.zeros([batch_size,binary_dim,1] , dtype=int)
    for i in range(0 , batch_size):
        batch_x_arr = np.zeros([binary_dim,2] , dtype=int)
        batch_y_arr = np.zeros([binary_dim,1] , dtype=int)
        for j in range(binary_dim):
            batch_x_arr[j] =[int2binary[int(x[i][0])][-(j + 1)] , int2binary[int(x[i][1])][-(j + 1)]]
            batch_y_arr[j] =[int2binary[ int(x[i][0]) + int(x[i][1])][-(j+1)]]
        y_arr[i] = batch_y_arr
        x_arr[i] = batch_x_arr
    return x_arr , y_arr , x

def getInt(y , bit_size):
    arr = np.zeros([len(y[0])])
    for i in range(len(y[0])):
        for j in range(bit_size):
            arr[i] += (int(y[j][i][0]) * pow(2 , j))
    return arr

if __name__ == '__main__':
    input_size = 2
    hidden_size = 8
    batch_size = 100
    net = RNN(input_size, hidden_size)
    net.init_weight()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = nn.MSELoss()#.CrossEntropyLoss()
    h = torch.Tensor(1,batch_size ,hidden_size)
    for i in range(100000):
        net.zero_grad()
        x ,y , t = getBatch(batch_size)
        in_x = torch.Tensor(x)
        y = torch.Tensor(y)
        output , h2 = net(in_x)
        loss = loss_function(output , y)
        loss.backward()
        optimizer.step()

        if i % 100== 0:
            output2 = torch.round(output)
            result = getInt(output2,binary_dim)
            print(t , result)
            print('iterater:%d  loss:%f'%(i , loss))

