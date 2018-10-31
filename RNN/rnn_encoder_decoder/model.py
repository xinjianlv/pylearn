import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pdb

# torch._cudnn_rnn()

class RNN_Encoder(nn.Module):

    def __init__(self,input_dim , hidden_dim):
        super(RNN_Encoder,self).__init__()
        self._rnn = nn.RNN(input_size = input_dim , hidden_size= hidden_dim )
        self.linear = nn.Linear(hidden_dim , 1)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_dim
    def forward(self , input , hidden_input):
        input = input.view(1, 1, -1)
        layer1 , h = self._rnn(input,hidden_input)
        layer2 = self.relu(self.linear(layer1))
        return layer2 , h

    def init_weight(self):
        nn.init.normal_(self.linear.weight.data  , 0 , np.sqrt(2 / 16))
        nn.init.uniform_(self.linear.bias, 0, 0)
    def init_hidden(self):
        return torch.zeros([1,1,self.hidden_size])

def train(input_seq , target, encoder , optim , criterion ,max_length):
    optim.zero_grad()
    hidden = encoder.init_hidden()
    encoder_outputs = torch.zeros(max_length)
    for ndx in range(max_length):
        x_in = torch.Tensor([input_seq[0][ndx] , input_seq[1][ndx]])
        output , hidden = encoder(x_in , hidden)
        encoder_outputs[ndx] = output[0,0]

    target = torch.Tensor(target)
    loss = criterion(encoder_outputs, target)
    loss.backward()
    optim.step()

    return loss , encoder_outputs

def trainIter(batch_x , batch_y , encoder , max_length,learning_rate):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss = 0
    predict = np.zeros([batch_size , max_length])
    for ndx in range(len(batch_x)):
        loss_ , encoder_outputs = train(batch_x[ndx],batch_y[ndx], encoder ,encoder_optimizer,criterion, max_length)
        loss += loss_
        predict[ndx] = encoder_outputs.detach().numpy()
    return loss , predict


def getBinDict(bit_size = 16):
    max = pow(2,bit_size)
    bin_dict = {}
    for i in range(max):
        s = '{:016b}'.format(i)
        arr = np.array(list(reversed(s)))
        arr = arr.astype(int)
        bin_dict[i] = arr
    return bin_dict

binary_dim = 16
int2binary = getBinDict(binary_dim)

def getBatch( batch_size , binary_size):
    x = np.random.randint(0,256,[batch_size , 2])
    batch_x = np.zeros([batch_size , 2,binary_size] )
    batch_y = np.zeros([batch_size , binary_size])
    for i in range(0 , batch_size):
        batch_x[i][0] = int2binary[x[i][0]]
        batch_x[i][1] = int2binary[x[i][1]]
        batch_y[i] = int2binary[x[i][0] + x[i][1]]
    return batch_x , batch_y , [a + b for a,b in x]

def getInt(y , bit_size):
    arr = np.zeros([len(y)])
    for i in range(len(y)):
        for j in range(bit_size):
            arr[i] += (int(y[i][j]) * pow(2 , j))
    return arr

if __name__ == '__main__':
    input_size = 2
    hidden_size = 8
    batch_size = 100
    net = RNN_Encoder(input_size, hidden_size)
    net.init_weight()
    print(net)
    for i in range(100000):
        net.zero_grad()
        h0 = torch.zeros(1, batch_size, hidden_size)
        x , y , t = getBatch(batch_size , binary_dim)
        loss , outputs = trainIter(x , y , net , binary_dim , 0.01)

        if i % 100== 0:
            output2 = np.round(outputs)
            result = getInt(output2,binary_dim)
            print(t ,'\n', result)

            print('iterater:%d  loss:%f'%(i , loss))

