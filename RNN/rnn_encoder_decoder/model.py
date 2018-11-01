import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
import pdb



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

class RNN_Decoder(nn.Module):
    def __init__(self, input_size , hidden_size, output_size):
        super(RNN_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

teacher_forcing_ratio = 0.5

def train(input_seq , target, encoder , decoder , encoder_optimizer ,decoder_optimizer, criterion ,max_length):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    hidden = encoder.init_hidden()
    encoder_outputs = torch.zeros(max_length)
    for ndx in range(max_length):
        x_in = torch.Tensor([input_seq[0][ndx] , input_seq[1][ndx]])
        output , hidden = encoder(x_in , hidden)
        encoder_outputs[ndx] = output[0,0]

    decoder_input = torch.tensor([0.0])

    decoder_hidden = hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    loss = 0
    if True :#use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len(target)):
            # decoder_input = torch.Tensor(np.array([target[di]]))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, torch.Tensor(np.array([[target[di]]])))
            decoder_input = torch.Tensor(np.array([target[di]])).float()  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(len(target)):
            # decoder_input = torch.Tensor(np.array([target[di]]))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(0).detach().float()  # detach from history as input
            loss += criterion(decoder_output, torch.Tensor(np.array([[target[di]]])))


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()



    return loss.item() / len(target) , encoder_outputs

def trainIter(batch_x , batch_y , encoder ,decoder, max_length,learning_rate):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loss = 0
    predict = np.zeros([batch_size , max_length])
    for ndx in range(len(batch_x)):
        loss_ , encoder_outputs = train(batch_x[ndx],batch_y[ndx], encoder ,decoder,encoder_optimizer,decoder_optimizer,criterion, max_length)
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
    hidden_size = 16
    batch_size = 100
    output_size = 1
    encoder = RNN_Encoder(input_size, hidden_size)
    decoder = RNN_Decoder(1,hidden_size , output_size)
    encoder.init_weight()
    print(encoder)
    print(decoder)
    for i in range(100000):
        encoder.zero_grad()
        decoder.zero_grad()
        h0 = torch.zeros(1, batch_size, hidden_size)
        x , y , t = getBatch(batch_size , binary_dim)
        loss , outputs = trainIter(x , y , encoder ,decoder, binary_dim , 0.01)
        print('iterater:%d  loss:%f' % (i, loss))
        if i % 100== 0:
            output2 = np.round(outputs)
            result = getInt(output2,binary_dim)
            print(t ,'\n', result)

            print('iterater:%d  loss:%f'%(i , loss))

