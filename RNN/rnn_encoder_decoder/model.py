import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
import pdb

MAX_LENGTH = 8

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
        # layer2 = self.relu(self.linear(layer1))
        return layer1 , h

    def init_weight(self):
        nn.init.normal_(self.linear.weight.data  , 0 , np.sqrt(2 / self.hidden_size))
        nn.init.uniform_(self.linear.bias, 0, 0)
    def init_hidden(self):
        return torch.zeros([1,1,self.hidden_size])


class RNN_Decoder(nn.Module):

    def __init__(self, input_size , hidden_size, output_size, max_length= MAX_LENGTH):
        super(RNN_Decoder, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(hidden_size + input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # self.relu = nn.ReLU()
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size + input_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    def score(self, hidden, encoder_outputs):
        '''
          :param hidden: 
              previous hidden state of the decoder, in shape (layers*directions,B,H)
          :param encoder_outputs:
              encoder outputs from Encoder, in shape (T,B,H)
          :return
              attention energies in shape (B,T)
          '''
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

    def forward(self, input, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_hiddens = encoder_outputs.transpose(0, 1)  # [B*T*H]

        attn_energies = self.score( H , encoder_hiddens)

        attn_weights = F.softmax(attn_energies,dim=1).unsqueeze(1)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)

        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((input, context), 2)
        # rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        # pdb.set_trace()
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        # pdb.set_trace()
        output = self.softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden

    def init_weight(self):
        nn.init.normal_(self.out.weight.data, 0, np.sqrt(2 / self.hidden_size))
        nn.init.uniform_(self.out.bias, 0, 0)

    def initHidden(self):

        return torch.zeros(1, 1, self.hidden_size)

teacher_forcing_ratio = 0.5

def train(input_seq , target, encoder , decoder , encoder_optimizer ,decoder_optimizer, criterion ,max_length):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    hidden = encoder.init_hidden()
    # encoder_outputs = torch.zeros(max_length)
    encoder_outputs = torch.zeros(encoder.hidden_size, 1 , encoder.hidden_size)
    # encoder_hiddens = torch.zeros(max_length, encoder.hidden_size)
    decoder_outputs = torch.zeros(decoder.hidden_size , decoder.input_size)
    for ndx in range(max_length):
        x_in = torch.Tensor([input_seq[0][ndx] , input_seq[1][ndx]])
        output , hidden = encoder(x_in , hidden)
        encoder_outputs[ndx] = output[0]

    decoder_input = torch.tensor([[[0.0 , 0.0]]])

    decoder_hidden = hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    loss = 0
    if use_teacher_forcing :#use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len(target)):
            # decoder_input = torch.Tensor(np.array([target[di]]))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[di] = decoder_output[0]
            # loss += criterion(decoder_output, torch.Tensor(np.array([target[di]])).long())
            arr = np.zeros([1,1,2],dtype=np.int)
            arr[0][0][int(target[di])] = 1
            decoder_input = torch.Tensor(arr).float()  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(len(target)):
            # decoder_input = torch.Tensor(np.array([target[di]]))
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,encoder_outputs)
            decoder_outputs[di] = decoder_output[0]
            topi = decoder_output[0]
            decoder_input = topi.repeat(1,1,1).float()  # detach from history as input
            # loss += criterion(decoder_output, torch.Tensor(np.array([target[di]])).long())

            #criterion(decoder_output, torch.Tensor(np.array([target[di]])).long())
    loss = criterion(decoder_outputs , torch.Tensor(target).long())
    indices = torch.argmax(decoder_outputs , dim=1)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()



    return loss.item() / len(target) , indices

def trainIter(batch_x , batch_y , encoder ,decoder, max_length,learning_rate):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    encoder.zero_grad()
    decoder.zero_grad()
    criterion = nn.CrossEntropyLoss()#nn.MSELoss()
    loss = 0
    predict = np.zeros([batch_size , len(batch_y[0])])
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
    output_size = 2
    encoder = RNN_Encoder(input_size, hidden_size)
    decoder = RNN_Decoder(output_size,hidden_size , output_size)
    decoder.init_weight()
    encoder.init_weight()
    print(encoder)
    print(decoder)
    for i in range(100000):

        h0 = torch.zeros(1, batch_size, hidden_size)
        x , y , t = getBatch(batch_size , binary_dim)
        loss , outputs = trainIter(x , y , encoder ,decoder, 5 , 0.001)
        print('iterater:%d  loss:%f' % (i, loss))
        if i % 100== 0:
            result = getInt(outputs,binary_dim)
            print(t ,'\n', result)

            print('iterater:%d  loss:%f'%(i , loss))

