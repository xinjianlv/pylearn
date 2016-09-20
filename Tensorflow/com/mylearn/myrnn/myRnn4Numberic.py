import copy,numpy as np
from com.mylearn.myrnn import DataProcess
np.random.seed(0)

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)



base = [str(x) for x in range(10)] + [ chr(x) for x in range(ord('A'),ord('A')+6)]
print('base:' , base)
def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num,rem = divmod(num, 2)
        mid.insert(0, int(base[rem]))
    ss =  ''.join([str(x) for x in mid[::-1]])
    sslength = len(ss)
    for i in range( 32 - sslength ):
       mid.insert(0, 0)
    return mid
alpha = 0.1
input_dim = 32
hidden_dim = 128
output_dim = 32

wordDic = DataProcess.getdic('/Users/nocml/Documents/workspace/python/Tensorflow/com/mylearn/myrnn/data/people.txt')
data = DataProcess.loadData('/Users/nocml/Documents/workspace/python/Tensorflow/com/mylearn/myrnn/data/people.txt')
synapse_0 = 2*np.random.random((input_dim , hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim , output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim , hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)


layer_2_deltas = list()
layer_1_values = list()
layer_1_values.append(np.zeros(hidden_dim))



for i in range(1000000):
    Xline =[i]
    yline =[i + 1]
    out = []
    overallError = 0
    slength = len(Xline)
    for position in range(slength):
        X = np.array([dec2bin(Xline[position])])
        y = np.array([dec2bin(yline[position])])
        #hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X , synapse_0)  + np.dot(layer_1_values[-1],synapse_h))

        #output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1 , synapse_1))

        #did we miss? .. if so , by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))

        overallError += np.sum( np.abs(layer_2_error[0]))

        #decode estimate so we can print it out
        bitvalue = []
        for ndx in range( len(layer_2[0])):
            bitvalue.append( str( int(np.round(layer_2[0][ndx]))))
        out.append(int("".join(bitvalue),2))
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(slength):
        X = np.array(Xline[-position - 1])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        #error at output layer
        layer_2_delta = layer_2_deltas[-position-1]

        #error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)


        #let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    #print out progress
    if( i % 1000== 0):
        print ("Error:)"+str(overallError))
        print ("predict:" , out)
        print ("true:" , Xline)
        print('--------------')