import numpy as np

def getBinDict(size = 1024 , bit_size = 16):
    bin_dict = {}
    for i in range(size):
        s = '{:016b}'.format(i)
        arr = np.array(list(s))
        arr = arr.astype(int)
        bin_dict[i] = arr
    return bin_dict

def getBinDict2():
    int2binary = {}
    binary_dim = 8

    largest_number = pow(2, binary_dim)
    binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]
    print(int2binary)
if __name__ == '__main__':
    s = '{:08b}'.format(1)
    arr = np.array(list(s))
    arr = arr.astype(int)
    print(arr[0])
    getBinDict2()