import numpy as np

def relu(z): 
    return np.maximum(z,0)

def xor_relu(a= 0,b= 0): 
    #nueron 1,2
    x1 = [a, b]
    w1 = [1, 1]

    # relu unit 1, bias 0
    z1 = np.dot(x1, w1) + 0
    a1 = relu(z1)
    
    # relu unit 2, bias -1
    z2 = np.dot(x1, w1) -1
    a2 = relu(z2)


    #pass both units into final relu unit
    new_inps = np.array([a1,a2])
    #special bias, unit 1: 1, unit 2, -2
    new_weights = np.array([1,-2])

    z_final = np.dot(new_inps, new_weights) + 0

    y = relu(z_final)
    print(y)


xor_relu(0,0)
xor_relu(0,1)
xor_relu(1,0)
xor_relu(1,1)