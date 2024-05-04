import numpy as np
import matplotlib.pyplot as plt 


def weighted_sum_with_bias(x,w,b): 
    return np.dot(x,w) +b

def tanH(z): 
    return (np.exp(z) - np.exp(-z))/ (np.exp(z)+np.exp(-z))

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def relu(z): 
    return np.maximum(z,0)

def leakyRelu(z):
    return np.maximum(z, 0.01*z)

def softmax(x): 
    # for each in z generate an exp(z) then divide their sum by sum of them
    exps = np.exp(x- np.max(x))
    return exps/ np.sum(exps)
    
#input
x = np.array([0,0,1,1])
#weights
w = np.array([0.1, 0.5, .8, .4])
#bias
b = 0.5
#weighted Sum with bais
z = weighted_sum_with_bias(x,w,b)
#activation funciton
a = tanH(z)

#since a is our final output 
y = a

print("weighted sum: ", z)
print("tanH output: ", y)
print("sigmoid output: ", sigmoid(z))
print("relu: ", relu(z))
print("leaky relu", leakyRelu(z))
print("softmax: ", softmax(z))