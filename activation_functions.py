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


#visualize all activation funcitons
x_axis = np.linspace(-10, 10, 100)
# activation functions
tanh_output = tanH(x_axis)
sigmoid_output = sigmoid(x_axis)
relu_output = relu(x_axis)
leaky_relu_output = leakyRelu(x_axis)
softmax_output = softmax(x_axis)




plt.plot(x_axis, sigmoid(x_axis), label="Sigmoid")
plt.plot(x_axis, tanH(x_axis), label="Tanh")
plt.plot(x_axis, relu(x_axis), label="ReLU")
plt.plot(x_axis, leakyRelu(x_axis), label="Leaky ReLU")
plt.plot(x_axis, softmax(x_axis), label="Softmax")

plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()