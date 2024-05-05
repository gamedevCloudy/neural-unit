#same as relu xor but here we try to use a unit based code structure instead of a single funciton
import numpy as np

class ReLU:
    def __init__(self, x, w=[1,1], b = 0): 
        self.x = x
        self.w = w
        self.b = b
    
    def weightedSum(self):
        return np.dot(self.x, self.w) + self.b
    
    def activation(self, z):
        return np.maximum(0, z)


# start conditions
inps = np.array([0,0])
w1 = np.array([1,1])
b1 = 0
b2 = -1

# layer 1
# unit 1
h1 = ReLU(inps, w1, b1)
z1= h1.weightedSum()
a1 = h1.activation(z1)

# layer 1, unit 2
h2 = ReLU(inps, w1, b2)
z2= h2.weightedSum()
a2 = h2.activation(z2)


# final layer
inp_final = np.array([a1,a2])
w_final = np.array([1,-2])
b_final = 0 


output = ReLU(inp_final, w_final, b_final)
z3 = output.weightedSum()
y = output.activation(z3)

print(y)