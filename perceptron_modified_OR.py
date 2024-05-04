import numpy as np


#and gate     x1, x2,
t1 = np.array([0,0])
t2 = np.array([0,1])
t3 = np.array([1,0])
t4 = np.array([1,1])
#weights 1,1, 
w = np.array([1,1])

# shift bias to here instead of input 
# for or bias is 0, and is 1
b = 0

def perceptron(x, w, b = 0.5): 
   if(( np.dot(w,x) + b) > 0):
      return 1
   else: return 0


y = perceptron(t1, w, b)
print(y)
y = perceptron(t2, w, b)
print(y)
y = perceptron(t3, w, b)
print(y)
y = perceptron(t4, w, b)
print(y)

