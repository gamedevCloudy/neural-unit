import numpy as np


#and gate       x1, x2, special input 1
t1 = np.array([0,0, 1])
t2 = np.array([0,1, 1])
t3 = np.array([1,0, 1])
t4 = np.array([1,1, 1])
#weights 1,1, special bias 0
w = np.array([1,1,0])
b = 0.5

def perceptron(x, w, b = 0.5): 
   if(( np.dot(w,x) + b) > 0):
      return 1
   else: return 0


y = perceptron(t1, w, 0)
print(y)
y = perceptron(t2, w, 0)
print(y)
y = perceptron(t3, w, 0)
print(y)
y = perceptron(t4, w, 0)
print(y)

