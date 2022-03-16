import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a):
    return (1/(1+np.exp(-a)))

def sigmoid2(x, derivate = False):
    if derivate:
        return np.exp(-x)/((np.exp(-x)+1)**2)
    else:
        return (1/(1+np.exp(-x)))

def step(a):
    return np.piecewise(x, [x<0.0, x>0.0], [0,1])

def relu(x, derivate = False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.maximum(0,x)

def softmax(a):
    return np.exp(x) / np.sum(np.exp(x))

x = np.linspace(10,-10, 100) #Me crea un array de 100 valores entre -10 y 10
plt.plot(x, sigmoid(x))
plt.plot(x, step(x))
plt.plot(x, relu(x))
plt.plot(x, softmax(x))
plt.show()

def mse(y, y_hat, derivate = False):
    if derivate:
        return y_hat-y
    else:
        return np.mean((y_hat - y)**2)

real = np.array([0,0,1,1])
prediction = np.array([0.9,0.5,0.2,0.0])
print(mse(real, prediction))
