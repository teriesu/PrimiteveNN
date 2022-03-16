import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

#CREACIÓN DEL DATASET
N  = 1000 #Creación de 1000 ejemplos
gaussian_quantiles= make_gaussian_quantiles(mean=None, cov=0.1, n_samples = N, n_features=2,
                                            n_classes = 2, shuffle=True, random_state = None)

X, Y = gaussian_quantiles
Y = Y[:, np.newaxis]

plt.scatter(X[:, 0], X[:, 1], c = Y[:, 0], s = 40, cmap=plt.cm.Spectral)
plt.show()
####################### FUNCIONES DE ACTIVACION #######################
def sigmoid(x, derivate = False):
    if derivate:
        return np.exp(-x)/((np.exp(-x)+1)**2)
    else:
        return (1/(1+np.exp(-x)))

def relu(x, derivate = False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.maximum(0,x)
####################### Loss Function #######################
def mse(y, y_hat, derivate = False):
    if derivate:
        return y_hat-y
    else:
        return np.mean((y_hat - y)**2)

############### INICIALIZACION DE PARÁMETROS ###############
def initialize_parameters_deep(layers_dim):
    parameters = {}
    L = len(layers_dim)
    # Creación de los pesos iniciales #
    for l in range(0, L-1):
        parameters['W' + str(l+1)] =  (np.random.rand(layers_dim[l],layers_dim[l+1]) * 2 ) - 1 #inicializamos pesos (W)
        parameters['b' + str(l+1)] =  (np.random.rand(1,layers_dim[l+1]) * 2 ) - 1 #inicializamos bias (b)
    return parameters

################ ENTRENAMIENTO Y PREDICCIÓN #################
def train(x_data, lr, params, training = True):
    ## FORWARD
    params['A0'] = x_data

    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1'] #producto punto de los parámetros de entrada * pesos +bias de la capa
    params['A1'] = relu(params['Z1']) #Al resultado anterior lo paso por una funcion de activación

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2'] #Recibe el resultado de la anterior (A1)
    params['A2'] = relu(params['Z2']) #Al resultado anterior lo paso por una funcion de activación

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3'] #Recibe el resultado de la anterior (A2)
    params['A3'] = sigmoid(params['Z3']) #Al resultado anterior lo paso por una funcion de activación

    output = params['A3'] #En esta variable almacenamos los datos de salida

    ## BACKPROPAGATION
    if training:
        #Solo aplica para la última capa:
        params['dZ3'] = mse(Y, output, True) * sigmoid(params['A3'], True) #Delta de Z3 = Derivada del loss fnction * funcion de activación
        params['dW3'] = np.matmul(params['A2'].T,params['dZ3']) #Delta de los pesos = producto punto de lo que obtuvo en la fx anterior * delta de Z3

        #Demás capas
        params['dZ2'] = np.matmul(params['dZ3'],params['W3'].T) * relu(params['A2'],True) #salida de la capa anterior 'punto' pesos de la capa anterior * fx de activación
        params['dW2'] = np.matmul(params['A1'].T,params['dZ2']) #Delta de los pesos = producto punto de lo que obtuvo en la fx anterior * delta de Z2

        params['dZ1'] = np.matmul(params['dZ2'],params['W2'].T) * relu(params['A1'],True)
        params['dW1'] = np.matmul(params['A0'].T,params['dZ1'])

        ## Gradient Descent:
        params['W3'] = params['W3'] - params['dW3'] * lr #Nuevos pesos son = Los pesos - delta de los pesos * el learning rate
        params['b3'] = params['b3'] - (np.mean(params['dZ3'],axis=0, keepdims=True)) * lr #Nuevos bias son = Los bias - media del delta de los bias * el learning rate

        params['W2'] = params['W2'] - params['dW2'] * lr
        params['b2'] = params['b2'] - (np.mean(params['dZ2'],axis=0, keepdims=True)) * lr

        params['W1'] = params['W1'] -params['dW1'] * lr
        params['b1'] = params['b1'] - (np.mean(params['dZ1'],axis=0, keepdims=True)) * lr

    return output

################## GUARDADO DE PARÁMETROS ##################
layers_dim = [2, 4, 8, 1] #Creamos las dimensiones de la red neuronal:
                            #Dos parámetros de entrada
                            #4 Neuronas en la primera capa oculta
                            #8 neuronas en la capa siguiente
                            #1 Como es clasificación binaria desemboca en una sola neurona

params = initialize_parameters_deep(layers_dim) #Almacena los parámetros de cada capa con su respectivo bias
errors = []
for _ in range(80000): #Epocas
    output = train(X, 0.00001, params)
    if _ % 50 == 0:
        print(mse(Y, output))
        errors.append(mse(Y, output))

#plt.plot(errors)

data_test = (np.random.rand(1000, 2) * 2) - 1
y = train(data_test, 0.0001, params, training = False)
y = np.where(y >= 0.1, 1, 0)
plt.scatter(data_test[:, 0], data_test[:, 1], c = y[:, 0], s = 40, cmap=plt.cm.Spectral)
plt.show()
