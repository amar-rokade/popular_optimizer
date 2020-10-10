import numpy as np


def initilization(input_size,layer_size):
    params = {}
    params['W' + str(0)] = np.random.randn(layer_size[0], input_size) * np.sqrt(2 / input_size)
    params['b' + str(0)] = np.zeros((layer_size[0], 1))
    for l in range(1,len(layer_size)):
        params['W' + str(l)] = np.random.randn(layer_size[l],layer_size[l-1]) * np.sqrt(2/layer_size[l])
        params['b' + str(l)] = np.zeros((layer_size[l],1))
    return params


#forward_propagations-----------------------------------------------
def forward_activation(A_prev, w, b, activation):
    z = np.dot(A_prev, w.T) + b.T
    if activation == 'relu':
        A = np.maximum(0, z)
    elif activation == 'sigmoid':
        A = 1/(1+np.exp(-z))
    else:
        A = np.tanh(z)
    return A

#_______ cost___________________________________________________________________________________

def cost_f(Y_pred, Y):
    m = Y.shape[0]
    cost = -1/m * np.sum(Y * np.log(Y_pred ) + (1-Y) * np.log(1-Y_pred))
    return cost

#________model forward ____________________________________________________________________________________________________________
def model_forward(X,params, L,keep_prob):
    cache = {}
    A =X

    for l in range(L-1):
        w = params['W' + str(l)]
        b = params['b' + str(l)]
        A = forward_activation(A, w, b, 'relu')
        if l%2 == 0:
            cache['D' + str(l)] = np.random.randn(A.shape[0],A.shape[1]) < keep_prob
            A = A * cache['D' + str(l)] / keep_prob
        cache['A' + str(l)] = A
    w = params['W' + str(L-1)]
    b = params['b' + str(L-1)]
    A = forward_activation(A, w, b, 'sigmoid')
    cache['A' + str(L-1)] = A
    return cache, A

#____________________________________________________________________________________________________________________
def backward(X, Y, params, cach,L,keep_prob):
    grad ={}
    m = Y.shape[0]

    cach['A' + str(-1)] = X
    grad['dz' + str(L-1)] = cach['A' + str(L-1)] - Y
    cach['D' + str(- 1)] = 0
    for l in reversed(range(L)):
        grad['dW' + str(l)] = (1 / m) * np.dot(grad['dz' + str(l)].T, cach['A' + str(l-1)])
        grad['db' + str(l)] = 1 / m * np.sum(grad['dz' + str(l)].T, axis=1, keepdims=True)
        if l%2 != 0:
            grad['dz' + str(l-1)] = ((np.dot(grad['dz' + str(l)], params['W' + str(l)]) * cach['D' + str(l-1)] / keep_prob) *
                                 np.int64(cach['A' + str(l-1)] > 0))
        else :
            grad['dz' + str(l - 1)] = (np.dot(grad['dz' + str(l)], params['W' + str(l)]) *
                                       np.int64(cach['A' + str(l - 1)] > 0))


    return grad

#______________________________________________________________________________________________________________________
def update_params(params, grad, learning_rate, L): 
    for l in range(L):
        params['W' + str(l)] = params['W' + str(l)] - learning_rate * grad['dW' + str(l)]
        params['b' + str(l)] = params['b' + str(l)] - learning_rate * grad['db' + str(l)]
    return params


#__________________________________________________________________________________________________________________________
def model(X,Y,learning_rate,num_iter,hidden_size,keep_prob):
    L = len(hidden_size)
    params = initilization(X.shape[1], hidden_size)

    for i in range(1,num_iter):

        cache, A = model_forward(X, params, L,keep_prob)
        cost = cost_f(A, Y)
        grad = backward(X, Y, params, cache, L,keep_prob)
        params = update_params(params, grad, learning_rate, L)
        if i%20 == 0:
            print('cost of iteration________________', i)
            print(cost)
    return params

#_____________________________________________________________________________________________________________
def predict(X, params, L):

    A = X
    for l in range(L-1):
        w = params['W' + str(l)]
        b = params['b' + str(l)]
        A = forward_activation(A, w, b, 'relu')
    w = params['W' + str(L-1)]
    b = params['b' + str(L-1)]
    A = forward_activation(A, w, b, 'sigmoid')
    Y_predict = (A > 0.5)
    return Y_predict
#===============================================================================================================



