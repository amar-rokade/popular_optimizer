def update_params_with_momentum(params, grads, v, beta, learning_rate):
    
    # grads has the dw and db parameters from backprop
    # params  has the W and b parameters which we have to update 
    for l in range(len(params) // 2 ):

        # HERE WE COMPUTING THE VELOCITIES 
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        
        #updating parameters W and b
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    return params