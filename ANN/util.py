import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmode(x)
    return s*(1-s)

def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    return np.where(x>0, 0, 1)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def d_tanh(x):
    return  1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def leaky_relu(x, alpha = 0.1):
    return np.where(x>0, x, x*alpha)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


if __name__ == "__main__":
    # Test data
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Apply each activation function
    print("Input:", x)
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    print("Softmax:", softmax(x))
    print("Leaky ReLU:", leaky_relu(x))