import numpy as np
from tool import cross_entropy_loss, d_cross_entropy_loss
from util import tanh, d_tanh, softmax


class RNN:
    def__init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.Wx = np.random.randn() * 0.01
        self.Wh = np.random.randn() * 0.01
        self.Wy = np.random.randn() * 0.01

        self.bh = np.zeros((hidden_size, 1))  # Bias for hidden layer
        self.by = np.zeros((output_size, 1))  # Bias for output layer
    
    def forward(self, inputs, h_prev):

    def backward(self, inputs, targets, h_prev):
        
                

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def forward(self, X):
        """Performs forward propagation."""
        h = np.zeros((1, self.hidden_size))  # Initial hidden state
        self.cache = []  # Store intermediate values for backpropagation
        
        for x in X:
            x = x.reshape(1, -1)
            h = tanh(np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.bh)
            self.cache.append((x, h))
        
        y = softmax(np.dot(h, self.Wy) + self.by)
        return y, h
    
    def backward(self, y_true, lr=0.01):
        """Performs backward propagation through time (BPTT)."""
        dWy = np.zeros_like(self.Wy)
        dWh = np.zeros_like(self.Wh)
        dWx = np.zeros_like(self.Wx)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((1, self.hidden_size))
        
        y_pred, h = self.forward(y_true)
        dL_dy = d_cross_entropy_loss(y_true, y_pred)
        dWy += np.dot(h.T, dL_dy)
        dby += dL_dy
        
        for x, h in reversed(self.cache):
            dh = np.dot(dL_dy, self.Wy.T) + dh_next
            dh_raw = d_tanh(h) * dh
            dWx += np.dot(x.T, dh_raw)
            dWh += np.dot(h.T, dh_raw)
            dbh += dh_raw
            dh_next = np.dot(dh_raw, self.Wh.T)
        
        # Update weights
        self.Wx -= lr * dWx
        self.Wh -= lr * dWh
        self.Wy -= lr * dWy
        self.bh -= lr * dbh
        self.by -= lr * dby
    
    def train(self, X_train, y_train, epochs=10, lr=0.01):
        for epoch in range(epochs):
            loss = 0
            for X, y in zip(X_train, y_train):
                y_pred, _ = self.forward(X)
                loss += cross_entropy_loss(y, y_pred)
                self.backward(y, lr)
            
            print(f"Epoch {epoch+1}, Loss: {loss/len(X_train)}")
