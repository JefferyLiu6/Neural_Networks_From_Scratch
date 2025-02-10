import numpy as np
from tool import cross_entropy_loss, d_cross_entropy_loss
from util import relu, d_relu, softmax

class ANN:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
    
    def forward(self, X):
        """Performs forward propagation."""
        self.X = X.reshape(1, -1)  # Flatten input
        
        self.Z1 = np.dot(self.X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        
        return self.A3
    
    def backward(self, y_true, lr=0.01):
        """Performs backward propagation and updates weights."""
        m = y_true.shape[0]
        dZ3 = d_cross_entropy_loss(y_true, self.A3)
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * d_relu(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * d_relu(self.Z1)
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
    
    def train(self, X_train, y_train, epochs=10, lr=0.01):
        for epoch in range(epochs):
            loss = 0
            for X, y in zip(X_train, y_train):
                output = self.forward(X)
                loss += cross_entropy_loss(y, output)
                self.backward(y, lr)
            
            print(f"Epoch {epoch+1}, Loss: {loss/len(X_train)}")
