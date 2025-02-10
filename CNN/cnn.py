import numpy as np
from tool import mse_loss, d_mse_loss, cross_entropy_loss, d_cross_entropy_loss
from util import relu, d_relu, softmax


class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size)

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size +1):
            for j in range(w - self.filter_size +1):
                region = image[i:i+self.filter_size, j:j+self.filter_size]
                yield i, j, region

    def forward(self, input):
        self.last_input = input 
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for i, j, region in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1,2))

        return output
    

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, input):
        h, w, num_filters = input.shape
        for i in range(0, h, self.pool_size):
            for j in range(0, w, self.pool_size):
                region = input[i:i+self.pool_size, j:j+self.pool_size]
                yield i, j, region

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        
        for i, j, region in self.iterate_regions(input):
            output[i//self.pool_size, j//self.pool_size] = np.amax(region, axis=(0, 1))
        
        return output

    def backward(self, dL_dout):
        return None  # No learning, no need for backpropagation

class FullyConnectedLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

    def forward(self, input):
        self.last_input = input.flatten()
        return np.dot(self.last_input, self.weights) + self.biases

    def backward(self, dL_dout, lr):
        dL_dweights = np.outer(self.last_input, dL_dout)
        dL_dbiases = dL_dout
        
        self.weights -= lr * dL_dweights
        self.biases -= lr * dL_dbiases
        
        return None  # No further backpropagation

class CNN:
    def __init__(self):
        self.conv = ConvLayer(8, 3)
        self.pool = MaxPoolLayer(2)
        self.fc = FullyConnectedLayer(13*13*8, 10)  # Assuming input size 28x28 and pooling

    def forward(self, input):
        output = self.conv.forward(input)
        output = relu(output)
        output = self.pool.forward(output)
        output = self.fc.forward(output)
        return softmax(output)

    def train(self, images, labels, epochs, lr):
        for epoch in range(epochs):
            loss = 0
            for img, label in zip(images, labels):
                out = self.forward(img)
                loss += cross_entropy_loss(label, out)
                dL_dout = d_cross_entropy_loss(label, out)
                self.fc.backward(dL_dout, lr)
            
            print(f"Epoch {epoch+1}, Loss: {loss/len(images)}")