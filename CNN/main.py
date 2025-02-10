import numpy as np
from cnn import CNN  # Import the CNN class from the existing script

if __name__ == "__main__":
    # Generate simple training data: 100 samples of 28x28 grayscale images
    images = np.random.rand(100, 28, 28)
    labels = np.eye(10)[np.random.randint(0, 10, 100)]  # One-hot encoded labels
    
    cnn = CNN()
    cnn.train(images, labels, epochs=10, lr=0.01)
