import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from ann import ANN  # Import the ANN class

# Load the MNIST dataset
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Flatten images to 1D vectors
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)
    
    # Convert labels to one-hot encoding
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Initialize ANN model
    ann = ANN()
    
    # Train model on a small subset of MNIST for testing
    ann.train(train_images[:1000], train_labels[:1000], epochs=10, lr=0.01)  # Train on 1000 samples

    # Test a sample prediction
    sample_image = test_images[0]
    prediction = ann.forward(sample_image)
    predicted_label = np.argmax(prediction)

    # Display the image along with the predicted label
    plt.imshow(sample_image.reshape(28, 28), cmap='gray')  # Show in grayscale
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
