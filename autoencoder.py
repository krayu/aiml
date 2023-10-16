import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Build Autoencoder
autoencoder = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(784,)),  
    layers.Dense(784, activation="sigmoid")  
])

autoencoder.compile(optimizer="adam", loss="mse")

# Train Autoencoder
autoencoder.fit(X_train, X_train, epochs=5, batch_size=256, validation_data=(X_test, X_test))

# Test reconstruction
decoded_imgs = autoencoder.predict(X_test[:10])

# Display original vs reconstructed images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.subplot(2, 10, i+11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()