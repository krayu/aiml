import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), _ = keras.datasets.mnist.load_data()
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# Define models
generator = keras.Sequential([
    layers.Dense(128, activation="relu", input_dim=100),
    layers.Dense(784, activation="tanh"),
    layers.Reshape((28, 28, 1))
])

discriminator = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
discriminator.trainable = False

gan = keras.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="adam")

# Training loop
for epoch in range(10000):
    noise = np.random.normal(0, 1, (128, 100))
    fake_imgs = generator.predict(noise)
    real_imgs = X_train[np.random.randint(0, X_train.shape[0], 128)]
    
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((128, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((128, 1)))
    
    gan.train_on_batch(noise, np.ones((128, 1)))

    if epoch % 2000 == 0:
        plt.imshow(fake_imgs[0].reshape(28, 28), cmap="gray")
        plt.show()
