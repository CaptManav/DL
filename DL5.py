import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images
X_train = X_train / 127.5 - 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator Model
generator = tf.keras.Sequential([
    layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, (5,5), strides=(1,1),
                           padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(64, (5,5), strides=(2,2),
                           padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    layers.Conv2DTranspose(1, (5,5), strides=(2,2),
                           padding='same', use_bias=False,
                           activation='tanh')
])

# Discriminator Model
discriminator = tf.keras.Sequential([
    layers.Conv2D(64, (5,5), strides=(2,2),
                  padding='same', input_shape=[28,28,1]),
    layers.LeakyReLU(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1)
])

# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

print("DCGAN Model Created Successfully")
