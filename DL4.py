import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build CNN model
model = models.Sequential([

    # First Convolution Layer
    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(32,32,3)
    ),

    layers.MaxPooling2D((2,2)),

    # Second Convolution Layer
    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    # Flatten Layer
    layers.Flatten(),

    # Dense Layer
    layers.Dense(64, activation='relu'),

    # Dropout Layer
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
