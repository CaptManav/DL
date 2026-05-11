import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# Padding
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

# Build model
model = Sequential([
    Embedding(5000, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=2, batch_size=64)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

#sentiment Pred
from tensorflow.keras.datasets import imdb
word_index = imdb.get_word_index()

from tensorflow.keras.preprocessing.sequence import pad_sequences
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=100)
    return padded_review

user_review = input("Enter a review: ")
processed_user_review = preprocess_text(user_review)
prediction = model.predict(processed_user_review)

if prediction[0][0] > 0.5:
    sentiment = "Positive"
else:
    sentiment = "Negative"

print(f"Your review: {user_review}")
print(f"Prediction probability: {prediction[0][0]:.4f}")
print(f"Predicted sentiment: {sentiment}")














