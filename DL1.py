import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Price": [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

print("Predicted Price:", prediction)
print("MSE:", mean_squared_error(y_test, prediction))
