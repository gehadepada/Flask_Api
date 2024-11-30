from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and prepare the data
data = pd.read_csv('patients.csv')
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Add a column of ones for the intercept term
features = np.c_[np.ones(features.shape[0]), features]

# Split data into training and test sets
train_size = int(0.8 * features.shape[0])
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Compute theta using the normal equation
def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

theta = normal_equation(X_train, y_train)

# Prediction function
def predict(feature1, feature2, feature3):
    # Create a feature array
    features = np.array([[1, feature1, feature2, feature3]])
    
    # Make prediction
    prediction = features.dot(theta)
    
    # Return the prediction
    return prediction

# Example usage of the predict function
feature1 = 1.0
feature2 = 2.0
feature3 = 3.0
prediction = predict(feature1, feature2, feature3)
print(f"The predicted value is: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)
