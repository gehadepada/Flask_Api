from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Linear regression by gradient descent--------------------------------------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('patients.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Manual Standardization
means = np.mean(features, axis=0)
stds = np.std(features, axis=0)
features_scaled = (features - means) / stds
features_scaled = np.c_[np.ones(features_scaled.shape[0]), features_scaled]

# Split into training and testing sets
train_size = int(0.8 * features_scaled.shape[0])
X_train, X_test = features_scaled[:train_size], features_scaled[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Initialize theta with zeros
theta = np.zeros(X_train.shape[1])
# Learning rate
alpha = 0.01
# Number of iterations
iterations = 1000

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/predict', methods=['GET'])
def predict():
    # Extract query parameters
    feature1 = request.args.get('feature1', type=float)
    feature2 = request.args.get('feature2', type=float)
    feature3 = request.args.get('feature3', type=float)
    
    # Create a feature array
    features = np.array([[feature1, feature2, feature3]])
    
    # Manual Standardization of input features
    features_scaled = (features - means) / stds
    features_scaled = np.c_[np.ones(features_scaled.shape[0]), features_scaled]
    
    # Make prediction
    prediction = features_scaled.dot(theta)
    
    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
