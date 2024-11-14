from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the data
data = pd.read_csv('patients.csv')

# Define features (input variables) and labels (output variable)
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Add a column of ones to features to account for the intercept term
features = np.c_[np.ones(features.shape[0]), features]

# Split the data into training and testing sets
train_size = int(0.8 * features.shape[0])
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Perform linear regression using the Normal Equation
theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

@app.route('/predict', methods=['GET'])
def predict():
    # Extract query parameters
    feature1 = request.args.get('feature1', type=float)
    feature2 = request.args.get('feature2', type=float)
    feature3 = request.args.get('feature3', type=float)
    
    # Create a feature array
    features = np.array([[1, feature1, feature2, feature3]])
    
    # Make prediction
    prediction = features.dot(theta)
    
    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
