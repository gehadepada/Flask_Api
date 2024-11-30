from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and prepare the data
data = pd.read_csv('patients.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

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
    # Create a feature array with a leading 1 for the intercept
    features = np.array([[1, feature1, feature2, feature3]])
    
    # Make prediction using the trained model (theta)
    prediction = features.dot(theta)
    
    # Return the prediction (a single value)
    return prediction[0]

# API route to make predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Parse the JSON body
        data = request.get_json()

        # Extract features from the request (ensure they exist in the body)
        feature1 = data['feature1']
        feature2 = data['feature2']
        feature3 = data['feature3']

        # Make prediction using the features
        prediction = predict(feature1, feature2, feature3)

        # Return the prediction in the response
        return jsonify({"prediction": prediction})

    except Exception as e:
        # Handle errors and return a bad request response
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
