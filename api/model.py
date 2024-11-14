import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the data

data = pd.read_excel('C:/Users/VIP/Desktop/patients.xlsx')

# Define features (input variables) and labels (output variable)
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')
