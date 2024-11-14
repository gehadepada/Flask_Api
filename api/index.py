from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(_name_)

# Load the trained model
model = joblib.load('/content/drive/My Drive/linear_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    # Make prediction
    prediction = model.predict(df)
    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

if _name_ == '_main_':
    app.run(debug=True)
