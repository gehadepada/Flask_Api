from flask import Flask, request, jsonify
import joblib
import pandas as pd
import gdown

app = Flask(__name__)

# Google Drive file ID
file_id = '1yY2fmAuuzfelOZpJkMGzIL1PRLKiu6rs'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the model file
output = 'linear_regression_model.pkl'
gdown.download(url, output, quiet=False)

# Load the trained model
model = joblib.load(output)

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

if __name__ == '__main__':
    app.run(debug=True)

