from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/', methods=['GET'])
def home():
    return "Model Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return jsonify({'prediction': int(pred[0])})

if __name__ == '__main__':
    print(" * Serving on http://127.0.0.1:5000")
    app.run(port=5000)