from flask import Flask, request, jsonify
import joblib
import pandas as pd

from predictor.predict_classifier import predict_flood
from predictor.predict_regressor import predict_regression

app = Flask(__name__)

@app.route('/predict_classifier', methods=['POST'])
def predict_classifier():
    data = request.json
    prediction = predict_flood(data)
    return jsonify(prediction)
    # return jsonify({
    #     "FloodRiskLevel": prediction[0]
    #     # "EventSeverity": prediction[1],
    #     # "AlertLevel": prediction[2]
    # })

@app.route('/predict_regressor', methods=['POST'])
def predict_regressor():
    data = request.json 
    prediction = predict_regression(data)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
