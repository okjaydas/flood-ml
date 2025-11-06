from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load classifier
clf_dict = joblib.load("flood_classifier_voting.joblib")
scaler_clf = clf_dict["scaler"]
classifier = clf_dict["classifier"]
features_clf = ["Rainfall_today", "DrainLevel_today", "RoadLevel_today", "SoilMoisture_today", "Rainfall_tomorrow", "FloodProbability"]

# Load regressor
reg_dict = joblib.load("flood_regressor.joblib")
scaler_reg = reg_dict["scaler"]
svd_reg = reg_dict["svd"]
regressor = reg_dict["regressor"]
features_reg = reg_dict["features"]
targets_reg = reg_dict["targets"]

@app.route('/predict_classifier', methods=['POST'])
def predict_classifier():
    data = request.json
    # Ensure incoming data covers all features
    input_df = pd.DataFrame([data], columns=features_clf)
    X_scaled = scaler_clf.transform(input_df)
    prediction = classifier.predict(X_scaled)[0]
    return jsonify({
        "FloodRiskLevel": prediction[0]
        # "EventSeverity": prediction[1],
        # "AlertLevel": prediction[2]
    })

@app.route('/predict_regressor', methods=['POST'])
def predict_regressor():
    data = request.json
    input_df = pd.DataFrame([data], columns=features_reg)
    X_scaled = scaler_reg.transform(input_df)
    X_svd = svd_reg.transform(X_scaled)
    prediction = regressor.predict(X_svd)[0]
    result = {targets_reg[i]: float(prediction[i]) for i in range(len(targets_reg))}
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
