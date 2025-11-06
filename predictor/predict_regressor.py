import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

def load_regressor(model_path=None):
    if model_path is None:
        model_path = os.getenv('REGRESSOR_MODEL_PATH')
        if model_path:
            model_path = str(PROJECT_ROOT / model_path)
        else:
            model_path = str(PROJECT_ROOT / 'models' / 'flood_regressor.joblib')
    model_dict = joblib.load(model_path)
    # svd may be optional in some saved models
    scaler = model_dict.get("scaler")
    svd = model_dict.get("svd")
    regressor = model_dict.get("regressor")
    features = model_dict.get("features")
    targets = model_dict.get("targets")
    return scaler, svd, regressor, features, targets

def predict_regression(input_data):
    scaler, svd, regressor, features, targets = load_regressor()

    # Accepts dict, Series or list
    if isinstance(input_data, dict):
        X = pd.DataFrame([input_data], columns=features)
    elif isinstance(input_data, pd.Series):
        X = pd.DataFrame([input_data.values], columns=features)
    elif isinstance(input_data, list):
        X = pd.DataFrame([input_data], columns=features)
    else:
        raise ValueError("Input must be dict, list, pd.Series")

    X_scaled = scaler.transform(X)
    # If svd is present, use it; otherwise pass scaled features directly to regressor
    if svd is not None:
        X_svd = svd.transform(X_scaled)
        Y_pred = regressor.predict(X_svd)
    else:
        Y_pred = regressor.predict(X_scaled)
    results = dict(zip(targets, Y_pred[0]))
    return results

def predict_batch(dataframe):
    scaler, svd, regressor, features, targets = load_regressor()
    X = dataframe[features]
    X_scaled = scaler.transform(X)
    # If svd is present, use it; otherwise use scaled features directly
    if svd is not None:
        X_svd = svd.transform(X_scaled)
        Y_pred = regressor.predict(X_svd)
    else:
        Y_pred = regressor.predict(X_scaled)
    return pd.DataFrame(Y_pred, columns=targets)

if __name__ == "__main__":
    test_cases = [
        {'Rainfall_today': 350, 'DrainLevel_today': 0.1, 'RoadLevel_today': 0.5, 'SoilMoisture_today': 0.2, 'Rainfall_tomorrow': 80},
        {'Rainfall_today': 1000, 'DrainLevel_today': 0.7, 'RoadLevel_today': 0.7, 'SoilMoisture_today': 0.8, 'Rainfall_tomorrow': 300},
        {'Rainfall_today': 0, 'DrainLevel_today': 0, 'RoadLevel_today': 0, 'SoilMoisture_today': 0, 'Rainfall_tomorrow': 0}
    ]
    for sd in test_cases:
        pred = predict_regression(sd)
        print("\n-----------------------\n")
        print("Prediction for input:", sd)
        print(pred)
        print("\n-----------------------\n")
