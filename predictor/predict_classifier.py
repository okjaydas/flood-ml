import pandas as pd
import joblib
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

def load_model(model_path=None):
    if model_path is None:
        model_path = os.getenv('CLASSIFIER_MODEL_PATH')
        if model_path:
            model_path = str(PROJECT_ROOT / model_path)
        else:
            model_path = str(PROJECT_ROOT / 'models' / 'flood_classifier_voting.joblib')
    model_dict = joblib.load(model_path)
    return model_dict['scaler'], model_dict['classifier']

def predict_flood(data_row):
    scaler, clf = load_model()
    
    if isinstance(data_row, pd.Series):
        data_row = data_row.values.reshape(1, -1)
    elif isinstance(data_row, dict):
        data_row = pd.DataFrame([data_row]).values
    elif isinstance(data_row, list):
        data_row = np.array(data_row).reshape(1, -1)
    
    data_scaled = scaler.transform(data_row)
    prediction = clf.predict(data_scaled)
    
    result = {
        'FloodRiskLevel': prediction[0][0]
    }
    
    return result

def predict_batch(data_df):
    scaler, clf = load_model()
    
    features = [
    'Rainfall_today', 'DrainLevel_today', 'RoadLevel_today',
    'SoilMoisture_today', 'Rainfall_tomorrow', 'FloodProbability'
    ]

    # Create DataFrame for a single row, with correct column order
    data_row_df = pd.DataFrame([sample_data], columns=features)

    data_scaled = scaler.transform(data_row_df)
    predictions = clf.predict(data_scaled)
    
    results_df = pd.DataFrame(
        predictions,
        columns=['FloodRiskLevel', 'EventSeverity', 'AlertLevel']
    )
    
    return results_df

if __name__ == "__main__":
    sample_data = {
        'Rainfall_today': 780.02,
        'DrainLevel_today': 1.75,
        'RoadLevel_today': 1.41,
        'SoilMoisture_today': 20.3,
        'Rainfall_tomorrow': 369.48,
        'FloodProbability': 2.60  
    }

    
    prediction = predict_flood(sample_data)
    print("Prediction:", prediction)
