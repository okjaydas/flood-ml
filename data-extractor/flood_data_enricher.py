import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def normalize_features(df: pd.DataFrame) -> Tuple[pd.Series, ...]:
    rainfall = df['Rainfall (mm)']
    humidity = df['Humidity (%)']
    discharge = df['River Discharge (m³/s)']
    water_level = df['Water Level (m)']
    historical_floods = df['Historical Floods']

    # Normalize each feature to [0,1] range
    rainfall_n = (rainfall - rainfall.min()) / (rainfall.max() - rainfall.min())
    humidity_n = (humidity - humidity.min()) / (humidity.max() - humidity.min())
    discharge_n = (discharge - discharge.min()) / (discharge.max() - discharge.min())
    water_level_n = (water_level - water_level.min()) / (water_level.max() - water_level.min())
    historical_n = historical_floods.astype(float)

    return rainfall_n, humidity_n, discharge_n, water_level_n, historical_n

def calculate_flood_probability(
    normalized_features: Tuple[pd.Series, ...],
    flood_occurred: pd.Series,
    weights: dict[str, float]
) -> pd.Series:
    rainfall_n, humidity_n, discharge_n, water_level_n, historical_n = normalized_features
    
    # Compute base probability
    prob_base = (
        rainfall_n * weights['rainfall'] +
        humidity_n * weights['humidity'] +
        discharge_n * weights['discharge'] +
        water_level_n * weights['water_level'] +
        historical_n * weights['historical']
    )
    
    # Calibrate probabilities
    score_boost = np.where(flood_occurred.astype(bool), 0.3, -0.3)
    return np.clip(prob_base + score_boost, 0.01, 0.99)

def calculate_risk_level(flood_probability: pd.Series) -> pd.Series:
    risk_thresholds = [0.00, 0.33, 0.60, 0.80, 1.0]
    risk_labels = ['Low', 'Moderate', 'High', 'Severe']
    return pd.cut(flood_probability, bins=risk_thresholds, labels=risk_labels, include_lowest=True)

def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Rainfall (mm)'] = df['Rainfall (mm)'].apply(lambda x: 0 if x < 5 else round(x, 2))
    df['Temperature'] = df['Temperature'].round(2)
    df['Humidity (%)'] = df['Humidity (%)'].round(2)
    df['River Discharge (m³/s)'] = df['River Discharge (m³/s)'].round(2)
    df['Water Level (m)'] = df['Water Level (m)'].round(2)
    return df

def process_flood_data(input_path: Path, output_path: Path) -> None:
    weights = {
        'rainfall': 0.35,
        'humidity': 0.08,
        'discharge': 0.23,
        'water_level': 0.23,
        'historical': 0.15
    }

    # Load and process data
    df = load_data(input_path)
    normalized_features = normalize_features(df)
    
    # Calculate flood probability and risk level
    flood_probability = calculate_flood_probability(
        normalized_features,
        df['Flood Occurred'],
        weights
    )
    flood_risk_level = calculate_risk_level(flood_probability)
    
    out_df = round_numeric_columns(df)
    out_df['FloodProbability'] = flood_probability
    out_df['FloodRiskLevel'] = flood_risk_level
    
    out_df.to_csv(output_path, index=False)

def main():
    # Get the project root directory (two levels up from this script)
    project_root = Path(__file__).parent.parent
    
    input_path = project_root / 'data' / 'Flood-Data.csv'
    output_path = project_root / 'data' / 'Flood-Data-Enriched.csv'
    
    process_flood_data(input_path, output_path)

if __name__ == '__main__':
    main()