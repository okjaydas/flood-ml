import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import List, Union

def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(filepath)

def remove_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns=columns)

def engineer_road_level(df: pd.DataFrame) -> pd.DataFrame:
    wl_norm = (df['Water Level (m)'] - df['Water Level (m)'].min()) / (df['Water Level (m)'].max() - df['Water Level (m)'].min())
    el_norm = (df['Elevation (m)'] - df['Elevation (m)'].min()) / (df['Elevation (m)'].max() - df['Elevation (m)'].min())
    rf_norm = (df['Rainfall (mm)'] - df['Rainfall (mm)'].min()) / (df['Rainfall (mm)'].max() - df['Rainfall (mm)'].min())

    df['RoadLevel_today'] = 0.5 * wl_norm - 0.5 * el_norm + 0.15 * rf_norm
    df['RoadLevel_today'] = df['RoadLevel_today'].clip(0,1)

    # I will use these in tomorrow feature engineering
    df['wl_norm'] = wl_norm
    df['el_norm'] = el_norm
    df['rf_norm'] = rf_norm

    return df

def engineer_soil_moisture(df):
    humidity_norm = (df['Humidity (%)'] - df['Humidity (%)'].min()) / (df['Humidity (%)'].max() - df['Humidity (%)'].min())
    soil_encoder = LabelEncoder()
    df['SoilType_num'] = soil_encoder.fit_transform(df['Soil Type'])
    soil_norm = (df['SoilType_num'] - df['SoilType_num'].min()) / (df['SoilType_num'].max() - df['SoilType_num'].min())
    df['SoilMoisture_today'] = 0.5 * humidity_norm + 0.5 * soil_norm

    # I will use these in tomorrow feature engineering
    df['humidity_norm'] = humidity_norm
    df['soil_norm'] = soil_norm
    return df

def normalize_features(df):
    df['RoadLevel_norm'] = (df['RoadLevel_today'] - df['RoadLevel_today'].min()) / (df['RoadLevel_today'].max() - df['RoadLevel_today'].min())
    df['PopDensity_norm'] = (df['Population Density'] - df['Population Density'].min()) / (df['Population Density'].max() - df['Population Density'].min())
    df['Infrastructure_norm'] = (df['Infrastructure'] - df['Infrastructure'].min()) / (df['Infrastructure'].max() - df['Infrastructure'].min())
    return df

def encode_land_cover(df):
    landcover_map = {'Urban':0.9, 'Agricultural':0.6, 'Forest':0.2, 'Water Body':0.1, 'Desert':0.7}
    df['LandCover_code'] = df['Land Cover'].map(landcover_map)
    return df

def compute_drain_level(df):
    df['DrainLevel_today'] = 1 - (
        0.35 * df['RoadLevel_norm'] +   # HIGH road water = HIGH drain level (bad)
        0.20 * df['LandCover_code'] +   # Higher LandCover_code = poorer drainage
        0.22 * df['PopDensity_norm'] +  # Denser = less drainage
        0.23 * df['Infrastructure_norm'] #  More infra = less drainage
    )
    df['DrainLevel_today'] = df['DrainLevel_today'].clip(0,1)
    return df

# --- Tomorrow feature engineering ---
def engineer_tomorrow_features(df, config=None):
    # Default config
    params = {
        'lambda_rainfall': 0.75,  # Persistence as no forecast, in real implementation I will use external forecast
        'delta_drain': 0.35, ## decay factors
        'delta_road': 0.50,
        'delta_soil': 0.25,
        'k_drain': 0.006,   ## response to rainfall
        'a_road': 0.008,
        'p_soil': 0.004,
        'q_soil': 0.2,
        'c_drain': 0.10 ## effect of land cover on drainage
    }
    if config:
        params.update(config)
    # Use external Rainfall_forecast_mm if present
    if 'Rainfall_forecast_mm' in df.columns:
        rainfall_tomorrow = df['Rainfall_forecast_mm']
    else:
        rainfall_tomorrow = params['lambda_rainfall'] * df['Rainfall (mm)']
    df['Rainfall_tomorrow'] = rainfall_tomorrow.round(2)
    df['rf_tomorrow_norm'] = (df['Rainfall_tomorrow'] - df['Rainfall_tomorrow'].min()) / (df['Rainfall_tomorrow'].max() - df['Rainfall_tomorrow'].min())

    # DrainLevel_tomorrow
    df['DrainLevel_tomorrow'] = (
        df['DrainLevel_today'] * (1 - params['delta_drain']) +
        params['k_drain'] * df['rf_tomorrow_norm'] +
        params['c_drain'] * df['LandCover_code']
    ).clip(0, 1)

    # RoadLevel_tomorrow
    df['RoadLevel_tomorrow'] = (
        df['RoadLevel_today'] * (1 - params['delta_road']) +
        params['a_road'] * df['rf_tomorrow_norm'] +
        0.2 * df['wl_norm'] - 0.2 * df['el_norm']
    ).clip(0, 1)

    # SoilMoisture_tomorrow
    df['SoilMoisture_tomorrow'] = (
        df['SoilMoisture_today'] * (1 - params['delta_soil']) +
        params['p_soil'] * df['rf_tomorrow_norm'] +
        params['q_soil'] * df['humidity_norm'] +
        0.3 * df['soil_norm']
    ).clip(0, 1)

    return df

def drop_originals(df, columns):
    return df.drop(columns=columns)

def save_data(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to CSV file."""
    df.to_csv(filepath, index=False)

def process_flood_data_2(input_path: Path, output_path: Path) -> None:
    flood_df = load_data(input_path)
    
    # Initial column removal
    columns_to_remove = ['Temperature', 'Historical Floods']
    df = remove_columns(flood_df, columns_to_remove)
    
    # Feature engineering
    df = engineer_road_level(df)
    df = engineer_soil_moisture(df)
    df = normalize_features(df)
    df = encode_land_cover(df)
    df = compute_drain_level(df)
    df = engineer_tomorrow_features(df)
    
    # Final cleanup
    df = df.rename(columns={"Rainfall (mm)": "Rainfall_today"})
    drop_cols = [
        'River Discharge (m³/s)', 'Land Cover', 'Flood Occurred', 'Population Density', 
        'Infrastructure', 'LandCover_code', 'Water Level (m)', 'Elevation (m)', 
        'Humidity (%)', 'Soil Type', 'SoilType_num', 'RoadLevel_norm', 
        'PopDensity_norm', 'Infrastructure_norm', 'wl_norm', 'el_norm', 'rf_norm',
        'humidity_norm', 'soil_norm', 'rf_tomorrow_norm'
    ]
    df_final = drop_originals(df, drop_cols)
    
    # Save results
    save_data(df_final, output_path)

def main():
    project_root = Path(__file__).parent.parent
    
    input_path = project_root / 'data' / 'Flood-Data-Enriched.csv'
    output_path = project_root / 'data' / 'Flood-Data-Transformed.csv'
    
    process_flood_data_2(input_path, output_path)

if __name__ == "__main__":
    main()