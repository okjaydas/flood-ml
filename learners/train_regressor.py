import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Features and targets
features = [
    "Rainfall_today", "DrainLevel_today", "RoadLevel_today", "SoilMoisture_today", "Rainfall_tomorrow"
]
targets = [
    "DrainLevel_tomorrow", "RoadLevel_tomorrow", "SoilMoisture_tomorrow", "FloodProbability"
]

def print_model_details(regressor):
    print("\n ======================= \n ")
    print("Regressor Coefficients:")
    # regressor is MultiOutputRegressor
    print("Underlying regressors:", regressor.estimators_)
    for i, est in enumerate(regressor.estimators_):
        print(f"Target: {targets[i]}")
        print("Coefficients:")
        for feat_idx, coef in enumerate(est.coef_):
            print(f"  {features[feat_idx]}: {coef}")
        print("Intercept:", est.intercept_)
        print("-----")

def print_randomforest_details(rf_model):
    for i, est in enumerate(rf_model.estimators_):
        print(f"Target: {targets[i]}")
        print("Feature importances:", est.feature_importances_)
        print("-----")


def train_regressor(data_path, model_regressor_path, model_randomforest_path):
    # Load data
    df = pd.read_csv(data_path)
    
    X = df[features]
    Y = df[targets]

    # Train/Test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler(with_mean=False, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dimensionality Reduction
    # print(X_train_scaled.shape[1])
    # n_components = max(5, int(0.36 * X_train_scaled.shape[1]))  
    # print(n_components)
    # svd = TruncatedSVD(n_components=n_components)
    # X_train_svd = svd.fit_transform(X_train_scaled)
    # X_test_svd = svd.transform(X_test_scaled)
    
    print(X_train.describe())
    print(Y_train.describe())
    # print(X_train_svd[:10])
    # Regression Model
    base = ElasticNet(alpha=0.001, l1_ratio=0.01,  max_iter=5000, random_state=10, fit_intercept=False)
    model = MultiOutputRegressor(base)
    model.fit(X_train_scaled, Y_train)

    # Evaluate
    Y_pred = model.predict(X_test_scaled)
    print("ElasticNet Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
    print("ElasticNet R2 Score:", r2_score(Y_test, Y_pred, multioutput='uniform_average'))

    # Save Pipeline
    joblib.dump(
        {
            "scaler": scaler,
            # "svd": svd,
            "regressor": model,
            "features": features,
            "targets": targets
        },
        model_regressor_path
    )
    print_model_details(model)

    # RandomForest model
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model = MultiOutputRegressor(rf_base)
    rf_model.fit(X_train_scaled, Y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    print("RandomForest MSE:", mean_squared_error(Y_test, rf_pred))
    print("RandomForest R2:", r2_score(Y_test, rf_pred, multioutput='uniform_average'))

    # Save Pipeline
    joblib.dump(
        {
            "scaler": scaler,
            # "svd": svd,
            "regressor": rf_model,
            "features": features,
            "targets": targets
        },
        model_randomforest_path
    )
    print_randomforest_details(rf_model)

    print(f"Regressor saved to {model_regressor_path} and {model_randomforest_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    transformed_data_path = project_root / 'data' / 'Flood-Data-Transformed.csv'
    model_regressor_path = project_root / 'models' / 'flood_regressor.joblib'
    model_randomforest_path = project_root / 'models' / 'flood_regressor_randomforest.joblib'

    train_regressor(transformed_data_path, model_regressor_path, model_randomforest_path)
