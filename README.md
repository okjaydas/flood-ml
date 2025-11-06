# Flood Prediction System

A machine learning system for predicting flood risks and related environmental metrics using both classification and regression models.

## Project Overview

This system provides two main prediction capabilities, in order:
1. Regression analysis for environmental metrics prediction
2. Classification of flood risk levels

## Models

### Classification Model
- **Purpose**: Predicts flood risk levels (Severity Classification)
- **Algorithm**: Logistic Regression
  - C=75.4312
  - Solver: newton-cg
  - Multi-class: ovr
- **Scaler**: StandardScaler
  - with_mean=False
  - with_std=True
- **Outputs**: 
  - FloodRiskLevel

### Regression Model
- **Purpose**: Predicts environmental metrics for the next day
- **Outputs**:
  - DrainLevel_tomorrow
  - RoadLevel_tomorrow
  - SoilMoisture_tomorrow
  - FloodProbability

## Setup Instructions

### Environment Setup
1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate flood-classification
   ```

3. Update environment (if needed):
   ```bash
   conda env update --file environment.yml --prune
   ```

### Tranform Flood Data
python .\data-extractor\process_flood_data.py

### Training Models

1. Train the classifier model:
   ```bash
   python .\learners\train_classifier.py
   ```

2. Train the regression model:
   ```bash
   python .\learners\train_regressor.py
   ```

### Testing Models

To test the prediction functionality:
```bash
python predict_classifier.py
python predict_regressor.py
```

## API Usage

### Starting the Server
```bash
python app_flood.py
```
The API server runs on http://127.0.0.1:5000

### API Endpoints

#### 1. Regression Predictions
**Endpoint**: `/predict_regressor`  
**Method**: POST

Example Request:
```json
{
  "Rainfall_today": 200,
  "DrainLevel_today": 1,
  "RoadLevel_today": 0.2,
  "SoilMoisture_today": 0.4,
  "Rainfall_tomorrow": 0
}
```

Example Response:
```json
{
    "DrainLevel_tomorrow": 0.7197195792379709,
    "FloodProbability": 0.641632323642282,
    "RoadLevel_tomorrow": 0.16953053818919334,
    "SoilMoisture_tomorrow": 0.5381199691241345
}
```

#### 2. Classification Predictions
**Endpoint**: `/predict_classifier`  
**Method**: POST

Example Request:
```json
{
  "Rainfall_today": 1000,
  "DrainLevel_today": 1,
  "RoadLevel_today": 1,
  "SoilMoisture_today": 0,
  "Rainfall_tomorrow": 2000,
  "FloodProbability": 0.9
}
```

Example Response:
```json
{
    "FloodRiskLevel": "Severe"
}
```

## Project Structure
```
├── data/                          # Dataset files
├── data-extractor/               # Data processing scripts
├── learners/                     # Model training scripts
├── models/                       # Saved model files
├── plot/                         # Visualization scripts
├── predictor/                    # Prediction scripts
└── app_flood.py                 # Flask API server
```

## Data Files
- `combined_rain_data - for local classification.csv`: Data for classification tasks
- `combined_rain_data - for local regession.csv`: Data for regression tasks
