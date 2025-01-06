# Greyhound Race Prediction System

This system processes historical greyhound race data and trains machine learning models to predict race outcomes. The pipeline consists of several Python scripts that handle data processing, feature engineering, and model training.

## System Requirements

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, xgboost, pyyaml

Install dependencies:
bash
```
pip install pandas numpy scikit-learn xgboost pyyaml
```

## Pipeline Overview

1. **Data Processing** (`process_race_data.py`): Cleans raw race data and engineers features
2. **ML Data Preparation** (`prepare_ml_data.py`): Prepares processed data for machine learning
3. **Model Training** (`train_model.py`): Trains an XGBoost model on the prepared data
4. **Future Race Preparation** (`prepare_future_races.py`): Processes upcoming races for prediction

## Usage

### 1. Process Raw Race Data

bash
```
python process_race_data.py input_races.csv processing_parameters.yaml processed_races.csv
```
This script:
- Cleans and standardizes raw race data
- Engineers features based on historical performance
- Uses parameters defined in `processing_parameters.yaml`

### 2. Prepare Data for Machine Learning
bash
```
python prepare_ml_data.py processed_races.csv --output-dir ml_data --features-file approved_features.yaml
```
This script:
- Filters features based on `approved_features.yaml`
- Splits data into features and target variables
- Creates train/test datasets
- Saves prepared data to the specified output directory

### 3. Train the Model
bash
```
python train_model.py --data-dir ml_data --config approved_features.yaml --output-dir models
```
This script:
- Loads prepared ML data
- Performs hyperparameter tuning using RandomizedSearchCV
- Trains an XGBoost model
- Saves the trained model and performance metrics

### 4. Prepare Future Races

bash
```
python prepare_future_races.py historical_races.csv future_races.csv prepared_future_races.csv
```
This script:
- Processes upcoming race data
- Engineers features using historical data
- Prepares data in the format expected by the trained model

## Configuration Files

### processing_parameters.yaml
Controls data cleaning and feature engineering parameters:
- Cleaning steps to apply
- Feature engineering options
- Column mappings
- Filtering options

### approved_features.yaml
Defines:
- Features to use in the ML model
- Model hyperparameters
- Training configuration
- Feature median values for handling missing data

## Directory Structure

.
├── process_race_data.py
├── prepare_ml_data.py
├── train_model.py
├── prepare_future_races.py
├── race_helper_functions.py
├── data_cleaning_functions.py
├── processing_parameters.yaml
├── approved_features.yaml
├── ml_data/
│ ├── features.csv
│ ├── target.csv
│ └── feature_columns.yaml
└── models/
└── model_[timestamp].pkl

## Helper Modules

### race_helper_functions.py
Contains functions for:
- Calculating win rates and average positions
- Processing weight and distance data
- Computing historical performance metrics

### data_cleaning_functions.py
Provides functions for:
- Cleaning date formats
- Standardizing positions and weights
- Processing race distances and grades

## Notes

- All scripts include logging to track progress and debug issues
- Use the `--help` flag with any script to see available options
- The system handles missing data and invalid values gracefully
- Model training includes cross-validation and hyperparameter tuning
- Future race predictions require historical data for feature engineering

## License

[Add your license information here]

## TODO
- [ ] update the target function to use rase level targets
- [ ] Validate the current data set
- [ ] Restructure where the ml data is stored for ease of recordkeeping
- [ ] Clean position has some values 10x the actual position
- [ ] Grade appears to be info on the race just ran, should not be used for future races
- [ ] Trap is not part of the future data, should add
- [ ] Add model diagnostics