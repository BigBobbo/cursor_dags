import pandas as pd
import numpy as np
import yaml
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import uniform, randint

def setup_logging(output_dir: Path) -> None:
    """Configure logging to both file and console"""
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(data_dir: Path, config: Dict[str, Any], config_file: Path, is_training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and target data"""
    features = pd.read_csv(data_dir / 'features.csv')
    target = pd.read_csv(data_dir / 'target.csv')
    
    # Filter features based on approved features list
    approved_features = config.get('features', [])
    if not approved_features:
        raise ValueError("No approved features found in config file")
    
    missing_features = [f for f in approved_features if f not in features.columns]
    if missing_features:
        raise ValueError(f"Following approved features are missing from data: {missing_features}")
    
    # Keep only approved features
    features = features[approved_features]
    
    # Log initial data info
    logging.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logging.info(f"Using approved features: {', '.join(approved_features)}")
    logging.info(f"Number of unique races: {target['race_id'].nunique()}")
    
    # Check for inf and nan values
    inf_cols = features.columns[features.isin([np.inf, -np.inf]).any()].tolist()
    nan_cols = features.columns[features.isna().any()].tolist()
    
    if inf_cols:
        logging.warning(f"Columns containing infinite values: {inf_cols}")
    if nan_cols:
        logging.warning(f"Columns containing NaN values: {nan_cols}")
    
    # Replace infinite values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    if is_training:
        # Calculate median values during training
        median_values = {}
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                median_values[col] = float(median_val)  # Convert to float for YAML serialization
                features[col] = features[col].fillna(median_val)
                logging.info(f"Filled NaN values in {col} with median value: {median_val}")
        
        # Update config with median values
        config['feature_medians'] = median_values
        
        # Save updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        # Use stored median values during prediction
        if 'feature_medians' not in config:
            raise ValueError("No stored median values found in config file")
        
        for col, median_val in config['feature_medians'].items():
            if col in features.columns:
                features[col] = features[col].fillna(median_val)
                logging.info(f"Filled NaN values in {col} with stored median value: {median_val}")
    
    # Verify data is clean
    assert not features.isin([np.inf, -np.inf]).any().any(), "Infinite values still present in features"
    assert not features.isna().any().any(), "NaN values still present in features"
    
    return features, target

def load_config(config_file: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_cv_splitter(cv_strategy: str, n_folds: int, target_df: pd.DataFrame) -> Any:
    """Create cross-validation splitter based on strategy"""
    if cv_strategy == 'random':
        return n_folds
    elif cv_strategy == 'random_races':
        return GroupKFold(n_splits=n_folds)
    elif cv_strategy == 'chronological':
        return TimeSeriesSplit(n_splits=n_folds)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")

def create_parameter_grid(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create parameter grid for RandomizedSearchCV"""
    hp = config['hyperparameters']
    return {
        'n_estimators': randint(hp['n_estimators']['min'], hp['n_estimators']['max']),
        'max_depth': randint(hp['max_depth']['min'], hp['max_depth']['max']),
        'learning_rate': uniform(hp['learning_rate']['min'], 
                               hp['learning_rate']['max'] - hp['learning_rate']['min']),
        'subsample': uniform(hp['subsample']['min'], 
                           hp['subsample']['max'] - hp['subsample']['min']),
        'colsample_bytree': uniform(hp['colsample_bytree']['min'], 
                                  hp['colsample_bytree']['max'] - hp['colsample_bytree']['min']),
        'min_child_weight': randint(hp['min_child_weight']['min'], hp['min_child_weight']['max'])
    }

def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, predictions),
        'rmse': mean_squared_error(y_test, predictions, squared=False),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }

def save_model(model: xgb.XGBRegressor, output_dir: Path) -> None:
    """Save trained model to file"""
    model_file = output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {model_file}")

def train_model(data_dir: str, config_file: str, output_dir: str):
    """Main training function"""
    # Setup paths
    data_path = Path(data_dir)
    config_path = Path(config_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_path)
    
    # Load configuration
    config = load_config(config_path)
    logging.info("Configuration loaded")
    
    # Load data with config and config_file path
    features_df, target_df = load_data(data_path, config, config_path, is_training=True)
    
    # Validate data
    if features_df.empty:
        raise ValueError("Features DataFrame is empty after cleaning")
    if len(features_df) != len(target_df):
        raise ValueError(f"Mismatched lengths: features ({len(features_df)}) vs target ({len(target_df)})")
    
    # Split features and target
    X = features_df
    y = target_df['est_time']
    
    # Create train/test split
    if config['training']['cv_strategy'] == 'chronological':
        # Sort by date for chronological split
        sort_idx = target_df['race_id'].argsort()
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_seed']
    )
    
    # Create CV splitter
    cv_splitter = create_cv_splitter(
        config['training']['cv_strategy'],
        config['training']['n_folds'],
        target_df
    )
    
    # Initialize model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=config['training']['random_seed']
    )
    
    # Create parameter grid
    param_grid = create_parameter_grid(config)
    
    # Create RandomizedSearchCV with additional parameters
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=config['search']['n_iter'],
        cv=cv_splitter,
        n_jobs=config['search']['n_jobs'],
        verbose=2,
        random_state=config['training']['random_seed'],
        scoring='neg_mean_squared_error',
        pre_dispatch='2*n_jobs',
        error_score='raise',
        return_train_score=False
    )
    
    # Handle missing values in target variable
    groups = None  # Initialize groups variable
    if 'target' in config:
        missing_value = config['target'].get('missing_value', '-')
        missing_strategy = config['target'].get('missing_strategy', 'drop')
        
        # Replace '-' with NaN
        y_train = y_train.replace(missing_value, np.nan)
        
        if missing_strategy == 'drop':
            # Get indices of non-null values
            valid_indices = y_train[~y_train.isna()].index
            # Filter both X and y
            X_train = X_train.loc[valid_indices]
            y_train = y_train.loc[valid_indices]
            
            # If using race_id groups, update them as well
            if config['training']['cv_strategy'] == 'random_races':
                groups = target_df.loc[X_train.index, 'race_id']
                
        elif missing_strategy == 'fill' and config['target'].get('fill_value') is not None:
            y_train = y_train.fillna(config['target']['fill_value'])
    
    # Fit model
    logging.info("Starting model training...")
    if config['training']['cv_strategy'] == 'random_races':
        if groups is None:  # If groups wasn't set in missing value handling
            groups = target_df.loc[X_train.index, 'race_id']
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    
    # Log best parameters
    logging.info("Best parameters found:")
    for param, value in search.best_params_.items():
        logging.info(f"{param}: {value}")
    
    # Evaluate model
    metrics = evaluate_model(search.best_estimator_, X_test, y_test)
    logging.info("Model performance on test set:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")
    
    # Save model
    save_model(search.best_estimator_, output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train XGBoost model on race data')
    parser.add_argument('--data-dir', type=str, default='ml_data',
                       help='Directory containing features.csv and target.csv')
    parser.add_argument('--config', type=str, default='approved_features.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained model and logs')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.config, args.output_dir)

if __name__ == "__main__":
    main() 