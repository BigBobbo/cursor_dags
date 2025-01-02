import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Tuple, List

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Create race_id and unique_id columns"""
    df = df.copy()
    
    # Create race_id (date + track + race_number)
    df['race_id'] = (df['clean_date'].astype(str) + '_' + 
                     df['track'].astype(str) + '_' + 
                     df['race_number'].astype(str))
    
    # Create unique_id (race_id + dog_name)
    df['unique_id'] = df['race_id'] + '_' + df['greyhound']
    
    return df

def filter_ml_features(df: pd.DataFrame, approved_features_file: str = 'approved_features.yaml') -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter DataFrame to only include approved features from YAML file.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing all columns
        approved_features_file (str): Path to YAML file containing approved feature list
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Filtered DataFrame and list of kept columns
    """
    df = df.copy()
    
    # Load approved features from YAML
    logging.info(f"Loading approved features from {approved_features_file}")
    with open(approved_features_file, 'r') as f:
        approved_features = yaml.safe_load(f)['features']
    
    logging.info(f"Found {len(approved_features)} approved features in YAML file")
    
    # Get list of available approved features
    available_features = [col for col in approved_features if col in df.columns]
    missing_features = [col for col in approved_features if col not in df.columns]
    
    if missing_features:
        logging.warning(f"Missing approved features: {missing_features}")
    logging.info(f"Found {len(available_features)} approved features in DataFrame")
    
    # Filter DataFrame to only include available approved features
    df_filtered = df[available_features]
    
    # Check data types and log details about non-numeric columns
    for col in available_features:
        dtype = df_filtered[col].dtype
        logging.debug(f"Column '{col}' has dtype: {dtype}")
    
    # Identify non-numeric columns
    non_numeric_cols = df_filtered.select_dtypes(exclude=['int64', 'float64', 'Int64']).columns
    if len(non_numeric_cols) > 0:
        logging.warning("The following columns will be dropped due to non-numeric dtype:")
        for col in non_numeric_cols:
            logging.warning(f"  - '{col}' (dtype: {df_filtered[col].dtype})")
            # Optional: show some unique values to help debugging
            unique_values = df_filtered[col].unique()[:5]  # Show first 5 unique values
            logging.warning(f"    Example values: {unique_values}")
        
        df_filtered = df_filtered.drop(columns=non_numeric_cols)
        available_features = [col for col in available_features if col not in non_numeric_cols]
    
    logging.info(f"Final feature count: {len(available_features)}")
    logging.info("Kept features:")
    for feature in available_features:
        logging.info(f"  - {feature} (dtype: {df_filtered[feature].dtype})")
    
    return df_filtered, available_features

def save_ml_data(df: pd.DataFrame, target_col: str = 'est_time', 
                output_dir: str = 'ml_data') -> None:
    """
    Prepare and save ML data files:
    - features.csv: Feature matrix
    - target.csv: Target values
    - feature_columns.yaml: List of feature columns for future use
    - identifiers.csv: race_id and unique_id columns
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create identifiers
    df = create_identifiers(df)
    
    # Split features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Extract target
    target = df[['unique_id', 'race_id', 'clean_position', target_col]]
    
    # Filter features
    features, feature_columns = filter_ml_features(df)
    
    # Save identifiers
    identifiers = df[['unique_id', 'race_id']]
    identifiers.to_csv(output_path / 'identifiers.csv', index=False)
    
    # Save features and target
    features.to_csv(output_path / 'features.csv', index=False)
    target.to_csv(output_path / 'target.csv', index=False)
    
    # Save feature columns for future use
    with open(output_path / 'feature_columns.yaml', 'w') as f:
        yaml.dump(feature_columns, f)
    
    logging.info(f"Saved ML data files to {output_path}")
    logging.info(f"Number of features: {len(feature_columns)}")
    logging.info(f"Feature columns: {feature_columns}")

def main():
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Prepare data for ML model')
    parser.add_argument('input_file', type=str, help='Path to processed race data CSV')
    parser.add_argument('--output-dir', type=str, default='ml_data',
                       help='Directory to save ML data files')
    parser.add_argument('--target-col', type=str, default='est_time',
                       help='Name of target column')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Read input data
    logging.info(f"Reading processed data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Save ML data
    save_ml_data(df, args.target_col, args.output_dir)
    
    logging.info("Processing complete!")

if __name__ == "__main__":
    main() 