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

def filter_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns that cannot be used in ML model and return filtered DataFrame
    and list of kept columns.
    """
    df = df.copy()
    
    # Columns to drop (add any additional columns that shouldn't be used for ML)
    drop_columns = [
        # Identifier columns
        'race_id', 'unique_id', 'greyhound', 'track', 'race_number',
        
        # Date columns (already used in race_id)
        'date', 'clean_date',
        
        # Text columns
        'race_name', 'grade',
        
        # Original columns that have been cleaned
        'pos', 'wt',
        
        # Target column
        'est_time'
    ]
    
    # Drop columns that exist in the DataFrame
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    df_filtered = df.drop(columns=columns_to_drop)
    
    # Get remaining columns (features to use in ML)
    kept_columns = df_filtered.columns.tolist()
    
    # Remove any remaining non-numeric columns
    non_numeric_cols = df_filtered.select_dtypes(exclude=['int64', 'float64', 'Int64']).columns
    if len(non_numeric_cols) > 0:
        logging.warning(f"Removing non-numeric columns: {non_numeric_cols.tolist()}")
        df_filtered = df_filtered.drop(columns=non_numeric_cols)
        kept_columns = [col for col in kept_columns if col not in non_numeric_cols]
    
    return df_filtered, kept_columns

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
    target = df[['unique_id', 'race_id', target_col]]
    
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