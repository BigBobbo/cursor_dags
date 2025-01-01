import argparse
import pandas as pd
import yaml
from pathlib import Path
import logging

# Import your helper functions
from race_helper_functions import (
    add_previous_race_count,
    add_previous_grade_race_count,
    calculate_win_rate,
    calculate_average_position,
    calculate_weight_change,
    calculate_average_weight,
    calculate_distance_performance,
    calculate_preferred_distance
)

# Import your cleaning functions (assuming they exist in data_cleaning_functions.py)
from data_cleaning_functions import (
    clean_date_column,
    clean_position_column,
    clean_grade_column,
    clean_going_column,
    clean_weight_column,
    clean_race_distance,
    clean_grade_column,
    clean_est_time_column,
    clean_race_name_column
)

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_parameters(param_file: str) -> dict:
    """Load parameters from YAML file"""
    with open(param_file, 'r') as f:
        return yaml.safe_load(f)

def process_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Process the data according to the parameters"""
    logging.info("Starting data processing...")
    
    # Get column mappings
    col_map = params['columns']
    
    # Apply cleaning steps first
    if params['cleaning_steps']['clean_date']:
        logging.info("Cleaning date column...")
        df = clean_date_column(df, col_map['date'])
    
    if params['cleaning_steps']['clean_grade']:
        logging.info("Cleaning grade column...")
        df = clean_grade_column(df, col_map['grade'])
    
    if params['cleaning_steps']['clean_position']:
        logging.info("Cleaning position column...")
        df = clean_position_column(df, col_map['position'])
    
    if params['cleaning_steps']['clean_weight']:
        logging.info("Cleaning weight column...")
        df = clean_weight_column(df, col_map['weight'])
    
    if params['cleaning_steps']['clean_distance']:
        logging.info("Cleaning race distance...")
        df = clean_race_distance(df, col_map['race_name'])
    
    if params['cleaning_steps']['clean_distance']:
        logging.info("Cleaning going column...")
        df = clean_going_column(df, col_map['going'])

    if params['cleaning_steps']['clean_est_time']:
        logging.info("Cleaning estimated time...")
        df = clean_est_time_column(df, col_map['est_time'])
    
    if params['cleaning_steps']['clean_race_name']:
        logging.info("Extracting race number and grade from race name...")
        df = clean_race_name_column(df, col_map['race_name'])
    
    
    # Only remove invalid positions if specified in parameters
    if params.get('remove_invalid_positions', False):
        initial_count = len(df)
        df = df[df['clean_est_time'].notna()]
        rows_dropped = initial_count - len(df)
        logging.info(f"Removed {rows_dropped} rows with invalid positions")
        logging.info(f"Remaining rows: {len(df)}")
    
    
    # Apply feature engineering using cleaned columns
    if params['feature_engineering']['previous_race_count']:
        logging.info("Adding previous race count...")
        df = add_previous_race_count(df, col_map['greyhound'])
    
    if params['feature_engineering']['previous_grade_race_count']:
        logging.info("Adding previous grade race count...")
        df = add_previous_grade_race_count(
            df, 
            col_map['greyhound'], 
            'clean_grade',
            'previous_grade_race_count'
        )
    
    if params['feature_engineering']['win_rate']['enabled']:
        logging.info("Calculating win rates...")
        df = calculate_win_rate(
            df,
            col_map['greyhound'],
            'clean_position',  # Use cleaned position column
            params['feature_engineering']['win_rate']['lookback']
        )
    
    if params['feature_engineering']['average_position']['enabled']:
        logging.info("Calculating average positions...")
        df = calculate_average_position(
            df,
            col_map['greyhound'],
            'clean_position',  # Use cleaned position column
            params['feature_engineering']['average_position']['lookback']
        )
    
    if params['feature_engineering']['weight_analysis']['enabled']:
        logging.info("Calculating weight-related features...")
        if df['clean_weight'].notna().any():  # Only process if we have weight data
            df = calculate_weight_change(df, col_map['greyhound'])
            df = calculate_average_weight(
                df,
                col_map['greyhound'],
                lookback=params['feature_engineering']['weight_analysis']['lookback']
            )
    
    if params['feature_engineering']['distance_analysis']['enabled']:
        logging.info("Calculating distance-related features...")
        if df['clean_distance'].notna().any():
            df = calculate_distance_performance(
                df,
                col_map['greyhound'],
                lookback=params['feature_engineering']['distance_analysis']['lookback']
            )
            df = calculate_preferred_distance(
                df,
                col_map['greyhound'],
                min_races=params['feature_engineering']['distance_analysis']['min_races']
            )
    
    return df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process dog race data')
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    parser.add_argument('params_file', type=str, help='Path to parameters YAML file')
    parser.add_argument('output_file', type=str, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load parameters
    logging.info(f"Loading parameters from {args.params_file}")
    params = load_parameters(args.params_file)
    
    # Read input data
    logging.info(f"Reading data from {args.input_file}")
    df = pd.read_csv(args.input_file, encoding='latin-1')
    
    # Process data
    df_processed = process_data(df, params)
    
    # Save output
    logging.info(f"Saving processed data to {args.output_file}")
    df_processed.to_csv(args.output_file, index=False)
    logging.info("Processing complete!")

if __name__ == "__main__":
    main() 