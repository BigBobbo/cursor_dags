import argparse
import pandas as pd
import logging
from pathlib import Path

# Import helper functions
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

# Import cleaning functions
from data_cleaning_functions import (
    clean_date_column,
    clean_weight_column,
    clean_race_distance,
    clean_grade_column
)

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def prepare_future_races(historical_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for future races using historical data
    """
    logging.info("Preparing future race predictions...")
    
    # Clean the historical data dates
    historical_df = clean_date_column(historical_df, 'date')
    
    # Clean and standardize the future race data
    future_df = clean_date_column(future_df, 'date')
    future_df = clean_weight_column(future_df, 'weight')
    future_df = clean_race_distance(future_df, 'race_name')
    future_df = clean_grade_column(future_df, 'grade')
    
    # Extract race number from race name for future races
    future_df['race_number'] = future_df['race_name'].str.extract(r'Race\s+(\d+)').astype(int)
    
    # Create a combined DataFrame for feature creation
    combined_df = pd.concat([
        historical_df,
        future_df.assign(clean_position=None)  # Add null position for future races
    ]).sort_values('clean_date')
    
    logging.info("Calculating historical features...")
    
    # Add race counts
    combined_df = add_previous_race_count(combined_df, 'greyhound')
    combined_df = add_previous_grade_race_count(
        combined_df,
        'greyhound',
        'clean_grade',
        'previous_same_dog_grade_count'
    )
    combined_df = add_previous_grade_race_count(
        combined_df,
        'greyhound',
        'race_grade',
        'previous_same_race_grade_count'
    )
    
    # Calculate performance metrics
    combined_df = calculate_win_rate(
        combined_df,
        'greyhound',
        'clean_position'
    )
    
    combined_df = calculate_average_position(
        combined_df,
        'greyhound',
        'clean_position'
    )
    
    # Calculate weight-related features
    combined_df = calculate_weight_change(combined_df, 'greyhound')
    combined_df = calculate_average_weight(combined_df, 'greyhound')
    
    # Calculate distance-related features
    if combined_df['clean_distance'].notna().any():
        combined_df = add_previous_race_count(
            combined_df,
            'greyhound',
            group_col='clean_distance'
        )
        combined_df = calculate_average_position(
            combined_df,
            'greyhound',
            'clean_position',
            group_col='clean_distance'
        )
        combined_df = calculate_win_rate(
            combined_df,
            'greyhound',
            'clean_position',
            group_col='clean_distance'
        )
    
    # Extract only the future races with their calculated features
    future_races_with_features = combined_df[combined_df['clean_date'].isin(future_df['clean_date'])]
    
    # Ensure all required features are present
    required_features = [
        'clean_weight',
        'clean_distance',
        'race_number',
        'previous_race',
        'previous_same_dog_grade_count',
        'previous_same_race_grade_count',
        'win_rate',
        'avg_position',
        'weight_change',
        'weight_change_pct',
        'avg_weight',
        'weight_dev_from_avg',
        'previous_race_by_clean_distance',
        'avg_position_by_clean_distance',
        'win_rate_by_clean_distance'
    ]
    
    # Fill missing features with appropriate default values
    for feature in required_features:
        if feature not in future_races_with_features.columns:
            logging.warning(f"Missing feature {feature}, filling with 0")
            future_races_with_features[feature] = 0
    
    logging.info("Feature creation complete!")
    return future_races_with_features

def main():
    parser = argparse.ArgumentParser(description='Prepare future race data with historical features')
    parser.add_argument('historical_file', type=str, help='Path to historical race CSV file')
    parser.add_argument('future_file', type=str, help='Path to future race CSV file')
    parser.add_argument('output_file', type=str, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Read input files
    logging.info(f"Reading historical data from {args.historical_file}")
    historical_df = pd.read_csv(args.historical_file, encoding='latin-1', low_memory=False)
    
    logging.info(f"Reading future race data from {args.future_file}")
    future_df = pd.read_csv(args.future_file, encoding='latin-1', low_memory=False)
    
    # Process data
    future_races_with_features = prepare_future_races(historical_df, future_df)
    
    # Save output
    logging.info(f"Saving processed future races to {args.output_file}")
    future_races_with_features.to_csv(args.output_file, index=False)
    logging.info("Processing complete!")

if __name__ == "__main__":
    main() 