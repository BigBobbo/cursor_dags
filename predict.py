import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

def setup_logging() -> None:
    """Configure logging to both file and console"""
    log_file = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str):
    """Load the pickled model"""
    logging.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def load_data(features_path: str, identifier_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and identifier data"""
    logging.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    
    logging.info(f"Loading identifiers from {identifier_path}")
    identifier_df = pd.read_csv(identifier_path)
    
    if len(features_df) != len(identifier_df):
        raise ValueError("Features and identifier files have different numbers of rows")
    
    return features_df, identifier_df

def make_predictions(model, features_df: pd.DataFrame) -> pd.Series:
    """Make predictions using the loaded model"""
    logging.info("Making predictions...")
    try:
        predictions = model.predict(features_df)
        return pd.Series(predictions, name='predicted_time')
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def save_results(identifier_df: pd.DataFrame, predictions: pd.Series, output_path: str):
    """Save results by joining predictions with identifier data"""
    logging.info("Joining predictions with identifier data...")
    
    # Add predictions to identifier DataFrame
    results_df = identifier_df.copy()
    results_df['predicted_time'] = predictions
    
    # Save to file
    output_path = Path(output_path)
    output_file = output_path.parent / f"{output_path.stem}_results{output_path.suffix}"
    
    logging.info(f"Saving results to {output_file}")
    results_df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pickled model file')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features CSV file')
    parser.add_argument('--identifier', type=str, required=True,
                       help='Path to identifier CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Load data
        features_df, identifier_df = load_data(args.features, args.identifier)
        
        # Make predictions
        predictions = make_predictions(model, features_df)
        
        # Save results
        save_results(identifier_df, predictions, args.identifier)
        
        logging.info("Prediction process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during prediction process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 