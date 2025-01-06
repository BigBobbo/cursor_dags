import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap

def setup_logging(output_dir: Path) -> None:
    """Configure logging to both file and console"""
    log_file = output_dir / f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_model_and_data(model_path: str, data_dir: str, config_file: str) -> tuple:
    """Load the trained model, test data, and configuration"""
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reference the load_data function from train_model.py
    from train_model import load_data
    features_df, target_df = load_data(Path(data_dir), config, Path(config_file), is_training=False)
    
    return model, features_df, target_df, config

def plot_feature_importance(model, feature_names: list, output_dir: Path):
    """Plot feature importance using built-in XGBoost importance scores"""
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()
    
    # Log feature importance
    logging.info("\nFeature Importance:")
    for idx, row in importance_df.sort_values('importance', ascending=False).iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

def analyze_shap_values(model, features_df: pd.DataFrame, output_dir: Path):
    """Generate SHAP value analysis"""
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features_df, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png')
    plt.close()
    
    # Feature interaction plot for top feature
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(0, shap_values, features_df, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_interaction.png')
    plt.close()

def analyze_predictions(model, features_df: pd.DataFrame, target_df: pd.DataFrame, output_dir: Path):
    """Analyze model predictions and generate performance metrics"""
    predictions = model.predict(features_df)
    
    # Calculate metrics
    mse = mean_squared_error(target_df['est_time'], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_df['est_time'], predictions)
    r2 = r2_score(target_df['est_time'], predictions)
    
    # Log metrics
    logging.info("\nModel Performance Metrics:")
    logging.info(f"Mean Squared Error: {mse:.4f}")
    logging.info(f"Root Mean Squared Error: {rmse:.4f}")
    logging.info(f"Mean Absolute Error: {mae:.4f}")
    logging.info(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 10))
    plt.scatter(target_df['est_time'], predictions, alpha=0.5)
    plt.plot([target_df['est_time'].min(), target_df['est_time'].max()], 
             [target_df['est_time'].min(), target_df['est_time'].max()], 
             'r--', lw=2)
    plt.xlabel('Actual Times')
    plt.ylabel('Predicted Times')
    plt.title('Actual vs Predicted Race Times')
    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_predicted.png')
    plt.close()
    
    # Plot prediction error distribution
    errors = predictions - target_df['est_time']
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png')
    plt.close()

def analyze_model_parameters(model):
    """Analyze and log model parameters"""
    logging.info("\nModel Parameters:")
    params = model.get_params()
    for param, value in params.items():
        logging.info(f"{param}: {value}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained XGBoost model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pickled model file')
    parser.add_argument('--data-dir', type=str, default='ml_data',
                       help='Directory containing features.csv and target.csv')
    parser.add_argument('--config', type=str, default='approved_features.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='model_evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_path)
    
    try:
        # Load model and data
        model, features_df, target_df, config = load_model_and_data(
            args.model, args.data_dir, args.config
        )
        
        # Run evaluations
        logging.info("Starting model evaluation...")
        
        # Analyze model parameters
        analyze_model_parameters(model)
        
        # Plot feature importance
        plot_feature_importance(model, features_df.columns, output_path)
        
        # Generate SHAP analysis
        analyze_shap_values(model, features_df, output_path)
        
        # Analyze predictions
        analyze_predictions(model, features_df, target_df, output_path)
        
        logging.info(f"\nEvaluation complete! Results saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 