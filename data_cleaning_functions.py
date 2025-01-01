import pandas as pd
import numpy as np
from datetime import datetime
import re

def clean_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Converts date strings to datetime objects and creates a clean_date column.
    Handles various date formats including 'DD-Mon-YYYY'.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        date_col (str): Name of the column containing dates
        
    Returns:
        pd.DataFrame: DataFrame with cleaned date column
    """
    try:
        df['clean_date'] = pd.to_datetime(df[date_col], format='%d-%b-%Y')
    except ValueError:
        # If the above format fails, try a more general parsing
        df['clean_date'] = pd.to_datetime(df[date_col])
    return df

def clean_position_column(df: pd.DataFrame, pos_col: str = 'pos') -> pd.DataFrame:
    """
    Cleans the position column by removing trailing periods and converting to integer.
    Handles empty strings, invalid values, and numeric inputs by setting them to NaN.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        pos_col (str): Name of the column containing positions
        
    Returns:
        pd.DataFrame: DataFrame with cleaned position column
    """
    # Create a copy to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Convert the position column to string first, handling NaN values
    df['clean_position'] = df[pos_col].astype(str)
    
    # Replace 'nan' strings with empty string
    df.loc[df['clean_position'] == 'nan', 'clean_position'] = ''
    
    # Now clean the string values
    df['clean_position'] = (df['clean_position']
                          .str.replace('.', '', regex=False)  # Remove periods
                          .str.strip()  # Remove whitespace
                          .replace('', pd.NA)  # Replace empty strings with NA
                          .replace(r'^\s*$', pd.NA, regex=True)  # Replace whitespace-only with NA
                          )
    
    # Convert to float first (to handle NaN values) then to Int64 (nullable integer type)
    df['clean_position'] = pd.to_numeric(df['clean_position'], errors='coerce').astype('Int64')
    
    return df

def clean_trap_column(df: pd.DataFrame, trap_col: str = 'trap') -> pd.DataFrame:
    """
    Extracts trap number from trap column and converts to integer.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        trap_col (str): Name of the column containing trap information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned trap column
    """
    df['clean_trap'] = df[trap_col].str.extract('(\d+)').astype(int)
    return df

def clean_prize_column(df: pd.DataFrame, prize_col: str = 'prize') -> pd.DataFrame:
    """
    Cleans prize column by removing currency symbols and converting to float.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        prize_col (str): Name of the column containing prize information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned prize column
    """
    df['clean_prize'] = (df[prize_col]
                        .str.replace('€', '')
                        .str.replace('£', '')
                        .str.replace(',', '')
                        .str.replace('$', '')
                        .astype(float))
    return df

def clean_time_column(df: pd.DataFrame, time_col: str = 'win_time') -> pd.DataFrame:
    """
    Converts time strings to float values in seconds.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        time_col (str): Name of the column containing time information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned time column
    """
    df['clean_time'] = pd.to_numeric(df[time_col], errors='coerce')
    return df

def clean_going_column(df: pd.DataFrame, going_col: str = 'going') -> pd.DataFrame:
    """
    Extracts numerical going allowance from going column.
    Example: '-.30 Slow' becomes -0.30
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        going_col (str): Name of the column containing going information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned going column
    """
    def extract_going_allowance(going_str):
        if pd.isna(going_str):
            return np.nan
        try:
            return float(going_str.split()[0])
        except (ValueError, AttributeError, IndexError):
            return np.nan
    
    df['clean_going'] = df[going_col].apply(extract_going_allowance)
    return df

def clean_weight_column(df: pd.DataFrame, weight_col: str = 'wt') -> pd.DataFrame:
    """
    Cleans weight column by converting to float and handling invalid values.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        weight_col (str): Name of the column containing weight information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned weight column
    """
    # Create a copy to avoid the SettingWithCopyWarning
    df = df.copy()
    
    # Convert weight to numeric, handling invalid values
    df['clean_weight'] = pd.to_numeric(df[weight_col], errors='coerce')
    
    return df

def clean_race_distance(df: pd.DataFrame, race_name_col: str = 'race_name') -> pd.DataFrame:
    """
    Extracts and cleans race distance from race name.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        race_name_col (str): Name of the column containing race names
        
    Returns:
        pd.DataFrame: DataFrame with cleaned distance column
    """
    def extract_distance(race_name):
        if pd.isna(race_name):
            return None
        # Look for 3-4 digit numbers (distances typically 300-1000m)
        match = re.search(r'(\d{3,4})', str(race_name))
        if match:
            return int(match.group(1))
        return None
    
    df = df.copy()
    df['clean_distance'] = df[race_name_col].apply(extract_distance)
    return df

def clean_grade_column(df: pd.DataFrame, grade_col: str = 'grade') -> pd.DataFrame:
    """
    Cleans and standardizes grade column values.
    Handles common grade formats (A1, A2, etc.) and variations.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        grade_col (str): Name of the column containing grade information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned grade column
    """
    df = df.copy()
    
    # Convert to string and uppercase
    df['clean_grade'] = df[grade_col].astype(str).str.upper()
    
    # Remove any whitespace
    df['clean_grade'] = df['clean_grade'].str.strip()
    
    # Handle invalid or missing grades
    df.loc[df['clean_grade'].isin(['NAN', 'NONE', '']), 'clean_grade'] = pd.NA
    
    return df

def clean_est_time_column(df: pd.DataFrame, est_time_col: str = 'est_time') -> pd.DataFrame:
    """
    Cleans estimated time column by converting to float values in seconds.
    Handles missing values and invalid formats.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        est_time_col (str): Name of the column containing estimated time information
        
    Returns:
        pd.DataFrame: DataFrame with cleaned estimated time column
    """
    df = df.copy()
    
    # Convert estimated time to numeric, handling invalid values
    df['clean_est_time'] = pd.to_numeric(df[est_time_col], errors='coerce')
    
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning functions to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing race data
        
    Returns:
        pd.DataFrame: DataFrame with all columns cleaned
    """
    df = clean_date_column(df)
    df = clean_position_column(df)
    df = clean_trap_column(df)
    df = clean_prize_column(df)
    df = clean_time_column(df)
    df = clean_going_column(df)
    df = clean_weight_column(df)
    return df

if __name__ == "__main__":
    # Example usage
    # sample_data = pd.read_csv('greyhound_races.csv')
    # cleaned_data = clean_dataframe(sample_data)
    pass 