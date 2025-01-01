import pandas as pd

def add_previous_race_count(df: pd.DataFrame, dog_col: str) -> pd.DataFrame:
    """
    Adds a new column 'previous_race' to the DataFrame, indicating the number of races 
    each dog has participated in prior to the current race.

    Args:
        df (pd.DataFrame): The input DataFrame containing race data
        dog_col (str): The name of the column containing the dog's name

    Returns:
        pd.DataFrame: DataFrame with added 'previous_race' column
    """
    df = df.sort_values(by=[dog_col, 'clean_date'])
    # Subtract 1 to get count of previous races only
    df['previous_race'] = df.groupby(dog_col).cumcount() - 1
    return df

def add_previous_grade_race_count(df: pd.DataFrame, dog_col: str, 
                                grade_col: str, output_col: str) -> pd.DataFrame:
    """
    Adds a new column indicating the number of races each dog has participated 
    in with the same grade prior to the current race.
    """
    df = df.sort_values(by=[dog_col, 'clean_date'])
    # Subtract 1 to get count of previous races only
    df[output_col] = df.groupby([dog_col, grade_col]).cumcount() - 1
    return df

def calculate_win_rate(df: pd.DataFrame, dog_col: str, 
                      position_col: str, lookback: int = None) -> pd.DataFrame:
    """
    Calculates the historical win rate for each dog up to (but not including) each race.
    """
    df = df.copy()
    df = df.sort_values(by=[dog_col, 'clean_date'])
    
    # Create is_win column first, handling None/NaN values
    df['is_win'] = pd.Series(
        [(pos == 1) if pd.notna(pos) else False for pos in df[position_col]],
        index=df.index
    ).astype(float)
    
    if lookback:
        df['win_rate'] = (df.groupby(dog_col)['is_win']
                         .shift()
                         .rolling(window=lookback, min_periods=1)
                         .mean()
                         .fillna(0))
    else:
        df['win_rate'] = (df.groupby(dog_col)['is_win']
                         .shift()
                         .expanding()
                         .mean()
                         .fillna(0))
    
    df = df.drop('is_win', axis=1)
    
    return df

def calculate_average_position(df: pd.DataFrame, dog_col: str, 
                             position_col: str, lookback: int = None) -> pd.DataFrame:
    """
    Calculates the historical average finishing position for each dog up to 
    (but not including) each race.
    """
    df = df.sort_values(by=[dog_col, 'clean_date'])
    
    if lookback:
        df['avg_position'] = (df.groupby(dog_col)[position_col]
                            .shift()  # Shift to exclude current race
                            .rolling(window=lookback, min_periods=1)
                            .mean()
                            .fillna(0))  # Fill first race with 0
    else:
        df['avg_position'] = (df.groupby(dog_col)[position_col]
                            .shift()  # Shift to exclude current race
                            .expanding()
                            .mean()
                            .fillna(0))  # Fill first race with 0
    
    return df

def calculate_weight_change(df: pd.DataFrame, dog_col: str, 
                          weight_col: str = 'clean_weight') -> pd.DataFrame:
    """
    Calculates the weight change from the previous race for each dog.
    """
    df = df.copy()
    df = df.sort_values(by=[dog_col, 'clean_date'])
    
    # Calculate weight change from previous race
    df['weight_change'] = df.groupby(dog_col)[weight_col].diff()
    
    # Calculate percentage weight change from previous race
    previous_weight = df.groupby(dog_col)[weight_col].shift(1)
    df['weight_change_pct'] = (df['weight_change'] / previous_weight * 100)
    
    # Forward fill weight changes within each dog group
    df['weight_change'] = df.groupby(dog_col)['weight_change'].fillna(method='ffill')
    df['weight_change_pct'] = df.groupby(dog_col)['weight_change_pct'].fillna(method='ffill')
    
    # For first race of each dog, set changes to 0
    df['weight_change'] = df.groupby(dog_col)['weight_change'].fillna(0)
    df['weight_change_pct'] = df.groupby(dog_col)['weight_change_pct'].fillna(0)
    
    return df

def calculate_average_weight(df: pd.DataFrame, dog_col: str, 
                           weight_col: str = 'clean_weight',
                           lookback: int = None) -> pd.DataFrame:
    """
    Calculates the historical average weight for each dog up to (but not including) each race.
    """
    df = df.copy()
    df = df.sort_values(by=[dog_col, 'clean_date'])
    
    if lookback:
        df['avg_weight'] = (df.groupby(dog_col)[weight_col]
                           .shift()
                           .rolling(window=lookback, min_periods=1)
                           .mean()
                           .fillna(method='bfill'))  # Use next available weight if no previous
    else:
        df['avg_weight'] = (df.groupby(dog_col)[weight_col]
                           .shift()
                           .expanding()
                           .mean()
                           .fillna(method='bfill'))  # Use next available weight if no previous
    
    # Forward fill average weights within each dog group
    df['avg_weight'] = df.groupby(dog_col)['avg_weight'].fillna(method='ffill')
    
    # If still no average weight, use current weight
    df['avg_weight'] = df['avg_weight'].fillna(df[weight_col])
    
    # Calculate deviation from average weight
    df['weight_dev_from_avg'] = df[weight_col] - df['avg_weight']
    
    return df

def calculate_distance_performance(df: pd.DataFrame, dog_col: str, 
                                distance_col: str = 'clean_distance',
                                position_col: str = 'clean_position',
                                lookback: int = None) -> pd.DataFrame:
    """
    Calculates performance metrics for each distance a dog has raced, 
    using only previous races.
    """
    df = df.copy()
    df = df.sort_values(by=[dog_col, 'clean_date'])
    
    # Calculate average position for each distance using previous races
    if lookback:
        df['avg_position_at_distance'] = (df.groupby([dog_col, distance_col])[position_col]
                                        .shift()  # Shift to exclude current race
                                        .rolling(window=lookback, min_periods=1)
                                        .mean()
                                        .fillna(0))
    else:
        df['avg_position_at_distance'] = (df.groupby([dog_col, distance_col])[position_col]
                                        .shift()  # Shift to exclude current race
                                        .expanding()
                                        .mean()
                                        .fillna(0))
    
    # Create is_win column first
    df['is_win'] = (df[position_col] == 1).astype(float)
    
    # Calculate win rate at each distance using previous races
    if lookback:
        df['win_rate_at_distance'] = (df.groupby([dog_col, distance_col])['is_win']
                                    .shift()  # Shift to exclude current race
                                    .rolling(window=lookback, min_periods=1)
                                    .mean()
                                    .fillna(0))
    else:
        df['win_rate_at_distance'] = (df.groupby([dog_col, distance_col])['is_win']
                                    .shift()  # Shift to exclude current race
                                    .expanding()
                                    .mean()
                                    .fillna(0))
    
    # Count previous races at each distance
    df['races_at_distance'] = (df.groupby([dog_col, distance_col]).cumcount())
    
    # Clean up temporary column
    df = df.drop('is_win', axis=1)
    
    return df

def calculate_preferred_distance(df: pd.DataFrame, dog_col: str,
                               distance_col: str = 'clean_distance',
                               position_col: str = 'clean_position',
                               min_races: int = 3) -> pd.DataFrame:
    """
    Identifies each dog's preferred racing distance based on previous performance.
    Only uses historical data (where position is not null) to determine preferences.
    If a dog doesn't have enough races at any distance, uses stats from all distances.
    Handles both historical analysis and future predictions.
    """
    df = df.copy()
    df = df.sort_values('clean_date')  # Ensure data is sorted by date
    
    # Initialize columns with None
    for col in [f'{distance_col}_preferred', 'avg_position_preferred', 'win_rate_preferred']:
        df[col] = None
    
    # Process each race date
    for date in df['clean_date'].unique():
        # Get historical data up to this date
        historical_mask = (df['clean_date'] < date) & (df[position_col].notna())
        historical_data = df[historical_mask].copy()
        
        # Get current date's dogs
        current_dogs = df[df['clean_date'] == date][dog_col].unique()
        
        if len(historical_data) == 0:
            continue
        
        # Create shifted position column for calculations
        historical_data['previous_position'] = historical_data.groupby(dog_col)[position_col].shift()
        historical_data['previous_is_win'] = (historical_data['previous_position'] == 1).astype(float)
        
        # Calculate stats using only historical races
        distance_stats = (historical_data.groupby([dog_col, distance_col])
                         .agg({
                             'previous_position': ['count', 'mean'],
                             'previous_is_win': 'mean'
                         })
                         .reset_index())
        
        # Rename columns
        distance_stats.columns = [dog_col, distance_col, 'total_races', 'avg_position', 'win_rate']
        
        # Find best distance based on average position for dogs with enough races
        qualified_distances = distance_stats[distance_stats['total_races'] >= min_races]
        best_distances = (qualified_distances
                         .sort_values('avg_position')
                         .groupby(dog_col)
                         .first()
                         .reset_index())
        
        # For dogs without enough races at any distance, use stats from all distances combined
        qualified_dogs = set(best_distances[dog_col])
        unqualified_dogs = set(current_dogs) - qualified_dogs
        
        if unqualified_dogs:
            # Calculate overall stats for each unqualified dog
            overall_stats = (historical_data[historical_data[dog_col].isin(unqualified_dogs)]
                           .groupby(dog_col)
                           .agg({
                               distance_col: 'first',
                               'previous_position': 'mean',
                               'previous_is_win': 'mean'
                           })
                           .reset_index())
            
            # Format unqualified dogs' stats
            unqualified_stats = overall_stats.rename(columns={
                distance_col: f'{distance_col}_preferred',
                'previous_position': 'avg_position_preferred',
                'previous_is_win': 'win_rate_preferred'
            })
            
            # Format qualified dogs' stats
            best_distances = best_distances.rename(columns={
                distance_col: f'{distance_col}_preferred',
                'avg_position': 'avg_position_preferred',
                'win_rate': 'win_rate_preferred'
            })
            
            # Combine stats
            best_distances = pd.concat([best_distances, unqualified_stats], ignore_index=True)
        else:
            # Rename columns for qualified dogs
            best_distances = best_distances.rename(columns={
                distance_col: f'{distance_col}_preferred',
                'avg_position': 'avg_position_preferred',
                'win_rate': 'win_rate_preferred'
            })
        
        # Drop the total_races column if it exists
        if 'total_races' in best_distances.columns:
            best_distances = best_distances.drop('total_races', axis=1)
        
        # Update the values for this date's races
        date_mask = df['clean_date'] == date
        df.loc[date_mask, [f'{distance_col}_preferred', 'avg_position_preferred', 'win_rate_preferred']] = (
            df[date_mask][[dog_col]].merge(best_distances, on=dog_col, how='left')
            [[f'{distance_col}_preferred', 'avg_position_preferred', 'win_rate_preferred']]
            .values
        )
    
    # Calculate if current race is at preferred distance
    df['is_preferred_distance'] = (
        df[distance_col] == df[f'{distance_col}_preferred']
    ).astype(int)
    
    return df

if __name__ == "__main__":
    # This section only runs if the script is run directly
    # You can add test cases here
    pass 