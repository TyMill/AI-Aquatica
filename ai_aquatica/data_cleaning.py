import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    """
    try:
        data_cleaned = data.drop_duplicates()
        return data_cleaned
    except Exception as e:
        print(f"Error removing duplicates: {e}")
        return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    - strategy: 'mean', 'median', 'interpolate'
    """
    try:
        if strategy == 'mean':
            data_filled = data.fillna(data.mean())
        elif strategy == 'median':
            data_filled = data.fillna(data.median())
        elif strategy == 'interpolate':
            data_filled = data.interpolate()
        else:
            raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'interpolate'.")
        return data_filled
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return data

def normalize_data(data):
    """
    Normalize data to range [0, 1].
    """
    try:
        scaler = MinMaxScaler()
        data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return data_normalized
    except Exception as e:
        print(f"Error normalizing data: {e}")
        return data

def standardize_data(data):
    """
    Standardize data to have mean 0 and variance 1 using Z-score.
    """
    try:
        scaler = StandardScaler()
        data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return data_standardized
    except Exception as e:
        print(f"Error standardizing data: {e}")
        return data
