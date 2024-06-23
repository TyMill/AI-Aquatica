import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def log_transform(data):
    """
    Apply logarithmic transformation to reduce skewness.
    """
    try:
        data_log_transformed = np.log1p(data)
        return pd.DataFrame(data_log_transformed, columns=data.columns)
    except Exception as e:
        print(f"Error applying log transformation: {e}")
        return data

def sqrt_transform(data):
    """
    Apply square root transformation to reduce skewness.
    """
    try:
        data_sqrt_transformed = np.sqrt(data)
        return pd.DataFrame(data_sqrt_transformed, columns=data.columns)
    except Exception as e:
        print(f"Error applying square root transformation: {e}")
        return data

def boxcox_transform(data):
    """
    Apply Box-Cox transformation to reduce skewness.
    
    Note: This transformation requires positive data.
    """
    from scipy.stats import boxcox
    try:
        data_boxcox_transformed = pd.DataFrame()
        for col in data.columns:
            # Ensure all values are positive
            if (data[col] <= 0).any():
                raise ValueError(f"Column {col} contains non-positive values, which are not suitable for Box-Cox transformation.")
            transformed, _ = boxcox(data[col])
            data_boxcox_transformed[col] = transformed
        return data_boxcox_transformed
    except Exception as e:
        print(f"Error applying Box-Cox transformation: {e}")
        return data
