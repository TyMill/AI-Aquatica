import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def remove_duplicates(data):
    """Remove duplicate observations while preserving the original order.

    The function first identifies columns that contain repeated values.
    If such columns exist, duplicates are removed based on these columns,
    otherwise a full-row duplicate check is performed. In both cases the
    first occurrence is preserved.
    """
    try:
        subset_columns = [
            column
            for column in data.columns
            if data[column].duplicated(keep=False).any()
        ]

        if subset_columns:
            data_cleaned = data.drop_duplicates(subset=subset_columns, keep="first")
        else:
            data_cleaned = data.drop_duplicates(keep="first")

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

    The transformation centers each column by its mean and scales by the
    sample standard deviation (ddof=1). Columns with zero or undefined
    variance are left at zero after centering.
    """
    try:
        means = data.mean()
        stds = data.std(ddof=1)

        # Replace zeros or NaNs in the standard deviation to avoid division by zero
        safe_stds = stds.replace(0, 1).fillna(1)

        data_standardized = (data - means) / safe_stds
        return pd.DataFrame(data_standardized, columns=data.columns)
    except Exception as e:
        print(f"Error standardizing data: {e}")
        return data
