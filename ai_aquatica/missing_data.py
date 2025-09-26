import importlib

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

try:  # pragma: no cover - exercised via dedicated tests
    tf = importlib.import_module("tensorflow")
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_models = importlib.import_module("tensorflow.keras.models")
    Input = keras_layers.Input
    Dense = keras_layers.Dense
    Model = keras_models.Model
    TENSORFLOW_AVAILABLE = True
except Exception:  # TensorFlow is optional dependency
    tf = None
    TENSORFLOW_AVAILABLE = False

    def _tensorflow_placeholder(*args, **kwargs):
        raise ImportError(
            "TensorFlow is required for autoencoder-based imputation. "
            "Install it with `pip install ai-aquatica[deep_learning]`."
        )

    Input = Dense = Model = _tensorflow_placeholder


# Metody Statystyczne

def fill_missing_with_mean(data):
    """
    Fill missing values with the mean of each column.
    """
    try:
        data_filled = data.fillna(data.mean())
        return data_filled
    except Exception as e:
        print(f"Error filling missing values with mean: {e}")
        return data

def fill_missing_with_median(data):
    """
    Fill missing values with the median of each column.
    """
    try:
        data_filled = data.fillna(data.median())
        return data_filled
    except Exception as e:
        print(f"Error filling missing values with median: {e}")
        return data

def fill_missing_with_mode(data):
    """
    Fill missing values with the mode of each column.
    """
    try:
        data_filled = data.copy()
        for column in data.columns:
            mode_value = data[column].mode()[0]
            data_filled[column].fillna(mode_value, inplace=True)
        return data_filled
    except Exception as e:
        print(f"Error filling missing values with mode: {e}")
        return data

# Algorytmy AI/ML

def fill_missing_with_knn(data, n_neighbors=5):
    """
    Fill missing values using KNN imputer.
    """
    try:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return data_filled
    except Exception as e:
        print(f"Error filling missing values with KNN: {e}")
        return data

def fill_missing_with_regression(data):
    """
    Fill missing values using regression imputer.
    """
    try:
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0)
        data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return data_filled
    except Exception as e:
        print(f"Error filling missing values with regression: {e}")
        return data

def fill_missing_with_autoencoder(data):
    """
    Fill missing values using autoencoder.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for autoencoder-based imputation. "
            "Install it with `pip install ai-aquatica[deep_learning]`."
        )

    try:
        # Prepare the data
        data_filled = data.copy()
        input_dim = data.shape[1]

        # Define the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(64, activation="relu")(input_layer)
        encoder = Dense(32, activation="relu")(encoder)
        decoder = Dense(64, activation="relu")(encoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer="adam", loss="mean_squared_error")

        # Fit the model
        data_filled = data.fillna(data.mean())  # Simple imputation for initial missing values
        autoencoder.fit(data_filled, data_filled, epochs=50, batch_size=32, shuffle=True, verbose=0)

        # Impute missing values using the fully observed data
        reconstructed = autoencoder.predict(data_filled)
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=data.index,
            columns=data.columns,
        )

        # Clip predictions to the observed range where possible
        col_mins = data.min()
        col_maxs = data.max()
        reconstructed_df = reconstructed_df.clip(lower=col_mins, upper=col_maxs, axis=1)

        # Preserve original non-missing values
        reconstructed_df[data.notna()] = data[data.notna()]

        return reconstructed_df
    except Exception as e:
        print(f"Error filling missing values with autoencoder: {e}")
        return data
