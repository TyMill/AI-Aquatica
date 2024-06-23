import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Regresja

def train_linear_regression(X, y):
    """
    Train a linear regression model.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.

    Returns:
    LinearRegression: Trained model.
    """
    try:
        model = LinearRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Error training linear regression model: {e}")
        return None

def train_logistic_regression(X, y):
    """
    Train a logistic regression model.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.

    Returns:
    LogisticRegression: Trained model.
    """
    try:
        model = LogisticRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Error training logistic regression model: {e}")
        return None

# Klasyfikacja

def train_classification_model(X, y, model_type='decision_tree'):
    """
    Train a classification model.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    model_type (str): Type of classification model ('decision_tree', 'svm', 'knn', 'random_forest').

    Returns:
    model: Trained classification model.
    """
    try:
        if model_type == 'decision_tree':
            model = DecisionTreeClassifier()
        elif model_type == 'svm':
            model = SVC()
        elif model_type == 'knn':
            model = KNeighborsClassifier()
        elif model_type == 'random_forest':
            model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type.")
        
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Error training classification model: {e}")
        return None

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model.

    Parameters:
    model: Trained classification model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.

    Returns:
    dict: Dictionary with evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        print(f"Error evaluating classification model: {e}")
        return {}

# Clustering

def perform_clustering(X, n_clusters=3, algorithm='kmeans'):
    """
    Perform clustering on the dataset.

    Parameters:
    X (pd.DataFrame): Features.
    n_clusters (int): Number of clusters.
    algorithm (str): Clustering algorithm ('kmeans', 'dbscan').

    Returns:
    tuple: Model and labels.
    """
    try:
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            model = DBSCAN()
        else:
            raise ValueError("Unsupported clustering algorithm.")
        
        labels = model.fit_predict(X)
        return model, labels
    except Exception as e:
        print(f"Error performing clustering: {e}")
        return None, None

def plot_clusters(X, labels):
    """
    Plot the clustering results.

    Parameters:
    X (pd.DataFrame): Features.
    labels (np.array): Cluster labels.

    Returns:
    None
    """
    try:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.title('Clustering Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    except Exception as e:
        print(f"Error plotting clusters: {e}")

# Detekcja Anomalii

def detect_anomalies(X, method='isolation_forest'):
    """
    Detect anomalies in the dataset.

    Parameters:
    X (pd.DataFrame): Features.
    method (str): Anomaly detection method ('isolation_forest', 'lof').

    Returns:
    np.array: Anomaly labels.
    """
    try:
        if method == 'isolation_forest':
            model = IsolationForest()
        elif method == 'lof':
            model = LocalOutlierFactor()
        else:
            raise ValueError("Unsupported anomaly detection method.")
        
        if method == 'lof':
            labels = model.fit_predict(X)
        else:
            labels = model.fit(X).predict(X)
        
        return labels
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        return np.array([])

# Generowanie Danych

def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.trainable = False
    gan_input = layers.Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

def build_generator(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(output_dim, activation='tanh'))
    return model

def build_discriminator(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def generate_synthetic_data(X, model_type='gan', epochs=100):
    """
    Generate synthetic data using generative models.

    Parameters:
    X (pd.DataFrame): Features.
    model_type (str): Type of generative model ('gan', 'vae').
    epochs (int): Number of training epochs.

    Returns:
    pd.DataFrame: Synthetic data.
    """
    try:
        if model_type == 'gan':
            input_dim = X.shape[1]
            generator = build_generator(input_dim=100, output_dim=input_dim)
            discriminator = build_discriminator(input_dim=input_dim)
            gan = build_gan(generator, discriminator)

            # Training GAN
            batch_size = 32
            half_batch = batch_size // 2

            for epoch in range(epochs):
                # Train discriminator
                idx = np.random.randint(0, X.shape[0], half_batch)
                real_data = X.iloc[idx].values
                noise = np.random.normal(0, 1, (half_batch, 100))
                generated_data = generator.predict(noise)

                d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
                d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                noise = np.random.normal(0, 1, (batch_size, 100))
                valid_y = np.array([1] * batch_size)
                g_loss = gan.train_on_batch(noise, valid_y)

                if epoch % 10 == 0:
                    print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")

            # Generate synthetic data
            noise = np.random.normal(0, 1, (X.shape[0], 100))
            synthetic_data = generator.predict(noise)
            synthetic_data = pd.DataFrame(synthetic_data, columns=X.columns)

            return synthetic_data

        else:
            raise ValueError("Currently, only 'gan' model_type is supported.")
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return pd.DataFrame()

# Przyklady Uzycia

if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })

    X = data[['feature1', 'feature2']]
    y = data['target']

    # Train and evaluate a linear regression model
    linear_model = train_linear_regression(X, y)
    print("Linear Regression Coefficients:", linear_model.coef_)

    # Train and evaluate a decision tree classifier
    clf_model = train_classification_model(X, y, model_type='decision_tree')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    evaluation = evaluate_classification_model(clf_model, X_test, y_test)
    print("Classification Evaluation:", evaluation)

    # Perform clustering and plot results
    cluster_model, labels = perform_clustering(X, n_clusters=3, algorithm='kmeans')
    plot_clusters(X, labels)

    # Detect anomalies
    anomaly_labels = detect_anomalies(X, method='isolation_forest')
    print("Anomaly Labels:", anomaly_labels)
