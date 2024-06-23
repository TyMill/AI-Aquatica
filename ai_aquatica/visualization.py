import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Podstawowe Wizualizacje

def plot_line(data, x_column, y_column):
    """
    Plot a line chart.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    x_column (str): Column name for the x-axis.
    y_column (str): Column name for the y-axis.

    Returns:
    None
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data[x_column], data[y_column])
        plt.title(f'Line Plot of {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
    except Exception as e:
        print(f"Error plotting line chart: {e}")

def plot_bar(data, x_column, y_column):
    """
    Plot a bar chart.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    x_column (str): Column name for the x-axis.
    y_column (str): Column name for the y-axis.

    Returns:
    None
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x_column, y=y_column, data=data)
        plt.title(f'Bar Plot of {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
    except Exception as e:
        print(f"Error plotting bar chart: {e}")

def plot_pie(data, column):
    """
    Plot a pie chart.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    column (str): Column name for the pie chart.

    Returns:
    None
    """
    try:
        plt.figure(figsize=(10, 6))
        data[column].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {column}')
        plt.ylabel('')
        plt.show()
    except Exception as e:
        print(f"Error plotting pie chart: {e}")

def plot_scatter(data, x_column, y_column):
    """
    Plot a scatter plot.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    x_column (str): Column name for the x-axis.
    y_column (str): Column name for the y-axis.

    Returns:
    None
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_column, y=y_column, data=data)
        plt.title(f'Scatter Plot of {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
    except Exception as e:
        print(f"Error plotting scatter plot: {e}")

def plot_heatmap(data):
    """
    Plot a heatmap of the correlation matrix.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.

    Returns:
    None
    """
    try:
        plt.figure(figsize=(10, 6))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlation Matrix')
        plt.show()
    except Exception as e:
        print(f"Error plotting heatmap: {e}")

# Zaawansowane Wizualizacje

def plot_pca(data, n_components=2):
    """
    Plot a PCA of the data.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    n_components (int): Number of components for PCA.

    Returns:
    None
    """
    try:
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title('PCA Plot')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    except Exception as e:
        print(f"Error plotting PCA: {e}")

def plot_tsne(data, perplexity=30, n_components=2, learning_rate=200, n_iter=1000):
    """
    Plot a t-SNE of the data.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    perplexity (int): Perplexity for t-SNE.
    n_components (int): Number of components for t-SNE.
    learning_rate (int): Learning rate for t-SNE.
    n_iter (int): Number of iterations for t-SNE.

    Returns:
    None
    """
    try:
        tsne = TSNE(perplexity=perplexity, n_components=n_components, learning_rate=learning_rate, n_iter=n_iter)
        tsne_result = tsne.fit_transform(data)
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
        plt.title('t-SNE Plot')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
    except Exception as e:
        print(f"Error plotting t-SNE: {e}")

def plot_interactive_bubble(data, x_column, y_column, size_column, hover_column):
    """
    Plot an interactive bubble chart.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    x_column (str): Column name for the x-axis.
    y_column (str): Column name for the y-axis.
    size_column (str): Column name for the bubble sizes.
    hover_column (str): Column name for the hover text.

    Returns:
    None
    """
    try:
        import plotly.express as px
        fig = px.scatter(data, x=x_column, y=y_column, size=size_column, hover_name=hover_column, size_max=60)
        fig.show()
    except Exception as e:
        print(f"Error plotting interactive bubble chart: {e}")

# Przyklady Uzycia

if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'size': np.random.randint(1, 100, 100)
    })

    # Podstawowe Wizualizacje
    plot_line(data, 'feature1', 'feature2')
    plot_bar(data, 'category', 'size')
    plot_pie(data, 'category')
    plot_scatter(data, 'feature1', 'feature2')
    plot_heatmap(data)

    # Zaawansowane Wizualizacje
    plot_pca(data[['feature1', 'feature2']])
    plot_tsne(data[['feature1', 'feature2']])
    plot_interactive_bubble(data, 'feature1', 'feature2', 'size', 'category')
