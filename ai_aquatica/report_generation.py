import pandas as pd
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_statistical_report(data, report_path='statistical_report.html'):
    """
    Generate a statistical report with basic and advanced statistics.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    report_path (str): Path to save the generated report.

    Returns:
    None
    """
    try:
        # Podstawowe statystyki
        basic_stats = data.describe().T
        basic_stats['range'] = basic_stats['max'] - basic_stats['min']

        # Korelacja
        correlation_matrix = data.corr()

        # Generowanie wykresów
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        heatmap_path = 'heatmap.png'
        plt.savefig(heatmap_path)
        plt.close()

        # Szablon Jinja2
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('statistical_report_template.html')

        # Renderowanie szablonu
        html_content = template.render(
            basic_stats=basic_stats.to_html(classes='table table-striped'),
            correlation_matrix=correlation_matrix.to_html(classes='table table-striped'),
            heatmap_path=heatmap_path
        )

        # Zapis raportu
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating statistical report: {e}")

def generate_interpretation_report(data, model, X_test, y_test, report_path='interpretation_report.html'):
    """
    Generate an interpretation report for model results.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    model: Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    report_path (str): Path to save the generated report.

    Returns:
    None
    """
    try:
        # Predykcje i interpretacja wyników
        y_pred = model.predict(X_test)
        evaluation_metrics = {
            'confusion_matrix': pd.DataFrame(confusion_matrix(y_test, y_pred)),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        # Szablon Jinja2
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('interpretation_report_template.html')

        # Renderowanie szablonu
        html_content = template.render(
            evaluation_metrics=evaluation_metrics,
            confusion_matrix=evaluation_metrics['confusion_matrix'].to_html(classes='table table-striped'),
            precision=evaluation_metrics['precision'],
            recall=evaluation_metrics['recall'],
            f1_score=evaluation_metrics['f1_score']
        )

        # Zapis raportu
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating interpretation report: {e}")

def suggest_further_analysis(data, report_path='further_analysis_report.html'):
    """
    Generate a report with suggestions for further analysis.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    report_path (str): Path to save the generated report.

    Returns:
    None
    """
    try:
        suggestions = [
            "Consider examining the distribution of each variable for normality.",
            "Investigate potential correlations between variables using scatter plots.",
            "Perform clustering analysis to identify potential groupings in the data.",
            "Use advanced models such as Random Forests or Gradient Boosting for better predictions.",
            "Analyze temporal trends if your data includes time series information."
        ]

        # Szablon Jinja2
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('further_analysis_template.html')

        # Renderowanie szablonu
        html_content = template.render(
            suggestions=suggestions
        )

        # Zapis raportu
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating further analysis report: {e}")
