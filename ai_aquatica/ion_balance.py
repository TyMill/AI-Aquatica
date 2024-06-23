import pandas as pd

def calculate_ion_balance(data, cations, anions):
    """
    Calculate the ion balance for the provided data.

    Parameters:
    data (pd.DataFrame): DataFrame containing ion concentrations.
    cations (list): List of cation columns in the DataFrame.
    anions (list): List of anion columns in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame with an additional column for ion balance.
    """
    try:
        data['Cations_Sum'] = data[cations].sum(axis=1)
        data['Anions_Sum'] = data[anions].sum(axis=1)
        data['Ion_Balance'] = (data['Cations_Sum'] - data['Anions_Sum']) / (data['Cations_Sum'] + data['Anions_Sum']) * 100
        return data
    except Exception as e:
        print(f"Error calculating ion balance: {e}")
        return data

def identify_potential_errors(data, threshold=5.0):
    """
    Identify potential errors in chemical analysis based on ion balance.

    Parameters:
    data (pd.DataFrame): DataFrame containing ion balance.
    threshold (float): Threshold for acceptable ion balance error percentage.

    Returns:
    pd.DataFrame: DataFrame with potential errors flagged.
    """
    try:
        data['Potential_Error'] = abs(data['Ion_Balance']) > threshold
        return data
    except Exception as e:
        print(f"Error identifying potential errors: {e}")
        return data

def correct_ion_discrepancies(data, cations, anions):
    """
    Correct discrepancies in ion balance by adjusting concentrations.

    Parameters:
    data (pd.DataFrame): DataFrame containing ion concentrations and balance.
    cations (list): List of cation columns in the DataFrame.
    anions (list): List of anion columns in the DataFrame.

    Returns:
    pd.DataFrame: DataFrame with corrected ion concentrations.
    """
    try:
        discrepancies = data['Cations_Sum'] - data['Anions_Sum']
        adjustment_factor = discrepancies / len(cations)
        
        for cation in cations:
            data[cation] -= adjustment_factor / len(cations)
        
        for anion in anions:
            data[anion] += adjustment_factor / len(anions)
        
        return data
    except Exception as e:
        print(f"Error correcting ion discrepancies: {e}")
        return data
