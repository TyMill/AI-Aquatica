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
        if not cations or not anions:
            return data

        discrepancies = data['Cations_Sum'] - data['Anions_Sum']

        cation_adjustment = discrepancies / (2 * len(cations))
        anion_adjustment = discrepancies / (2 * len(anions))

        data[cations] = data[cations].sub(cation_adjustment, axis=0)
        data[anions] = data[anions].add(anion_adjustment, axis=0)

        data['Cations_Sum'] = data[cations].sum(axis=1)
        data['Anions_Sum'] = data[anions].sum(axis=1)

        total = data['Cations_Sum'] + data['Anions_Sum']
        ion_balance = pd.Series(0.0, index=data.index)
        non_zero_total = total != 0
        ion_balance.loc[non_zero_total] = (
            (data.loc[non_zero_total, 'Cations_Sum'] - data.loc[non_zero_total, 'Anions_Sum'])
            / total.loc[non_zero_total]
            * 100
        )
        data['Ion_Balance'] = ion_balance

        return data
    except Exception as e:
        print(f"Error correcting ion discrepancies: {e}")
        return data
