import unittest
import pandas as pd
from ai_aquatica.ion_balance import calculate_ion_balance, identify_potential_errors, correct_ion_discrepancies

class TestIonBalance(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Ca': [10, 20, 30],
            'Mg': [5, 10, 15],
            'Na': [2, 4, 6],
            'K': [1, 2, 3],
            'Cl': [8, 16, 24],
            'SO4': [4, 8, 12],
            'HCO3': [6, 12, 18]
        })
        self.cations = ['Ca', 'Mg', 'Na', 'K']
        self.anions = ['Cl', 'SO4', 'HCO3']

    def test_calculate_ion_balance(self):
        data_with_balance = calculate_ion_balance(self.data, self.cations, self.anions)
        self.assertIn('Ion_Balance', data_with_balance.columns)

    def test_identify_potential_errors(self):
        data_with_balance = calculate_ion_balance(self.data, self.cations, self.anions)
        data_with_errors = identify_potential_errors(data_with_balance)
        self.assertIn('Potential_Error', data_with_errors.columns)

    def test_correct_ion_discrepancies(self):
        data_with_balance = calculate_ion_balance(self.data, self.cations, self.anions)
        corrected_data = correct_ion_discrepancies(data_with_balance, self.cations, self.anions)
        self.assertFalse(corrected_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
