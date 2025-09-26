import unittest
import pandas as pd
from ai_aquatica.ion_balance import (
    calculate_ion_balance,
    identify_potential_errors,
    correct_ion_discrepancies,
)

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

    def test_correct_ion_discrepancies_balances_varied_configurations(self):
        scenarios = [
            {
                'name': 'multiple_ions',
                'data': self.data.copy(),
                'cations': self.cations,
                'anions': self.anions,
            },
            {
                'name': 'single_pairs',
                'data': pd.DataFrame(
                    {
                        'Na': [4.0, 8.0, 0.0],
                        'Cl': [1.0, 5.0, 0.0],
                    }
                ),
                'cations': ['Na'],
                'anions': ['Cl'],
            },
            {
                'name': 'zero_totals',
                'data': pd.DataFrame(
                    {
                        'Ca': [0.0, 0.0],
                        'Mg': [0.0, 0.0],
                        'Cl': [0.0, 0.0],
                        'SO4': [0.0, 0.0],
                    }
                ),
                'cations': ['Ca', 'Mg'],
                'anions': ['Cl', 'SO4'],
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario['name']):
                data_with_balance = calculate_ion_balance(
                    scenario['data'], scenario['cations'], scenario['anions']
                )
                corrected_data = correct_ion_discrepancies(
                    data_with_balance, scenario['cations'], scenario['anions']
                )

                cation_minus_anion = corrected_data['Cations_Sum'] - corrected_data['Anions_Sum']
                self.assertTrue(
                    (cation_minus_anion.abs() < 1e-6).all(),
                    msg=f"Residual imbalance in scenario {scenario['name']} was {cation_minus_anion}",
                )

                if (corrected_data['Cations_Sum'] + corrected_data['Anions_Sum']).ne(0).any():
                    self.assertTrue(
                        corrected_data.loc[
                            (corrected_data['Cations_Sum'] + corrected_data['Anions_Sum']) != 0,
                            'Ion_Balance',
                        ]
                        .abs()
                        .lt(1e-6)
                        .all(),
                        msg=f"Ion balance not reduced for scenario {scenario['name']}",
                    )
                else:
                    self.assertTrue((corrected_data['Ion_Balance'] == 0).all())

                self.assertFalse(corrected_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
