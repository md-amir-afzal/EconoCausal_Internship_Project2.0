import unittest
from pathlib import Path
import pandas as pd
from src.generate_data import generate_pricing_data
from src.dml_model import DoubleMLPricingModel

class TestEconoCausal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_pricing_data(n=1200, seed=123)
        cls.model = DoubleMLPricingModel(n_splits=3, random_state=123).fit(cls.df)

    def test_dataset_shape(self):
        self.assertEqual(len(self.df), 1200)
        for col in ["price", "units_sold", "competitor_price", "cost"]:
            self.assertIn(col, self.df.columns)

    def test_no_negative_price(self):
        self.assertTrue((self.df["price"] > 0).all())

    def test_model_returns_finite_elasticity(self):
        self.assertTrue(pd.notna(self.model.result_.elasticity))
        self.assertTrue(abs(self.model.result_.elasticity) < 5)

    def test_elasticity_direction(self):
        # The synthetic data generating process uses a negative causal effect.
        self.assertLess(self.model.result_.elasticity, 0)

    def test_metrics_finite(self):
        for value in self.model.metrics_dict().values():
            if isinstance(value, float):
                self.assertTrue(pd.notna(value))

if __name__ == "__main__":
    unittest.main()
