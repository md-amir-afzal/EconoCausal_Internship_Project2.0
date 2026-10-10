
import unittest
import numpy as np
from src.data_generator import generate_data
from src.dml_model import estimate_dml

class TestPipeline(unittest.TestCase):
    def test_data_size_and_columns(self):
        df = generate_data(1000, 42)
        self.assertEqual(len(df), 1000)
        for c in ["price","units_sold","demand_index","inventory","promotion"]:
            self.assertIn(c, df.columns)

    def test_positive_prices(self):
        self.assertTrue((generate_data(500, 42)["price"] > 0).all())

    def test_dml_result(self):
        result = estimate_dml(generate_data(1500, 42), folds=3, seed=42)
        self.assertTrue(np.isfinite(result["elasticity"]))
        self.assertLess(result["elasticity"], 0)

if __name__ == "__main__":
    unittest.main()
