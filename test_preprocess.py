"""Checks for the blank-charge cleaner. Uses only the standard library."""

import unittest

import pandas as pd

from preprocess import TotalChargesCleaner


class TotalChargesCleanerTest(unittest.TestCase):
    def test_blank_becomes_missing_and_real_number_stays(self):
        frame = pd.DataFrame(
            {
                "customerID": ["a", "b", "c"],
                "TotalCharges": ["  ", "29.85", "not-a-number"],
                "tenure": [0, 9, 1],
            }
        )

        cleaner = TotalChargesCleaner()
        cleaned = cleaner.fit(frame).transform(frame)

        self.assertTrue(pd.isna(cleaned.loc[0, "TotalCharges"]))
        self.assertEqual(cleaned.loc[1, "TotalCharges"], 29.85)
        self.assertTrue(pd.isna(cleaned.loc[2, "TotalCharges"]))
        self.assertEqual(list(cleaned["customerID"]), ["a", "b", "c"])
        self.assertEqual(list(cleaned["tenure"]), [0, 9, 1])
        self.assertEqual(frame.loc[0, "TotalCharges"], "  ")

    def test_fit_does_not_store_a_fill_value(self):
        first = pd.DataFrame({"TotalCharges": ["10"]})
        second = pd.DataFrame({"TotalCharges": [" "]})
        cleaner = TotalChargesCleaner().fit(first)

        cleaned = cleaner.transform(second)

        self.assertTrue(pd.isna(cleaned.loc[0, "TotalCharges"]))

    def test_missing_column_is_rejected(self):
        frame = pd.DataFrame({"tenure": [1]})
        cleaner = TotalChargesCleaner()
        with self.assertRaises(ValueError):
            cleaner.fit(frame)


if __name__ == "__main__":
    unittest.main()
