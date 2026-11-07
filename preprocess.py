"""Stateless cleaning for the StreamLoop churn table.

Blank TotalCharges values are turned into missing numbers. This step does
not learn anything from the rows it sees, so it is safe to keep inside the
model pipeline and safe for cross-validation to copy.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Coerce TotalCharges to a number and leave every other column unchanged."""

    def fit(self, X, y=None):
        if "TotalCharges" not in X.columns:
            raise ValueError("TotalCharges column is missing.")
        return self

    def transform(self, X):
        if "TotalCharges" not in X.columns:
            raise ValueError("TotalCharges column is missing.")
        cleaned = X.copy()
        cleaned["TotalCharges"] = cleaned["TotalCharges"].apply(_to_number)
        return cleaned

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features is required.")
        return list(input_features)


def _to_number(value):
    """Return a float, or None when the cell is blank or not a number."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None
