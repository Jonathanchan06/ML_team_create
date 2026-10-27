"""Step 2 of the model ladder: linear regression on the features (sanity check, interpretable)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import LABEL
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class LinearModel:
    def fit(self, table):
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        self.pipeline = make_pipeline(prep, LinearRegression())
        self.pipeline.fit(table[NUMERIC_FEATURES + CATEGORICAL_FEATURES], table[LABEL])
        return self

    def predict(self, table):
        pred = self.pipeline.predict(table[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        return pred.clip(0, 1)

    def coefficients(self):
        """Coefficient per (standardized) feature, for reading off what the model learned."""
        names = self.pipeline[0].get_feature_names_out()
        return pd.Series(self.pipeline[-1].coef_, index=names).sort_values()
