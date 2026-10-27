"""Step 3 of the model ladder: XGBoost, tuned with Optuna inside song-grouped CV, explained with SHAP."""

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from evaluate import LABEL, cross_validate
from features import FEATURE_COLUMNS

DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}


class XGBModel:
    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}

    def _features(self, table):
        X = table[FEATURE_COLUMNS].copy()
        # Same category list at fit and predict time, so singer codes line up
        X["singer_id"] = pd.Categorical(X["singer_id"].astype(str), categories=self.singer_categories)
        return X

    def fit(self, table):
        self.singer_categories = sorted(table["singer_id"].astype(str).unique())
        self.model = xgb.XGBRegressor(objective="reg:logistic", enable_categorical=True,
                                      tree_method="hist", random_state=0, **self.params)
        self.model.fit(self._features(table), table[LABEL])
        return self

    def predict(self, table):
        return self.model.predict(self._features(table))

    def shap_values(self, table):
        """SHAP values per row and feature (XGBoost's built-in exact TreeSHAP), in log-odds units."""
        dmatrix = xgb.DMatrix(self._features(table), enable_categorical=True)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)
        return pd.DataFrame(contribs[:, :-1], columns=FEATURE_COLUMNS, index=table.index)

    def shap_importance(self, table):
        """Mean |SHAP| per feature, largest first."""
        return self.shap_values(table).abs().mean().sort_values(ascending=False)


def tune(table, n_trials=30, n_splits=5, seed=0):
    """Optuna search for the params with the lowest song-grouped CV MAE on `table`."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        return cross_validate(lambda: XGBModel(**params), table, n_splits)["mae"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
