"""Step 1 of the model ladder: predict each singer's mean accuracy from the training data."""

import numpy as np

from evaluate import LABEL


class SingerMeanBaseline:
    def fit(self, table):
        self.singer_means = table.groupby("singer_id", observed=True)[LABEL].mean()
        self.global_mean = table[LABEL].mean()  # for singers not seen in training
        return self

    def predict(self, table):
        means = table["singer_id"].astype(str).map(self.singer_means.rename(str))
        return means.fillna(self.global_mean).to_numpy(dtype=float)
