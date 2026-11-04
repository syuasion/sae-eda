"""Outlier scores: one per document, higher means more unusual."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def iforest(x, seed: int = 0) -> np.ndarray:
    return -IsolationForest(n_estimators=300, random_state=seed).fit(
        x).score_samples(x)


def lof(x, seed: int = 0) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=20, metric="cosine").fit(x)
    return -model.negative_outlier_factor_


def ocsvm(x, seed: int = 0) -> np.ndarray:
    return -OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(
        x).score_samples(x)


OUTLIERS = {"iforest": iforest, "lof": lof, "ocsvm": ocsvm}
