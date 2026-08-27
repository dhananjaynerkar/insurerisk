from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor


def _safe_dense(X: Any) -> np.ndarray:
    if sp_sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=float)


class IsolationForestAnomalyDetector(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 180,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "IsolationForestAnomalyDetector":
        X_fit = X
        if y is not None:
            y_arr = np.asarray(y).astype(int)
            mask = y_arr == 0
            if int(mask.sum()) >= max(50, int(0.25 * len(y_arr))):
                X_fit = X[mask]

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=float(np.clip(self.contamination, 0.01, 0.20)),
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(X_fit)

        train_scores = -self.model_.score_samples(X_fit)
        self.score_min_ = float(np.quantile(train_scores, 0.01))
        self.score_max_ = float(np.quantile(train_scores, 0.99))
        if not np.isfinite(self.score_min_):
            self.score_min_ = float(np.min(train_scores))
        if not np.isfinite(self.score_max_) or np.isclose(self.score_max_, self.score_min_):
            self.score_max_ = self.score_min_ + 1e-6
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = -self.model_.score_samples(X)
        denom = max(self.score_max_ - self.score_min_, 1e-9)
        p_anom = np.clip((scores - self.score_min_) / denom, 0.0, 1.0)
        return np.column_stack([1.0 - p_anom, p_anom])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AutoencoderAnomalyDetector(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        svd_components: int = 40,
        hidden_layer_sizes: tuple[int, ...] = (24, 8, 24),
        max_iter: int = 80,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.svd_components = svd_components
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.contamination = contamination
        self.random_state = random_state

    def _to_latent(self, X: Any, fit: bool = False) -> np.ndarray:
        if self.use_svd_:
            if fit:
                return self.svd_.fit_transform(X)
            return self.svd_.transform(X)
        return _safe_dense(X)

    def fit(self, X: Any, y: Any = None) -> "AutoencoderAnomalyDetector":
        X_fit = X
        if y is not None:
            y_arr = np.asarray(y).astype(int)
            mask = y_arr == 0
            if int(mask.sum()) >= max(50, int(0.25 * len(y_arr))):
                X_fit = X[mask]

        n_features = int(X_fit.shape[1])
        self.use_svd_ = n_features > (self.svd_components + 4)
        if self.use_svd_:
            n_comp = min(max(2, self.svd_components), max(2, n_features - 1))
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        else:
            self.svd_ = None

        Z_fit = self._to_latent(X_fit, fit=True)
        self.reconstructor_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=256,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=self.random_state,
        )
        self.reconstructor_.fit(Z_fit, Z_fit)
        Z_hat = self.reconstructor_.predict(Z_fit)
        err = np.mean(np.square(Z_fit - Z_hat), axis=1)
        self.err_min_ = float(np.quantile(err, 0.01))
        self.err_max_ = float(np.quantile(err, 0.99))
        if not np.isfinite(self.err_min_):
            self.err_min_ = float(np.min(err))
        if not np.isfinite(self.err_max_) or np.isclose(self.err_max_, self.err_min_):
            self.err_max_ = self.err_min_ + 1e-6
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        Z = self._to_latent(X, fit=False)
        Z_hat = self.reconstructor_.predict(Z)
        err = np.mean(np.square(Z - Z_hat), axis=1)
        denom = max(self.err_max_ - self.err_min_, 1e-9)
        p_anom = np.clip((err - self.err_min_) / denom, 0.0, 1.0)
        return np.column_stack([1.0 - p_anom, p_anom])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
