from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .base import BaseClassifier


class GWRandomForestClassifier(BaseClassifier):
    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        strict: bool = False,
        keep_models: bool = False,
        **kwargs,
    ):
        self._model_type = "random_forest"

        super().__init__(
            model=RandomForestClassifier,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        super().fit(X=X, y=y, geometry=geometry)

        if self.measure_performance:
            # OOB accuracy for RF can be measured both local and global
            true, n = zip(*self._score_data, strict=False)
            self.oob_score_ = sum(true) / sum(n)
            self.local_oob_score_ = pd.Series(
                np.array(true) / np.array(n), index=self._names
            )

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        return self


class GWGradientBoostingClassifier(BaseClassifier):
    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        strict: bool = False,
        keep_models: bool = False,
        **kwargs,
    ):
        self._model_type = "gradient_boosting"

        super().__init__(
            model=GradientBoostingClassifier,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        super().fit(X=X, y=y, geometry=geometry)

        if self.measure_performance:
            # OOB accuracy for stochastic GB can be measured as local only. GB is
            # stochastic if subsample < 1.0. Otherwise, oob_score_ is not available
            # as all samples were used in training
            self.local_oob_score_ = pd.Series(self._score_data, index=self._names)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        return self
