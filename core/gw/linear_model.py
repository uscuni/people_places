from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier


class GWLogisticRegression(BaseClassifier):
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
        self._model_type = "logistic"

        super().__init__(
            model=LogisticRegression,
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

        self.local_score_ = pd.Series([x[0] for x in self._score_data], index=X.index)
        self.local_coef_ = pd.concat(
            [x[1] for x in self._score_data], axis=1, keys=self._names
        ).T
        self.local_intercept_ = pd.DataFrame(
            [x[2] for x in self._score_data], index=self._names, columns=np.unique(y)
        )
        return self
