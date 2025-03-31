import inspect
import warnings
from collections.abc import Callable, Hashable

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from libpysal import graph
from sklearn import metrics

__all__ = ["BaseClassifier"]


def _triangular(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances, bandwidth):
    u = distances / bandwidth
    return np.exp(-((u / 2) ** 2)) / (np.sqrt(2) * np.pi)


def _bisquare(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - u**2) ** 2


def _cosine(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return (np.pi / 4) * np.cos(np.pi / 2 * u)


def _exponential(distances, bandwidth):
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances, bandwidth):
    r = (distances < bandwidth).astype(int)
    return r


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    "gaussian": _gaussian,
    "bisquare": _bisquare,
    "cosine": _cosine,
    "boxcar": _boxcar,
    "exponential": _exponential,
}


class BaseClassifier:
    """Generic geographically weighted modelling meta-class

    NOTE: local models leave out focal, unlike in traditional approaches. This allows
    assessment of geographically weighted metrics on unseen data without a need for
    train/test split, hence providing value for all samples. This is needed for
    futher spatial analysis of the model performance (and generalises to models
    that do not support OOB scoring).

    Parameters
    ----------
    model :  model class
        Scikit-learn model class
    bandwidth : int | float
        bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str, optional
        type of kernel function used to weight observations, by default "bisquare"
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
    strict : bool, optional
        Do not fit any models if at least one neighborhood has invariant ``y``,
        by default False
    keep_models : bool, optional
        Keep all local models (required for prediction), by default True. Note that
        for some models,
        like random forests, the objects can be large.
    temp_folder : str, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``.
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation
    """

    def __init__(
        self,
        model,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        strict: bool = False,
        keep_models: bool = False,
        temp_folder: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fixed = fixed
        self.model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.fit_global_model = fit_global_model
        self.measure_performance = measure_performance
        self.strict = strict
        self.keep_models = keep_models
        self.temp_folder = temp_folder
        self._measure_oob = "oob_score" in inspect.signature(model).parameters
        if self._measure_oob:
            self.model_kwargs["oob_score"] = self._get_score_data

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        """Fit the geographically weighted model

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        geometry : gpd.GeoSeries
            Geographic location
        """
        # build graph
        if self.fixed:  # fixed distance
            self.weights = graph.Graph.build_kernel(
                geometry, kernel=self.kernel, bandwidth=self.bandwidth
            )
        else:  # adaptive KNN
            weights = graph.Graph.build_kernel(
                geometry, kernel="identity", k=self.bandwidth
            )
            # post-process identity weights by the selected kernel
            # and kernel bandwidth derived from each neighborhood
            bandwidth = weights._adjacency.groupby(level=0).transform("max")
            self.weights = graph.Graph(
                adjacency=_kernel_functions[self.kernel](weights._adjacency, bandwidth),
                is_sorted=True,
            )

        self._global_classes = np.unique(y)
        # fit the models
        data = X.copy()
        data["_y"] = y
        data = data.loc[self.weights._adjacency.index.get_level_values(1)]
        data["_weight"] = self.weights._adjacency.values
        grouper = data.groupby(self.weights._adjacency.index.get_level_values(0))

        if self.strict:
            invariant = (
                data["_y"]
                .groupby(self.weights._adjacency.index.get_level_values(0))
                .nunique()
                == 1
            )
            if invariant.any():
                raise ValueError(
                    f"y at locations {invariant.index[invariant]} is invariant."
                )

        # models are fit in parallel
        traning_output = Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(self._fit_local)(
                self.model, group, name, focal_x, self.model_kwargs, self.keep_models
            )
            for (name, group), focal_x in zip(grouper, X.values, strict=False)
        )
        if self.keep_models:
            (
                self._names,
                self._score_data,
                self._feature_importances,
                focal_proba,
                models,
            ) = zip(*traning_output, strict=False)
            self.local_models = pd.Series(models, index=self._names)
            self._geometry = geometry
        else:
            self._names, self._score_data, self._feature_importances, focal_proba = zip(
                *traning_output, strict=False
            )

        self.focal_proba_ = pd.DataFrame(focal_proba).fillna(0).loc[:, np.unique(y)]

        if self.fit_global_model:
            if self._measure_oob:
                self.model_kwargs["oob_score"] = True
            # fit global model as a baseline
            if "n_jobs" in inspect.signature(self.model).parameters:
                self.model_kwargs["n_jobs"] = self.n_jobs
            self.global_model = self.model(**self.model_kwargs)
            self.global_model.fit(X=X, y=y)

        if self.measure_performance:
            # global GW accuracy
            focal_pred = self.focal_proba_.idxmax(axis=1)
            self.score_ = metrics.accuracy_score(y, focal_pred)
            self.f1_macro = metrics.f1_score(y, focal_pred, average="macro")
            self.f1_micro = metrics.f1_score(y, focal_pred, average="micro")
            self.f1_weighted = metrics.f1_score(y, focal_pred, average="weighted")

        return self

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x,
        model_kwargs: dict,
        keep_models: bool,
    ) -> tuple:
        """Fit individual local model

        Parameters
        ----------
        model : model class
            Scikit-learn model class
        data : pd.DataFrame
            data for training
        name : Hashable
            group name, matching the index of the focal geometry
        model_kwargs : dict
            additional keyword arguments for the model init

        Returns
        -------
        tuple
            name, fitted model
        """
        if data["_y"].nunique() == 1:
            warnings.warn(f"y at location {name} is invariant.", stacklevel=2)
        local_model = model(**model_kwargs)
        X = data.drop(columns=["_y", "_weight"])
        y = data["_y"]
        local_model.fit(
            X=X,
            y=y,
            sample_weight=data["_weight"],
        )
        focal_x = pd.DataFrame(
            focal_x.reshape(1, -1),
            columns=X.columns,
            index=[name],
        )
        focal_proba = pd.Series(
            local_model.predict_proba(focal_x).flatten(), index=local_model.classes_
        )

        if hasattr(local_model, "oob_score_"):
            score_data = local_model.oob_score_
        elif self._model_type == "logistic":
            score_data = (
                metrics.accuracy_score(y, local_model.predict(X)),
                pd.DataFrame(
                    local_model.coef_,
                    index=local_model.classes_,
                    columns=local_model.feature_names_in_,
                ),  # coefficients
                pd.Series(
                    local_model.intercept_, index=local_model.classes_
                ),  # intercept
            )
        else:
            score_data = np.nan

        output = [
            name,
            score_data,
            getattr(local_model, "feature_importances_", None),
            focal_proba,
        ]
        if keep_models:
            output.append(local_model)
        else:
            del local_model

        return output

    def _get_score_data(self, true, pred):
        return sum(true.flatten() == pred), len(pred)

    def predict_proba(self, X: pd.DataFrame, geometry: gpd.GeoSeries) -> pd.DataFrame:
        if self.fixed:
            input_ids, local_ids = self._geometry.sindex.query(
                geometry, predicate="dwithin", distance=self.bandwidth
            )
            distance = _kernel_functions[self.kernel](
                self._geometry.iloc[local_ids].distance(
                    geometry.iloc[input_ids], align=False
                ),
                self.bandwidth,
            )
        else:
            raise NotImplementedError

        split_indices = np.where(np.diff(input_ids))[0] + 1
        local_model_ids = np.split(local_ids, split_indices)
        distances = np.split(distance.values, split_indices)
        data = np.split(X, range(1, len(X)))

        probabilities = []
        for x_, models_, distances_ in zip(
            data, local_model_ids, distances, strict=True
        ):
            # there are likely ways of speeding this up using parallel processing
            # but I failed to do so efficiently. We are hitting GIL due to accessing
            # same local models many times so iterative loop is in the end faster
            probabilities.append(
                self._predict_proba(x_, models_, distances_, X.columns)
            )

        return pd.DataFrame(
            probabilities, columns=self._global_classes, index=X.index
        ).fillna(0)

    def _predict_proba(self, x_, models_, distances_, columns):
        x_ = pd.DataFrame(np.array(x_).reshape(1, -1), columns=columns)
        pred = pd.DataFrame(
            [
                pd.Series(
                    self.local_models[i].predict_proba(x_).flatten(),
                    index=self.local_models[i].classes_,
                )
                for i in models_
            ]
        ).fillna(0)
        weighted = np.average(pred, axis=0, weights=distances_)

        # normalize
        weighted = weighted / weighted.sum()
        return pd.Series(weighted, index=pred.columns)

    def predict(self, X, geometry):
        proba = self.predict_proba(X, geometry)

        return proba.idxmax(axis=1)
