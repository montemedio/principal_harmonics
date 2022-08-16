from typing import Optional

import librosa
import numpy as np
import pya
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


def join_vec(freqs, coefs):
    assert freqs.ndim == coefs.ndim == 2
    assert freqs.shape == coefs.shape
    return np.concatenate([freqs, coefs], axis=1)

def split_vec(X):
    assert X.ndim == 2
    n_trajs = X.shape[1] // 2
    return X[:, :n_trajs], X[:, n_trajs:]


class DropDCTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.dc = None

    def fit(self, X, y=None):
        self.dc = X[:, 0]
        return self
    
    def transform(self, X):
        return X[:, 1:]

    def inverse_transform(self, X):
        T = X.shape[0]
        return np.hstack((np.zeros(T).reshape(-1, 1), X))


class AweightTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        freqs, coefs = split_vec(X)
        weights = librosa.A_weighting(freqs)
        coefs *= weights
        return join_vec(freqs, coefs)

    def inverse_transform(self, X):
        freqs, coefs = split_vec(X)
        weights = librosa.A_weighting(freqs)
        coefs /= weights
        return join_vec(freqs, coefs)


class HoleImputer(TransformerMixin):
    def __init__(self, hole_size_limit, direction='backward') -> None:
        super().__init__()
        self.limit = hole_size_limit
        self.direction = direction

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.interpolate(limit=self.limit,
                              limit_area='inside').to_numpy()

    def inverse_transform(self, X):
        return X


class RidgeSubNoiseImputer(TransformerMixin):
    def __init__(self, constant_value=-120) -> None:
        self.constant_value: float                = constant_value
        self.estimators: list[Optional[Pipeline]] = []
        super().__init__()

    def fit(self, X, y=None):
        assert X.ndim == 2

        self.estimators = []
        ts = np.arange(X.shape[0]).reshape((-1, 1))
        found_nonempty_partial = False
        
        for i in range(X.shape[1]):
            pipeline = None
            nans = np.isnan(X[:, i])
            if (~nans).sum() > 10:
                found_nonempty_partial = True
                pipeline = make_pipeline(
                    StandardScaler(),
                    Ridge()
                )
                pipeline.fit(ts[~nans], X[~nans, i])
            self.estimators.append(pipeline)

        if not found_nonempty_partial:
            raise ValueError("No partial contains any values.")

        return self

    def transform(self, X):
        assert len(self.estimators) == X.shape[1]
        XT = X.copy()
        ts = np.arange(X.shape[0]).reshape((-1, 1))
        for i, estimator in enumerate(self.estimators):
            if not estimator:
                XT[:, i] = self.constant_value
            else:
                nans = np.isnan(X[:, i])
                if not nans.any():
                    continue
                XT[nans, i] = estimator.predict(ts[nans])
        return XT

    def inverse_transform(self, X):
        return X


class IterativeSubNoiseImputer(TransformerMixin):
    def __init__(self, constant_value=-120, use_time=False, *args, **kwargs) -> None:
        self.iterative_imputer = IterativeImputer(*args, **kwargs)
        self.nonempty_partials = []
        self.constant_value = constant_value
        self.use_time = use_time
        self.n_partials = -1
        super().__init__()

    def fit(self, X, y=None):
        assert X.ndim == 2
        self.n_partials = X.shape[1]
        self.nonempty_partials = []
        for i in range(self.n_partials):
            if (~np.isnan(X[:, i])).sum() > 10:
                self.nonempty_partials.append(i)

        if not self.nonempty_partials:
            raise ValueError("No partial contains any values.")

        X = X[:, self.nonempty_partials]
        if self.use_time:
            ts = np.arange(X.shape[0]).reshape((-1, 1))
            X = np.hstack((ts, X))

        self.iterative_imputer.fit(X)
        return self

    def transform(self, X):
        res_shape = X.shape

        X = X[:, self.nonempty_partials]
        if self.use_time:
            ts = np.arange(X.shape[0]).reshape((-1, 1))
            X = np.hstack((ts, X))

        transformed = self.iterative_imputer.transform(X)
        if self.use_time:
            transformed = transformed[:, 1:]

        res = np.full(res_shape, self.constant_value, dtype=float)
        res[:, self.nonempty_partials] = transformed
        
        return res

    def inverse_transform(self, X):
        return X


class FillSilenceTransformer(TransformerMixin):
    def __init__(self, silence=1e-12) -> None:
        super().__init__()
        self.silence = silence

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[np.isneginf(X)] = self.silence
        X[np.isnan(X)]    = self.silence
        return X

    def inverse_transform(self, X):
        return X


class DBTransformer(TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pya.ampdb(X)

    def inverse_transform(self, X):
        return pya.dbamp(X)


class FilterIncompleteTimbresTransformer(TransformerMixin):
    def __init__(self, silence_value=-240, values_present=0.8) -> None:
        super().__init__()
        self.silence_value = silence_value
        self.values_present = values_present
        self.include_trajs = []

    def fit(self, X, y=None):
        n_frames, n_trajs = X.shape
        self.include_trajs = np.zeros(n_trajs, dtype=bool)
        nans = np.isnan(X)

        for i in range(n_trajs):
            n_values_present = len(np.flatnonzero(~nans[:, i]))
            self.include_trajs[i] = n_values_present / n_frames >= self.values_present
            
        return self

    def transform(self, X):
        assert X.ndim == 2

        # since x + nan = nan, we can just sum along the timbre axis
        # to check if the frame contains no nans
        complete_frames = ~np.isnan(X[:, self.include_trajs].sum(axis=1))
        res = X[complete_frames].copy()
        res[:, ~self.include_trajs] = self.silence_value

        if res.shape[0] == 0:
            raise NoCompleteTimbreError()

        return res

    def inverse_transform(self, X):
        return X


class NoCompleteTimbreError(BaseException):
    message = "When filtering timbres, all frames contained at least one nan"


class ConstantAmplitudeTransformer(TransformerMixin):
    def __init__(self, ampl=1, enable_inverse=False, ord=1) -> None:
        if ampl == 0.0:
            raise ValueError("Amplitude cannot be zero.")

        self.amplitude = ampl
        self.lengths = None
        self.enable_inverse = enable_inverse
        self.ord = ord
        super().__init__()

    def fit(self, X, y=None):
        assert X.ndim == 2

        # consider nans as zero so that norm calculation works
        X = X.copy()
        X[np.isnan(X)] = 0.0
        self.lengths = np.linalg.norm(X, ord=self.ord, axis=1)

        # avoid zero division
        self.lengths[self.lengths == 0] = 1.0

        return self

    def transform(self, X):
        assert self.lengths is not None
        return X / self.lengths.reshape((-1, 1)) * self.amplitude

    def inverse_transform(self, X):
        assert self.lengths is not None
        if self.enable_inverse:
            return X * self.lengths.reshape((-1, 1)) / self.amplitude
        else:
            print('before:', X)
            return X * np.mean(self.lengths) / self.amplitude