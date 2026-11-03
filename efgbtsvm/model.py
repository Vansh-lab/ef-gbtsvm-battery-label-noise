"""EF-GBTSVM as a scikit-learn style estimator, with switches for the ablations.

use_balls=True,  features="enhanced"  -> EF-GBTSVM  (proposed)
use_balls=True,  features="hidden"    -> HF-GBTSVM
use_balls=True,  features="original"  -> GBTSVM
use_balls=False, features="enhanced"  -> EF-TSVM
use_balls=False, features="hidden"    -> HF-TSVM
use_balls=False, features="original"  -> TSVM

weighted=True (requires use_balls=True) adds purity-weighted balls: each
ball's TWSVM dual upper bound becomes C_i = C * purity_i instead of a flat C,
so a low-purity (noisier) ball constrains the opposite-class plane less.
weighted=False (default) is the plain model above, unchanged.
EF-GBTSVM (use_balls=True, features="enhanced") + weighted=True -> EF-GBTSVM-W.

purity_mode="fixed" (default) uses the `purity` hyperparameter as given.
purity_mode="adaptive" ignores `purity`: fit() estimates the training-label
noise (5-NN out-of-fold error on the raw training data, before balls exist)
and sets purity = clip(0.95 - noise_alpha * noise, 0.55, 0.95).
EF-GBTSVM + purity_mode="adaptive" -> EF-GBTSVM-A.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .granular_ball import generate_granular_balls
from .noise import estimate_label_noise, purity_from_noise
from .rvfl_features import RandomHiddenLayer
from .twsvm import TWSVM


class EFGBTSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C1=1e-5, C2=None, n_hidden=23, activation=1,
                 purity=0.925, min_samples=2, delta=0.05,
                 use_balls=True, features="enhanced", normalize=True,
                 weighted=False, purity_mode="fixed", noise_alpha=1.0,
                 random_state=0):
        self.C1 = C1
        self.C2 = C2
        self.n_hidden = n_hidden
        self.activation = activation
        self.purity = purity
        self.min_samples = min_samples
        self.delta = delta
        self.use_balls = use_balls
        self.features = features
        self.normalize = normalize
        self.weighted = weighted
        self.purity_mode = purity_mode
        self.noise_alpha = noise_alpha
        self.random_state = random_state

    def _map(self, X):
        if self.features == "original":
            return X
        if self.features == "hidden":
            return self.layer_.hidden(X)
        return self.layer_.enhanced(X)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("binary classification only")
        yb = (y == self.classes_[1]).astype(int)   # 1 -> classes_[1]

        if self.purity_mode == "adaptive":
            self.estimated_noise_ = estimate_label_noise(
                X, yb, random_state=self.random_state)
            self.purity_used_ = purity_from_noise(self.estimated_noise_,
                                                  self.noise_alpha)
        elif self.purity_mode == "fixed":
            self.estimated_noise_ = None
            self.purity_used_ = self.purity
        else:
            raise ValueError(f"purity_mode must be 'fixed' or 'adaptive', "
                             f"got {self.purity_mode!r}")

        purities = None
        if self.use_balls:
            Xin, yin, _, sizes, purities = generate_granular_balls(
                X, yb, self.purity_used_, self.min_samples, self.random_state,
                return_purity=True)
            self.n_balls_ = len(Xin)
        else:
            Xin, yin = X, yb
            self.n_balls_ = len(X)
        self.ball_purities_ = purities   # diagnostics only (e.g. "how many balls
                                          # are actually impure"); not used by fit/predict

        self.layer_ = RandomHiddenLayer(self.n_hidden, self.activation,
                                        random_state=self.random_state).fit(Xin)
        T = self._map(Xin)
        C2 = self.C1 if self.C2 is None else self.C2   # paper uses d1 = d2
        weights = None
        if self.weighted and purities is not None:
            # weight = purity itself (no floor): a ball at the 50/50 class
            # split gets half the box bound, a pure ball gets the full C.
            weights = (purities[yin == 1], purities[yin == 0])
        self.tw_ = TWSVM(self.C1, C2, self.delta, self.normalize).fit(
            T[yin == 1], T[yin == 0], weights=weights)
        return self

    def decision_function(self, X):
        """>0 -> classes_[1]."""
        return self.tw_.score(self._map(np.asarray(X, dtype=float)))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
