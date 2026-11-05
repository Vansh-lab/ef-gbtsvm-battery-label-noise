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
purity_mode="adaptive_offset" also ignores `purity`: it uses the same noise
estimate and sets purity = clip(0.95 - noise_alpha * noise + purity_offset,
0.55, 0.98), with purity_offset searched jointly with the other hyperparameters.
EF-GBTSVM + purity_mode="adaptive_offset" -> EF-GBTSVM-J (Phase A, candidate b).

ball_weight="mass" (requires use_balls=True, weighted=False) weights each ball
by s_i = (n_i * p_i) ** mass_power, the number of points carrying the ball's
label raised to mass_power, normalised to mean 1 within each class. The weight
enters both the ball's own-class term and its slack bound (C_i = C * s_i), so
with mass_power=1 a ball counts like the majority points it summarises and
with mass_power=0 every ball counts once, as in plain EF-GBTSVM.
ball_weight="none" (default) leaves the model unchanged.
EF-GBTSVM + ball_weight="mass" -> EF-GBTSVM-M (Phase A, candidate c).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .granular_ball import generate_granular_balls
from .noise import estimate_label_noise, purity_from_noise, purity_from_noise_offset
from .rvfl_features import RandomHiddenLayer
from .twsvm import TWSVM


class EFGBTSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C1=1e-5, C2=None, n_hidden=23, activation=1,
                 purity=0.925, min_samples=2, delta=0.05,
                 use_balls=True, features="enhanced", normalize=True,
                 weighted=False, purity_mode="fixed", noise_alpha=1.0,
                 purity_offset=0.0, ball_weight="none", mass_power=1.0,
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
        self.purity_offset = purity_offset
        self.ball_weight = ball_weight
        self.mass_power = mass_power
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
        elif self.purity_mode == "adaptive_offset":
            self.estimated_noise_ = estimate_label_noise(
                X, yb, random_state=self.random_state)
            self.purity_used_ = purity_from_noise_offset(
                self.estimated_noise_, self.purity_offset, self.noise_alpha)
        elif self.purity_mode == "fixed":
            self.estimated_noise_ = None
            self.purity_used_ = self.purity
        else:
            raise ValueError(f"purity_mode must be 'fixed', 'adaptive' or "
                             f"'adaptive_offset', got {self.purity_mode!r}")
        if self.ball_weight not in ("none", "mass"):
            raise ValueError(f"ball_weight must be 'none' or 'mass', "
                             f"got {self.ball_weight!r}")
        if self.ball_weight == "mass" and (self.weighted or not self.use_balls):
            raise ValueError("ball_weight='mass' needs use_balls=True and weighted=False")

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
        weights = own_weights = None
        if self.weighted and purities is not None:
            # weight = purity itself (no floor): a ball at the 50/50 class
            # split gets half the box bound, a pure ball gets the full C.
            weights = (purities[yin == 1], purities[yin == 0])
        if self.ball_weight == "mass":
            mass = (sizes * purities) ** self.mass_power
            sA, sB = mass[yin == 1], mass[yin == 0]
            weights = own_weights = (sA / sA.mean(), sB / sB.mean())
        self.ball_weights_ = None if weights is None else np.concatenate(weights)
        self.tw_ = TWSVM(self.C1, C2, self.delta, self.normalize).fit(
            T[yin == 1], T[yin == 0], weights=weights, own_weights=own_weights)
        return self

    def decision_function(self, X):
        """>0 -> classes_[1]."""
        return self.tw_.score(self._map(np.asarray(X, dtype=float)))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
