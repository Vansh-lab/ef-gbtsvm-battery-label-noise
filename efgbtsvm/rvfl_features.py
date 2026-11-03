"""Random hidden-layer ("RVFL") feature map + a plain RVFL classifier baseline."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946


def _selu(x):
    return _SCALE * np.where(x > 0, x, _ALPHA * (np.exp(np.minimum(x, 0)) - 1))


def _leaky(x):
    return np.where(x >= 0, x, x / 10.0)


ACTIVATIONS = {
    1: ("selu", _selu),
    2: ("relu", lambda x: np.maximum(0, x)),
    3: ("sigmoid", lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))),
    4: ("sin", np.sin),
    5: ("hardlim", lambda x: (np.sign(x) + 1) / 2),
    6: ("tribas", lambda x: np.maximum(1 - np.abs(x), 0)),
    7: ("radbas", lambda x: np.exp(-(x ** 2))),
    8: ("sign", np.sign),
    9: ("leaky_relu", _leaky),
    10: ("tanh", np.tanh),
}


class RandomHiddenLayer:
    """W ~ U[-1,1], b ~ s*U[0,1] (s=0.1), same as the reference code."""

    def __init__(self, n_hidden=23, activation=1, scale_bias=0.1, random_state=0):
        self.n_hidden = n_hidden
        self.activation = activation
        self.scale_bias = scale_bias
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        self.W_ = rng.rand(X.shape[1], self.n_hidden) * 2 - 1
        self.b_ = self.scale_bias * rng.rand(1, self.n_hidden)
        return self

    def hidden(self, X):
        return ACTIVATIONS[self.activation][1](X @ self.W_ + self.b_)

    def enhanced(self, X):
        """[original, hidden] -> the 'RVFL space'."""
        return np.hstack([X, self.hidden(X)])


class RVFLClassifier(BaseEstimator, ClassifierMixin):
    """Binary RVFL with direct links, closed-form ridge solution (baseline)."""

    def __init__(self, C=1.0, n_hidden=100, activation=1, random_state=0):
        self.C = C
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        self.layer_ = RandomHiddenLayer(self.n_hidden, self.activation,
                                        random_state=self.random_state).fit(X)
        H = np.hstack([self.layer_.enhanced(X), np.ones((len(X), 1))])
        lam = 1.0 / self.C
        self.beta_ = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ t)
        return self

    def decision_function(self, X):
        H = np.hstack([self.layer_.enhanced(X), np.ones((len(X), 1))])
        return H @ self.beta_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
