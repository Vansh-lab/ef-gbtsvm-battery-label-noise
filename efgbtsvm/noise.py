"""Training-label noise estimate and the noise -> purity mapping used by
EFGBTSVM(purity_mode="adaptive").

The estimate is the out-of-fold misclassification rate of a 5-NN classifier.
It is a proxy, not the true noise rate: on clean labels it still returns the
kNN's own error (class overlap), and flipped labels push it up.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


def estimate_label_noise(X, y, k=5, random_state=0):
    """Out-of-fold kNN (n_neighbors=5) error rate over a stratified k-fold CV."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_splits = min(k, np.bincount(y).min())       # every fold needs both classes
    if n_splits < 2:
        return 0.0
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), X, y, cv=cv)
    return float((pred != y).mean())


def purity_from_noise(noise, alpha=1.0, lo=0.55, hi=0.95):
    """purity = clip(hi - alpha * noise, lo, hi)."""
    return float(np.clip(hi - alpha * noise, lo, hi))
