"""Granular ball generation (purity-driven recursive 2-means splitting).

Re-implementation of the idea in the EF-GBTSVM paper / reference repo, with:
  * explicit random_state (reproducible),
  * iterative (stack based) splitting instead of list mutation,
  * labels are arbitrary class ids (not hard-coded to -1/+1).
"""
import numpy as np
from sklearn.cluster import KMeans


def _ball_stats(X, y):
    center = X.mean(axis=0)
    labels, counts = np.unique(y, return_counts=True)
    label = labels[counts.argmax()]
    purity = counts.max() / len(y)
    radius = np.linalg.norm(X - center, axis=1).mean()
    return center, label, purity, radius


def generate_granular_balls(X, y, purity=0.925, min_samples=2, random_state=0,
                            return_purity=False):
    """Split data into granular balls.

    A ball with purity < `purity` (fraction of majority class) and more than
    one sample is split in two with 2-means. Balls with fewer than
    `min_samples` points are discarded at the end (outlier removal), unless
    that would remove every ball of some class (then they are kept).

    Parameters
    ----------
    return_purity : bool, default False
        If True, also return each kept ball's purity (fraction of its
        majority label) as a 5th array. Default False keeps the original
        4-tuple return for backward compatibility; passing True (or any
        existing positional call) does not change which balls are formed —
        splitting only ever depends on `purity`/`min_samples`/`random_state`.

    Returns
    -------
    centers (k, d), labels (k,), radii (k,), sizes (k,)[, purities (k,)]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)

    stack = [np.arange(len(X))]
    final = []
    while stack:
        idx = stack.pop()
        c, l, p, r = _ball_stats(X[idx], y[idx])
        if p < purity and len(idx) > 1:
            km = KMeans(n_clusters=2, n_init=3,
                        random_state=rng.randint(0, 2**31 - 1)).fit(X[idx])
            lab = km.labels_
            if lab.min() == lab.max():           # degenerate (duplicates)
                stack.append(idx[:1])
                stack.append(idx[1:])
            else:
                stack.append(idx[lab == 0])
                stack.append(idx[lab == 1])
        else:
            final.append((c, l, r, len(idx), p))

    kept = [b for b in final if b[3] >= min_samples]
    classes = np.unique(y)
    if len({b[1] for b in kept}) < len(classes):  # never lose a whole class
        kept = final
    centers = np.array([b[0] for b in kept])
    labels = np.array([b[1] for b in kept])
    radii = np.array([b[2] for b in kept])
    sizes = np.array([b[3] for b in kept])
    if return_purity:
        purities = np.array([b[4] for b in kept])
        return centers, labels, radii, sizes, purities
    return centers, labels, radii, sizes
