"""Twin SVM (linear) solved through its two dual QPPs with CVXOPT.

Notation (as in the paper): plane 1 is close to class A (+), far from class B (-);
plane 2 is close to B, far from A.
"""
import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False


def _box_qp(P, ub):
    """min 0.5 a'Pa - e'a  s.t. 0 <= a <= ub.

    `ub` may be a scalar (same bound for every sample, the original
    behavior) or a length-m array (one bound per sample, used for
    purity-weighted balls).
    """
    m = P.shape[0]
    P = (P + P.T) / 2 + 1e-8 * np.eye(m)
    ub_vec = np.full(m, ub) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    G = np.vstack([np.eye(m), -np.eye(m)])
    h = np.hstack([ub_vec, np.zeros(m)])
    sol = solvers.qp(matrix(P, tc="d"), matrix(-np.ones(m), tc="d"),
                     matrix(G, tc="d"), matrix(h, tc="d"))
    return np.array(sol["x"]).ravel()


class TWSVM:
    def __init__(self, C1=1.0, C2=1.0, delta=0.05, normalize=True):
        self.C1, self.C2, self.delta, self.normalize = C1, C2, delta, normalize

    def fit(self, A, B, weights=None):
        """A: samples of positive class, B: negative class (2D arrays).

        weights : optional (wA, wB), default None
            Per-sample weights in (0, 1], one array matching A's rows and
            one matching B's rows. When given, the dual box constraint for
            each sample becomes 0 <= a_i <= C * weight_i instead of the
            plain 0 <= a_i <= C (so a low-weight, e.g. low-purity, sample
            constrains the opposite-class plane less). weights=None (the
            default) reproduces the original unweighted behavior exactly —
            every a_i is bounded by C1/C2 as before.
        """
        H = np.hstack([A, np.ones((len(A), 1))])
        G = np.hstack([B, np.ones((len(B), 1))])
        d = H.shape[1]
        I = np.eye(d)
        wA, wB = (None, None) if weights is None else weights
        ub1 = self.C1 if wB is None else self.C1 * np.asarray(wB, dtype=float)
        ub2 = self.C2 if wA is None else self.C2 * np.asarray(wA, dtype=float)

        # plane 1:  u1 = -(H'H + dI)^-1 G' alpha
        HHG = np.linalg.solve(H.T @ H + self.delta * I, G.T)
        a1 = _box_qp(G @ HHG, ub1)
        u1 = -HHG @ a1
        # plane 2:  u2 =  (G'G + dI)^-1 H' gamma
        GGH = np.linalg.solve(G.T @ G + self.delta * I, H.T)
        a2 = _box_qp(H @ GGH, ub2)
        u2 = GGH @ a2

        self.w1_, self.b1_ = u1[:-1], u1[-1]
        self.w2_, self.b2_ = u2[:-1], u2[-1]
        return self

    def distances(self, X):
        d1 = np.abs(X @ self.w1_ + self.b1_)
        d2 = np.abs(X @ self.w2_ + self.b2_)
        if self.normalize:   # Eq. 22: perpendicular distance |w'x+b| / ||w||
            d1 = d1 / (np.linalg.norm(self.w1_) + 1e-12)
            d2 = d2 / (np.linalg.norm(self.w2_) + 1e-12)
        return d1, d2

    def score(self, X):
        """>0  -> closer to plane 1 (class A)."""
        d1, d2 = self.distances(X)
        return d2 - d1
