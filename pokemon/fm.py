"""Second-order classical Factorization Machine with observed battle features.

All inputs come from P1's known team and the first 30 recorded turns.
No neural networks, external tables, test labels or future events are used.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix

STATUSES = ("slp", "par", "tox", "psn", "frz", "brn")


def battle_fm_features(battle):
    """Observed identities, known HP and statuses for an asymmetric FM."""
    timeline = battle["battle_timeline"]
    if not timeline or len(timeline) > 30:
        raise ValueError("Expected 1..30 observed turns")
    roster = [mon["name"] for mon in battle["p1_team_details"]]
    if len(roster) != 6 or len(set(roster)) != 6:
        raise ValueError("Expected six distinct P1 team members")
    last = {"p1": {}, "p2": {}}
    for turn in timeline:
        for side in ("p1", "p2"):
            state = turn[f"{side}_pokemon_state"]
            last[side][state["name"]] = state
    f = {}
    hp1, hp2 = [], []
    for name in roster:
        st = last["p1"].get(name)
        hp = float(st["hp_pct"]) if st is not None else 1.0
        hp1.append(hp)
        f[f"p1_species:{name}"] = 1.0
        f[f"p1_health:{name}"] = hp
        if hp == 0:
            f[f"p1_fainted:{name}"] = 1.0
        if st is not None and st.get("status") in STATUSES:
            f[f"p1_status:{name}:{st['status']}"] = 1.0

    for name, st in last["p2"].items():
        hp = float(st["hp_pct"])
        hp2.append(hp)
        f[f"p2_seen:{name}"] = 1.0
        f[f"p2_health:{name}"] = hp
        if hp == 0:
            f[f"p2_fainted:{name}"] = 1.0
        if st.get("status") in STATUSES:
            f[f"p2_status:{name}:{st['status']}"] = 1.0

    unknown = 6 - len(hp2)
    if unknown < 0:
        raise ValueError("More than six different P2 Pokemon observed")
    f.update({
        "ctx:p1_hp_fraction": sum(hp1) / 6,
        # Unknown P2 health is NOT imputed as known: retain separate count.
        "ctx:p2_observed_hp_fraction": sum(hp2) / 6,
        "ctx:hp_difference_known": (sum(hp1) - sum(hp2)) / 6,
        "ctx:p1_fainted": sum(h == 0 for h in hp1) / 6,
        "ctx:p2_fainted_observed": sum(h == 0 for h in hp2) / 6,
        "ctx:p2_unknown_count": unknown / 6,
        "ctx:p1_status_count": sum(
            st.get("status") in STATUSES for st in last["p1"].values()) / 6,
        "ctx:p2_status_count": sum(
            st.get("status") in STATUSES for st in last["p2"].values()) / 6,
    })
    return f


@dataclass
class LogisticFM:
    """Sparse second-order FM, logistic loss, L2 penalty, L-BFGS-B optimizer."""
    rank: int = 8
    reg_linear: float = 0.001
    reg_latent: float = 0.0005
    maxiter: int = 125
    seed: int = 42

    def fit(self, X, y):
        X = csr_matrix(X, dtype=np.float64)
        X2 = X.multiply(X).tocsr()
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        if n == 0 or len(y) != n or np.any((y != 0) & (y != 1)):
            raise ValueError("Expected a nonempty binary labeled matrix")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        rng = np.random.default_rng(self.seed)
        prior = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        initial = np.r_[np.log(prior / (1 - prior)),
                        np.zeros(p),
                        rng.normal(0, 0.05, size=(p, self.rank)).ravel()]

        def objective(theta):
            bias = theta[0]
            w = theta[1:p + 1]
            V = theta[p + 1:].reshape(p, self.rank)
            XV = np.asarray(X @ V)
            score = (bias + np.asarray(X @ w).ravel()
                     + 0.5 * np.sum(XV * XV - np.asarray(X2 @ (V * V)), axis=1))
            err = expit(score) - y
            loss = (np.mean(np.logaddexp(0, score) - y * score)
                    + 0.5 * self.reg_linear * np.sum(w * w)
                    + 0.5 * self.reg_latent * np.sum(V * V))
            gw = np.asarray(X.T @ err).ravel() / n + self.reg_linear * w
            gV = ((np.asarray(X.T @ (err[:, None] * XV))
                   - np.asarray(X2.T @ err)[:, None] * V) / n
                  + self.reg_latent * V)
            return float(loss), np.r_[err.mean(), gw, gV.ravel()]

        opt = minimize(objective, initial, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter, "ftol": 1e-9})
        self.intercept_ = float(opt.x[0])
        self.linear_ = opt.x[1:p + 1].copy()
        self.latent_ = opt.x[p + 1:].reshape(p, self.rank).copy()
        self.n_features_in_ = p
        self.optimization_ = {
            "success": bool(opt.success), "message": str(opt.message),
            "iterations": int(opt.nit), "final_loss": float(opt.fun)}
        return self

    def decision_function(self, X):
        X = csr_matrix(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension mismatch")
        XV = np.asarray(X @ self.latent_)
        return (self.intercept_ + np.asarray(X @ self.linear_).ravel()
                + 0.5 * np.sum(
                    XV ** 2 - np.asarray(X.multiply(X) @ (self.latent_ ** 2)),
                    axis=1))

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack((1 - p, p))
