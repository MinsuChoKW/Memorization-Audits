"""Scoring functions and the paper-clustered bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import B, DEV_MODELS


def wauc(y, p, w):
    """Weighted AUC with ties counted as one half (equal to sklearn's roc_auc_score with sample_weight)."""
    y, p, w = np.asarray(y), np.asarray(p, float), np.asarray(w, float)
    o = np.argsort(p, kind="mergesort")
    p, y, w = p[o], y[o], w[o]
    _, start = np.unique(p, return_index=True)
    gn = np.add.reduceat(np.where(y == 0, w, 0.0), start)
    gp = np.add.reduceat(np.where(y == 1, w, 0.0), start)
    below = np.cumsum(gn) - gn
    return float(np.sum(gp * (below + 0.5 * gn)) / (gp.sum() * gn.sum()))


def paper_weights(x):
    """Equal paper weights, 1 / (claims of the paper in x)."""
    return (1 / x.groupby("orig_id").orig_id.transform("size")).values


def auc(x, col, y="y"):
    """AUC with equal paper weights recomputed within x."""
    return wauc(x[y].values, x[col].values, paper_weights(x))


def paper_draws(groups, n=B, seed=0):
    """Paper-clustered bootstrap draws as (row index, cluster id) pairs. Every drawn copy of a paper gets its own
    cluster id, so that equal paper weights recomputed inside a resample count each copy as a separate paper."""
    groups = np.asarray(groups)
    papers = list(dict.fromkeys(groups))
    pos = {p: np.flatnonzero(groups == p) for p in papers}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        draw = rng.choice(papers, len(papers))
        out.append((np.concatenate([pos[p] for p in draw]),
                    np.repeat(np.arange(len(draw)), [len(pos[p]) for p in draw])))
    return out


def boot(d, fn, seed=0):
    """Applies fn to every paper-clustered resample of d and returns one row per resample."""
    out = []
    for ix, cl in paper_draws(d.orig_id.values, B, seed):
        try:
            out.append(fn(d.iloc[ix].assign(orig_id=cl)))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            pass
    return pd.DataFrame(out)


def interval_rows(est, bs, **tag):
    """Point estimates with 95% percentile intervals from the resamples."""
    return [{**tag, "quantity": k, "estimate": v, "lo": np.percentile(bs[k].dropna(), 2.5),
             "hi": np.percentile(bs[k].dropna(), 97.5)} for k, v in est.items()]


def gaps(x, y="y"):
    """AUC gap between Opus 5.5 and each other model on each model's own passing claims and on shared claim sets,
    and the change from the former to the latter."""
    rec = x[[f"rec_{m}" for m in DEV_MODELS]].any(axis=1)
    prb = x[[f"probe_{m}" for m in DEV_MODELS]].any(axis=1)
    sets = {"all": x, "common": x[~rec], "strict": x[~rec & ~prb]}
    out = {}
    for o in DEV_MODELS[1:]:
        own = auc(x[~x.rec_opus], "q_opus", y) - auc(x[~x[f"rec_{o}"]], f"q_{o}", y)
        out[f"opus-{o} own"] = own
        for sn, s in sets.items():
            g = auc(s, "q_opus", y) - auc(s, f"q_{o}", y)
            out[f"opus-{o} {sn}"] = g
            out[f"opus-{o} {sn} minus own"] = g - own
    return out


def delta(x, m, y, ref):
    """Reference-adjusted AUC drop of model m under outcome y with reference column ref."""
    sp = x[~x[f"rec_{m}"]]
    return (auc(x, f"q_{m}", y) - auc(sp, f"q_{m}", y)) - (auc(x, ref, y) - auc(sp, ref, y))


def ece(y, p, w, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        k = idx == b
        if k.any():
            tot += w[k].sum() / w.sum() * abs(np.average(y[k], weights=w[k]) - np.average(p[k], weights=w[k]))
    return tot


def metrics(y, p, w):
    return {"AUC": roc_auc_score(y, p, sample_weight=w), "Brier": brier_score_loss(y, p, sample_weight=w),
            "Acc": np.average((p >= .5) == y, weights=w), "ECE": ece(y, p, w)}


class Scorer:
    """Scores columns of one frame on a shared set of paper-clustered resamples, so that differences are paired."""

    def __init__(self, k, ycol="y", seed=0):
        self.k = k.reset_index(drop=True)
        self.y = self.k[ycol].values.astype(int)
        rng = np.random.default_rng(seed)
        ug = self.k.orig_id.unique()
        pos = {g: np.where(self.k.orig_id.values == g)[0] for g in ug}
        self.idx = [np.concatenate([pos[g] for g in rng.choice(ug, len(ug))]) for _ in range(B)]
        self.idx = [ix for ix in self.idx if len(set(self.y[ix])) > 1]
        self.w = {"claim": np.ones(len(self.k)),
                  "paper": (1 / self.k.groupby("orig_id").orig_id.transform("size")).values}

    def _b(self, p, w, ix):
        return np.average((p[ix] - self.y[ix]) ** 2, weights=w[ix])

    def row(self, col, wn, ref=None, tag=None):
        """Brier score and AUC of column col under weighting wn, and its paired difference from column ref."""
        p, w = self.k[col].values, self.w[wn]
        m = metrics(self.y, p, w)
        const = np.allclose(p, p[0])
        bs = np.array([self._b(p, w, ix) for ix in self.idx])
        au = np.full(len(self.idx), .5) if const else np.array(
            [roc_auc_score(self.y[ix], p[ix], sample_weight=w[ix]) for ix in self.idx])
        q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        out = {**(tag or {}), "forecast": col, "weighting": wn, "n_claims": len(self.k),
               "n_papers": self.k.orig_id.nunique(), "Brier": m["Brier"], "Brier_lo": q(bs)[0], "Brier_hi": q(bs)[1],
               "AUC": .5 if const else m["AUC"], "AUC_lo": q(au)[0], "AUC_hi": q(au)[1], "Acc": m["Acc"],
               "ECE": m["ECE"], "dConst": m["Brier"] - .25, "dConst_lo": q(bs - .25)[0], "dConst_hi": q(bs - .25)[1]}
        if ref is not None:
            pr = self.k[ref].values
            d = bs - np.array([self._b(pr, w, ix) for ix in self.idx])
            ar = np.array([roc_auc_score(self.y[ix], pr[ix], sample_weight=w[ix]) for ix in self.idx])
            out.update({"ref": ref, "dBrier": m["Brier"] - metrics(self.y, pr, w)["Brier"], "dBrier_lo": q(d)[0],
                        "dBrier_hi": q(d)[1], "dAUC": out["AUC"] - metrics(self.y, pr, w)["AUC"],
                        "dAUC_lo": q(au - ar)[0], "dAUC_hi": q(au - ar)[1]})
        return out
