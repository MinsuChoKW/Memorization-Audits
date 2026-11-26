"""Replication success criteria, ported from the FReD R package (forrtproject/FReD, R/replication_outcomes.R
and R/helpers.R, commit cc4c2b0). All criteria work on correlations r and sample sizes n.
Each function returns a float array: 1 success, 0 failure, NaN not assessable
("OS not significant" in FReD is returned as NaN, since the criterion is undefined there).
"""
import numpy as np
import pandas as pd
from scipy import stats

CRITERIA = ["significance_r", "significance_agg", "consistency_ci", "consistency_pi", "homogeneity",
            "homogeneity_significance", "small_telescopes"]


def _ok(r, n, nmin=3):
    return ~(np.isnan(r) | np.isnan(n) | (n <= nmin) | (np.abs(r) >= 1))


def p_from_r(r, n):
    p = np.full(len(r), np.nan)
    v = _ok(r, n, 2)
    t = r[v] * np.sqrt((n[v] - 2) / (1 - r[v] ** 2))
    p[v] = 2 * stats.t.sf(np.abs(t), n[v] - 2)
    return p


def coded_significance(p, op, tails, alpha=.05):
    """FReD is_p_significant(): TRUE/FALSE only when the coded inequality settles it, else NaN."""
    p = np.where((p < 0) | (p > 1), np.nan, p.astype(float))
    p = np.where(tails == 1, np.minimum(2 * p, 1), p)
    out = np.full(len(p), np.nan)
    out = np.where((op == "eq") & ~np.isnan(p), (p < alpha).astype(float), out)
    out = np.where((op == "lt") & (p <= alpha), 1.0, out)
    out = np.where((op == "gt") & (p >= alpha), 0.0, out)
    return out


def resolve_significance(r, n, coded=None, alpha=.05):
    calc = p_from_r(r, n)
    calc = np.where(np.isnan(calc), np.nan, (calc < alpha).astype(float))
    return calc if coded is None else np.where(np.isnan(coded), calc, coded)


def fe_meta(r1, n1, r2, n2):
    """Fixed-effect meta-analysis of two raw correlations (metafor escalc measure='COR', rma method='FE')."""
    v = _ok(r1, n1, 2) & _ok(r2, n2, 2)
    est, lb, ub, qp = (np.full(len(r1), np.nan) for _ in range(4))
    w1, w2 = (n1[v] - 1) / (1 - r1[v] ** 2) ** 2, (n2[v] - 1) / (1 - r2[v] ** 2) ** 2
    e = (w1 * r1[v] + w2 * r2[v]) / (w1 + w2)
    se = np.sqrt(1 / (w1 + w2))
    q = w1 * (r1[v] - e) ** 2 + w2 * (r2[v] - e) ** 2
    est[v], lb[v], ub[v], qp[v] = e, e - 1.959964 * se, e + 1.959964 * se, stats.chi2.sf(q, 1)
    return est, lb, ub, qp


def power_r(r, n, alpha=.05):
    """One-sided power to detect r with sample size n (FReD compute_power_r)."""
    out = np.full(len(r), np.nan)
    v = _ok(r, n)
    rv, nv = np.abs(r[v]), n[v]
    tt = stats.t.isf(alpha, nv - 2)
    rc = np.sqrt(tt ** 2 / (tt ** 2 + nv - 2))
    out[v] = stats.norm.cdf((np.arctanh(rv) + rv / (2 * (nv - 1)) - np.arctanh(rc)) * np.sqrt(nv - 3))
    return out


def r_for_power(power, n, alpha=.05):
    out = np.full(len(n), np.nan)
    v = ~np.isnan(n) & (n > 3)
    nv = n[v]
    tt = stats.t.isf(alpha, nv - 2)
    zreq = stats.norm.ppf(power) / np.sqrt(nv - 3) + np.arctanh(np.sqrt(tt ** 2 / (tt ** 2 + nv - 2)))
    r0 = np.tanh(zreq)
    out[v] = np.tanh(zreq - (r0 - r0 / (2 * (nv - 1))) / (2 * (nv - 1)))
    return out


def assess(r_o, n_o, r_r, n_r, sig_o_coded=None, sig_r_coded=None):
    r_o, n_o, r_r, n_r = (np.asarray(x, float) for x in (r_o, n_o, r_r, n_r))
    nan = np.full(len(r_o), np.nan)
    os_sig = resolve_significance(r_o, n_o, sig_o_coded)
    rep_sig = resolve_significance(r_r, n_r, sig_r_coded)
    has_dir = ~np.isnan(r_o) & ~np.isnan(r_r) & (r_o != 0) & (r_r != 0)
    same = np.sign(r_o) == np.sign(r_r)
    out = {}

    s = nan.copy()
    s = np.where((os_sig == 1) & (rep_sig == 0), 0.0, s)
    s = np.where((os_sig == 1) & (rep_sig == 1) & has_dir, same.astype(float), s)
    out["significance_r"] = s

    est, lb, ub, qp = fe_meta(r_o, n_o, r_r, n_r)
    odir = ~np.isnan(r_o) & (r_o != 0)
    same_dir = odir & np.where(r_o < 0, ub < 0, lb > 0)
    opp_dir = odir & np.where(r_o < 0, lb > 0, ub < 0)
    not_sig = (lb <= 0) & (ub >= 0)
    agg = np.where(same_dir, 1.0, np.where(opp_dir | not_sig, 0.0, np.nan))
    out["significance_agg"] = agg

    v = _ok(r_r, n_r)
    z, se = np.arctanh(np.where(v, r_r, 0)), 1 / np.sqrt(np.where(v, n_r, 4) - 3)
    lo, hi = np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)
    out["consistency_ci"] = np.where(v & ~np.isnan(r_o), ((r_o > lo) & (r_o < hi)).astype(float), np.nan)

    v = _ok(r_o, n_o) & ~np.isnan(n_r) & (n_r > 3)
    z = np.arctanh(np.where(v, r_o, 0))
    se = np.sqrt(1 / (np.where(v, n_o, 4) - 3) + 1 / (np.where(v, n_r, 4) - 3))
    lo, hi = np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)
    out["consistency_pi"] = np.where(v & ~np.isnan(r_r), ((r_r > lo) & (r_r < hi)).astype(float), np.nan)

    out["homogeneity"] = np.where(np.isnan(qp), np.nan, (qp >= .05).astype(float))
    hs = np.where(np.isnan(qp) | np.isnan(agg), np.nan, ((qp >= .05) & same_dir).astype(float))
    out["homogeneity_significance"] = np.where(os_sig == 0, np.nan, hs)

    thr = r_for_power(.33, n_o)
    al = np.where(r_o < 0, -r_r, r_r)
    out["small_telescopes"] = np.where(np.isnan(thr) | np.isnan(al) | np.isnan(r_o), np.nan, (al >= thr).astype(float))
    return pd.DataFrame(out)


if __name__ == "__main__":  # example from the FReD documentation plus hand checks
    a = lambda *x: np.array(x, float)
    print(assess(a(.3, .3, .3, -.3), a(100, 100, 30, 100), a(.25, .02, -.4, -.25), a(120, 500, 200, 120)).T)
    print("power", power_r(a(.3), a(120)), "r33", r_for_power(.33, a(100)))
