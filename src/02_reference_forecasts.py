"""Step 2. Reference forecasts without access to the outcomes of the claims they forecast.

For each scoring outcome of the development set (author-reported outcome y_reported, significance criterion y_stat),
supervised models are fitted under five-fold cross-validation grouped by source paper and, for the primary reference,
also grouped by connected components of source papers and replication reports. Folds are assigned to whole groups
without looking at the outcome, and out-of-fold probabilities are averaged over five fold assignments.

Outputs output/reference_forecasts_y_reported.csv and output/reference_forecasts_y_stat.csv, one row per labelled
claim with the columns
  paper_es_n               logistic regression on effect size and N, paper-grouped
  paper_p_n                logistic regression on -log10 p and log10 n, paper-grouped
  paper_gbm                gradient-boosted model on the harmonized record, paper-grouped
  paper_features_text      mean of paper_gbm and a TF-IDF text model, paper-grouped (primary reference)
  component_features_text  the primary reference under component-grouped cross-validation"""
import numpy as np
import pandas as pd

from config import OUTPUT
from forecasts import load_claims
from models import fit_weighted, m_es_n, m_gbm, m_p_n, m_text

SEEDS, K = 5, 5
FITTED = {"paper": {"es_n": m_es_n, "p_n": m_p_n, "gbm": m_gbm, "text": m_text},
          "component": {"gbm": m_gbm, "text": m_text}}


def folds(groups, seed):
    """Assigns whole groups to K folds without looking at the outcome (largest groups first, to the smallest fold)."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).value_counts()
    order = g.sample(frac=1, random_state=int(rng.integers(1e9))).sort_values(ascending=False, kind="stable")
    size, assign = np.zeros(K), {}
    for name, n in order.items():
        k = int(np.argmin(size))
        assign[name] = k
        size[k] += n
    return pd.Series(groups).map(assign).values


def out_of_fold(k, label, gcol, models):
    """Seed-averaged out-of-fold probabilities."""
    y = k[label].values.astype(int)
    out = {m: np.zeros(len(k)) for m in list(models) + ["features_text"]}
    for s in range(SEEDS):
        f = folds(k[gcol].values, s)
        for j in range(K):
            tr, te = np.where(f != j)[0], np.where(f == j)[0]
            if len(te) == 0 or len(np.unique(y[tr])) < 2:
                continue
            T, E = k.iloc[tr], k.iloc[te]
            pr = {name: fit_weighted(mk, T, y[tr], T.w.values).predict_proba(E)[:, 1] for name, mk in models.items()}
            pr["features_text"] = (pr["gbm"] + pr["text"]) / 2
            for name, v in pr.items():
                out[name][te] += v / SEEDS
    return out


def main():
    OUTPUT.mkdir(exist_ok=True)
    f = load_claims(fill_text=True).reset_index(drop=True)
    f["z_absr"] = np.arctanh(f.absr_o.clip(upper=.999))
    for label in ["y_reported", "y_stat"]:
        k = f[f[label].notna()].reset_index(drop=True).copy()
        k["w"] = 1 / k.groupby("orig_id").orig_id.transform("size")
        o = k[["claim_uid", "orig_id", "group", label]].copy()
        for gname, gcol in [("paper", "orig_id"), ("component", "group")]:
            p = out_of_fold(k, label, gcol, FITTED[gname])
            for m in (["es_n", "p_n", "gbm", "features_text"] if gname == "paper" else ["features_text"]):
                o[f"{gname}_{m}"] = p[m]
        o.to_csv(OUTPUT / f"reference_forecasts_{label}.csv", index=False)
        print(label, len(o), "claims")


if __name__ == "__main__":
    main()
