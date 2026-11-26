"""Step 3. Forecasting performance on the test set (262 claims from 161 source papers, two release rounds pooled).

Supervised baselines and recalibration mappings are fitted once on the development set with the significance
criterion as the target, after removing development claims whose source paper is in the test set, and are applied
to the test set without refitting. Every score is computed with equal claim weights and with equal paper weights.

Outputs in output/
  test_predictions.csv          per-claim forecasts on the test set (contains the test outcomes; do not share)
  test_performance.csv          baselines, elicited forecasts (evidence condition, J = 5) and recalibrated forecasts,
                                with paired differences from the regression on the p-value and sample size
  test_elicitation_choices.csv  information condition, number of samples, recalibration variants, full text
  test_model_pairs.csv          paired differences between every two elicited forecasts
  test_rounds.csv               scores by release round
  test_sample_consistency.csv   consistency of the five samples per claim in the evidence condition
  summary_info.csv              sample sizes, recalibration parameters, single- versus multi-claim papers"""
import numpy as np
import pandas as pd

from config import B, MODELS, OUTPUT, RECALLED
from forecasts import dev_frame, load_claims, load_test, mean_forecast, on_test, read_forecasts
from models import STACK_FEATS, baselines, fit_recalibration, logit, platt_params
from scoring import Scorer, paper_draws, wauc

SUPERVISED = ["const", "p_n", "harmonized", "features_text"]


def predictions():
    t = load_test()
    f = load_claims(fill_text=True)
    info = {"test_claims": len(t), "test_papers": t.orig_id.nunique(), "test_success_rate": t.y.mean(),
            "test_multi_claim_papers": int((t.groupby("orig_id").size() > 1).sum()),
            "test_claims_in_multi": int((t.groupby("orig_id").orig_id.transform("size") > 1).sum()),
            "dev_claims": len(f), "dev_papers": f.orig_id.nunique(),
            "dev_y_reported": int(f.y_reported.notna().sum()), "dev_y_stat": int(f.y_stat.notna().sum()),
            "test_claims_paper_in_dev": int(t.doi_o.isin(set(f.doi_o.dropna())).sum())}

    # supervised baselines fitted on the development set without the source papers of the test set
    fd = f[f.y_stat.notna() & ~f.doi_o.isin(t.doi_o.dropna())]
    bp = baselines(fd, t, "y_stat")
    t["const"] = 0.5
    for c in SUPERVISED[1:]:
        t[c] = bp[c]

    # elicited forecasts, and their recalibration fitted on development claims that pass the self-report
    dev = {}
    for nm, mid in MODELS.items():
        t["q_" + nm] = on_test(mean_forecast, "stats", mid).set_index("claim_uid")["mean"].reindex(t.claim_uid).values
        t["q1_" + nm] = on_test(mean_forecast, "stats", mid, True).set_index("claim_uid")["mean"].reindex(t.claim_uid).values
        for cond in ["blind", "identified"]:
            h = on_test(mean_forecast, cond, mid, True)
            t[f"{cond}_{nm}"] = h.set_index("claim_uid")["mean"].reindex(t.claim_uid).values
        k = dev_frame(mid)
        if k is None:
            continue
        dev[nm] = k
        cal = k[k.y_stat.notna() & ~k.doi_o.isin(t.doi_o.dropna())]
        passing = cal[~cal.recalled]
        t["llm"] = logit(t["q_" + nm])
        m_pass, m_all = fit_recalibration(passing, ["llm"]), fit_recalibration(cal, ["llm"])
        m_stack = fit_recalibration(passing, STACK_FEATS + ["llm"])
        t["recal_" + nm] = m_pass.predict_proba(t[["llm"]])[:, 1]
        t["recal_all_" + nm] = m_all.predict_proba(t[["llm"]])[:, 1]
        t["stack_" + nm] = m_stack.predict_proba(t[STACK_FEATS + ["llm"]])[:, 1]
        a, b = platt_params(m_pass)
        a2, b2 = platt_params(m_all)
        info.update({f"{nm}_recal_claims": len(passing), f"{nm}_recal_papers": passing.orig_id.nunique(),
                     f"{nm}_alpha": a, f"{nm}_beta": b, f"{nm}_alpha_all_claims": a2, f"{nm}_beta_all_claims": b2})
    t = t.drop(columns="llm")
    return t, dev, info


def performance(t, info):
    S = Scorer(t)
    cols = SUPERVISED + [c for nm in MODELS for c in (f"q_{nm}", f"recal_{nm}") if c in t]
    main = [S.row(c, wn, ref="p_n") for wn in ("claim", "paper") for c in cols]

    choices = []
    for nm in MODELS:
        for wn in ("claim", "paper"):
            for name, col, ref in [("information: blind", f"blind_{nm}", f"q1_{nm}"),
                                   ("information: evidence", f"q1_{nm}", None),
                                   ("information: identified", f"identified_{nm}", f"q1_{nm}"),
                                   ("samples: J=1", f"q1_{nm}", None), ("samples: J=5", f"q_{nm}", f"q1_{nm}"),
                                   ("recalibration: none", f"q_{nm}", None),
                                   ("recalibration: passing claims", f"recal_{nm}", f"q_{nm}"),
                                   ("recalibration: all claims", f"recal_all_{nm}", f"q_{nm}"),
                                   ("recalibration: stacked", f"stack_{nm}", f"q_{nm}")]:
                if col in t:
                    choices.append(S.row(col, wn, ref=ref, tag={"model": nm, "choice": name}))
    # full-text condition (Sonnet 5.5, single sample) on the claims with an available source paper
    ft = on_test(read_forecasts, "fulltext", MODELS["sonnet"])[["claim_uid", "probability", "recalled_replication_outcome"]]
    tf = t.merge(ft.rename(columns={"probability": "fulltext_sonnet"}), on="claim_uid").dropna(subset=["fulltext_sonnet"])
    SF = Scorer(tf)
    for wn in ("claim", "paper"):
        choices.append(SF.row("q1_sonnet", wn, tag={"model": "sonnet", "choice": "full-text subset: evidence"}))
        choices.append(SF.row("fulltext_sonnet", wn, ref="q1_sonnet",
                              tag={"model": "sonnet", "choice": "full-text subset: full text"}))
    info.update({"fulltext_claims": len(tf), "fulltext_papers": tf.orig_id.nunique(),
                 "fulltext_recall_rate": tf.recalled_replication_outcome.isin(RECALLED).mean(),
                 "fulltext_corr_evidence": np.corrcoef(tf.fulltext_sonnet, tf.q1_sonnet)[0, 1]})

    npap = t.groupby("orig_id").orig_id.transform("size")
    for nm in MODELS:
        if f"recal_{nm}" in t:
            for lab, sub in [("single", t[npap == 1]), ("multi", t[npap > 1])]:
                info[f"{nm}_recal_brier_{lab}_claim_papers"] = np.mean((sub["recal_" + nm] - sub.y) ** 2)
    return pd.DataFrame(main), pd.DataFrame(choices)


def model_pairs(t):
    """Paired differences in AUC and Brier score between every two elicited forecasts (evidence, J = 5)."""
    models = ["opus", "sonnet", "astra", "sol", "haiku"]
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    draws = paper_draws(t.orig_id.values, B, 0)

    def stats(x, wn):
        w = np.ones(len(x)) if wn == "claim" else (1 / x.groupby("orig_id").orig_id.transform("size")).values
        out = {}
        for a, b in pairs:
            out[f"dAUC {a}-{b}"] = wauc(x.y, x[f"q_{a}"], w) - wauc(x.y, x[f"q_{b}"], w)
            out[f"dBrier {a}-{b}"] = (np.average((x[f"q_{a}"] - x.y) ** 2, weights=w)
                                      - np.average((x[f"q_{b}"] - x.y) ** 2, weights=w))
        return out

    rows = []
    for wn in ("claim", "paper"):
        est = stats(t, wn)
        bs = pd.DataFrame([stats(t.iloc[ix].assign(orig_id=cl), wn) for ix, cl in draws])
        for k, v in est.items():
            rows.append({"weighting": wn, "quantity": k, "estimate": v, "lo": np.percentile(bs[k], 2.5),
                         "hi": np.percentile(bs[k], 97.5)})
    return pd.DataFrame(rows)


def by_round(t):
    t = t.copy()
    t["round"] = t.claim_uid.str[:2]
    cols = SUPERVISED + [f"q_{m}" for m in MODELS] + [f"recal_{m}" for m in MODELS if f"recal_{m}" in t]
    out = []
    for rd, s in t.groupby("round"):
        s = s.reset_index(drop=True)
        papers = s.orig_id.unique()
        draws = paper_draws(s.orig_id.values, B, 0)
        for wn in ("claim", "paper"):
            def w_of(x):
                return np.ones(len(x)) if wn == "claim" else (1 / x.groupby("orig_id").orig_id.transform("size")).values
            for c in cols:
                w = w_of(s)
                a = .5 if c == "const" else wauc(s.y, s[c], w)
                bs = [] if c == "const" else [wauc(s.y.values[ix], s[c].values[ix], w_of(s.iloc[ix].assign(orig_id=cl)))
                                              for ix, cl in draws]
                out.append({"round": rd, "weighting": wn, "forecast": c, "claims": len(s), "papers": len(papers),
                            "success_rate": s.y.mean(), "Brier": np.average((s[c] - s.y) ** 2, weights=w), "AUC": a,
                            "AUC_lo": np.percentile(bs, 2.5) if bs else .5, "AUC_hi": np.percentile(bs, 97.5) if bs else .5})
    return pd.DataFrame(out)


def icc_absolute(X):
    """Two-way random-effects intraclass correlations for absolute agreement (Shrout & Fleiss, 1979),
    for a single sample, ICC(2,1), and for the mean of the k samples, ICC(2,k). X: claims x samples."""
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / (msr + (msc - mse) / n)


def sample_consistency():
    """Agreement between the J = 5 stated probabilities of each model on the test claims (evidence condition)."""
    rows = []
    for nm, mid in MODELS.items():
        r = on_test(read_forecasts, "stats", mid)
        r["probability"] = pd.to_numeric(r.probability, errors="coerce")
        X = r.pivot_table(index="claim_uid", columns="sample", values="probability").dropna().values
        cors = [np.corrcoef(X[:, a], X[:, b])[0, 1] for a in range(X.shape[1]) for b in range(a + 1, X.shape[1])]
        icc1, icck = icc_absolute(X)
        rows.append({"model": nm, "claims": X.shape[0], "samples": X.shape[1], "mean_pairwise_r": np.mean(cors),
                     "min_pairwise_r": np.min(cors), "max_pairwise_r": np.max(cors), "ICC_2_1": icc1, "ICC_2_k": icck})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    t, dev, info = predictions()
    keep = ["claim_uid", "orig_id", "y"] + [c for c in t.columns if c in SUPERVISED or c.split("_")[0] in
                                            ("q", "q1", "blind", "identified", "recal", "stack")]
    t[keep].to_csv(OUTPUT / "test_predictions.csv", index=False)
    main_rows, choices = performance(t, info)
    main_rows.to_csv(OUTPUT / "test_performance.csv", index=False)
    choices.to_csv(OUTPUT / "test_elicitation_choices.csv", index=False)
    pd.Series(info).to_csv(OUTPUT / "summary_info.csv", header=["value"])

    t = pd.read_csv(OUTPUT / "test_predictions.csv")
    model_pairs(t).to_csv(OUTPUT / "test_model_pairs.csv", index=False)
    by_round(t).to_csv(OUTPUT / "test_rounds.csv", index=False)
    sample_consistency().to_csv(OUTPUT / "test_sample_consistency.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["forecast", "weighting", "Brier", "Brier_lo", "Brier_hi", "AUC", "AUC_lo", "AUC_hi", "dBrier", "dBrier_lo",
            "dBrier_hi"]
    print(main_rows[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
