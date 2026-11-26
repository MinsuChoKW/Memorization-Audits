"""Step 4. Recall signals: the claim-level self-report and the paper-level recognition probe.

The self-report flag of a claim is taken from the first response in the evidence condition (stats) and is set when
the recalled outcome is success, failure or mixed. The probe flags a source paper in the same way. Agreement is the
share of definite reports (success or failure) that match the observed outcome, compared with the agreement of
always reporting the more common outcome among the same claims (majority-outcome baseline).

Outputs in output/
  recall_rates.csv      recall and recognition rates by set, model and condition, with agreement
  recall_counts.csv     flagged units, definite reports and agreement with exact binomial intervals
  recall_agreement.csv  agreement on the development set with paper-clustered bootstrap intervals and the
                        majority-outcome baseline
  recall_overlap.csv    overlap of the claims flagged by the self-reports of two models"""
import numpy as np
import pandas as pd
from scipy.stats import beta

from config import B, DEV_MODELS, MODELS, OUTPUT, RECALLED
from forecasts import (add_recalled_outcomes, audit_frame, dev_frame, first_responses, load_claims, load_test,
                       on_test, read_forecasts, recall_reports)
from scoring import paper_draws


def hits(d, y="y", rec="rec"):
    return ((d[rec] == "success") & (d[y] == 1)) | ((d[rec] == "failure") & (d[y] == 0))


def rates():
    t = load_test()
    rows = []
    for nm, mid in MODELS.items():
        k = dev_frame(mid)
        if k is None:
            continue
        L = k[k.y_reported.notna()].assign(y=lambda x: x.y_reported.astype(int))
        defi = L[L.rec.isin(["success", "failure"])]
        rows.append({"set": "development", "model": nm, "condition": "evidence", "n": len(k),
                     "recognized": (k.recog == True).mean(), "recall_rate": k.recalled.mean(),
                     "n_definite_labelled": len(defi), "recall_agreement": hits(defi).mean()})
        p = read_forecasts("dev", "probe", mid)
        if p is not None:
            rows.append({"set": "development", "model": nm, "condition": "probe (per paper)", "n": len(p),
                         "recognized": (p.recognized_study == True).mean(),
                         "recall_rate": p.recalled_replication_outcome.isin(RECALLED).mean()})
        r2 = recall_reports("dev", "identified", mid)
        if r2 is not None:
            k2 = L[["claim_uid", "y"]].merge(r2, on="claim_uid")
            defi = k2[k2.rec.isin(["success", "failure"])]
            rows.append({"set": "development", "model": nm, "condition": "identified", "n": len(r2),
                         "recognized": (r2.recog == True).mean(), "recall_rate": r2.rec.isin(RECALLED).mean(),
                         "n_definite_labelled": len(defi), "recall_agreement": hits(defi).mean()})
    for nm, mid in MODELS.items():
        for cond in ["stats", "identified"]:
            k2 = t[["claim_uid", "y"]].merge(on_test(recall_reports, cond, mid), on="claim_uid")
            defi = k2[k2.rec.isin(["success", "failure"])]
            rows.append({"set": "test", "model": nm, "condition": "evidence" if cond == "stats" else cond,
                         "n": len(k2), "recognized": (k2.recog == True).mean(),
                         "recall_rate": k2.rec.isin(RECALLED).mean(), "n_definite_labelled": len(defi),
                         "recall_agreement": hits(defi).mean() if len(defi) else np.nan})
        p = on_test(read_forecasts, "probe", mid).drop_duplicates("orig_id")
        rows.append({"set": "test", "model": nm, "condition": "probe (per paper)", "n": len(p),
                     "recognized": (p.recognized_study == True).mean(),
                     "recall_rate": p.recalled_replication_outcome.isin(RECALLED).mean()})
    return pd.DataFrame(rows)


def clopper_pearson(k, n):
    return (beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def counts():
    dev = load_claims()
    test = load_test()
    rows = []
    for nm, mid in MODELS.items():
        for st, lab, base in [("dev", "y_reported", dev), ("test", "y", test)]:
            for cond in ["stats", "identified"]:
                if st == "dev":
                    r = first_responses("dev", cond, mid)
                else:
                    r = [first_responses(s, cond, mid) for s in ("r1", "r2")]
                    r = None if any(x is None for x in r) else pd.concat(r)
                if r is None:
                    continue
                k = base[["claim_uid", lab]].merge(r[["claim_uid", "recalled_replication_outcome"]], on="claim_uid")
                k["flag"] = k.recalled_replication_outcome.isin(RECALLED)
                d = k[k.recalled_replication_outcome.isin(["success", "failure"]) & k[lab].notna()]
                hit = hits(d, lab, "recalled_replication_outcome")
                lo, hi = clopper_pearson(int(hit.sum()), len(d)) if len(d) else (np.nan, np.nan)
                rows.append({"set": st, "model": nm,
                             "signal": "self-report, " + ("evidence" if cond == "stats" else cond),
                             "units": len(k), "flagged": int(k.flag.sum()), "definite_labelled": len(d),
                             "agreeing": int(hit.sum()), "agreement": hit.mean() if len(d) else np.nan,
                             "agreement_lo": lo, "agreement_hi": hi,
                             "success_rate_in_subset": d[lab].mean() if len(d) else np.nan,
                             "share_success_reports": (d.recalled_replication_outcome == "success").mean()
                             if len(d) else np.nan})
            if st == "dev":
                p = first_responses("dev", "probe", mid)
            else:
                p = [first_responses(s, "probe", mid) for s in ("r1", "r2")]
                p = None if any(x is None for x in p) else pd.concat(p)
            if p is not None:
                p = p.drop_duplicates("orig_id")
                rows.append({"set": st, "model": nm, "signal": "probe", "units": len(p),
                             "flagged": int(p.recalled_replication_outcome.isin(RECALLED).sum())})
    return pd.DataFrame(rows)


def agreement_bootstrap():
    d = add_recalled_outcomes(audit_frame())
    rows = []
    for m in DEV_MODELS:
        dd = d[d[f"o_{m}"].isin(["success", "failure"])].reset_index(drop=True)
        hit = ((dd[f"o_{m}"] == "success") == (dd.y == 1)).astype(float)
        dd = dd.assign(hit=hit)
        bsv = [dd.hit.values[ix].mean() for ix, _ in paper_draws(dd.orig_id.values, B, 0)]
        rows.append({"model": m, "n": len(dd), "papers": dd.orig_id.nunique(), "agreement": hit.mean(),
                     "lo": np.percentile(bsv, 2.5), "hi": np.percentile(bsv, 97.5),
                     "majority_baseline": max(dd.y.mean(), 1 - dd.y.mean())})
    return pd.DataFrame(rows)


def overlap():
    flags = {nm: set(dev_frame(MODELS[nm]).query("recalled").claim_uid) for nm in DEV_MODELS}
    rows = []
    for i, a in enumerate(DEV_MODELS):
        for b in DEV_MODELS[i + 1:]:
            A, Bf = flags[a], flags[b]
            rows.append({"a": a, "b": b, "n_a": len(A), "n_b": len(Bf), "both": len(A & Bf),
                         "jaccard": len(A & Bf) / max(1, len(A | Bf)), "share_of_b_in_a": len(A & Bf) / max(1, len(Bf))})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    r = rates()
    r.to_csv(OUTPUT / "recall_rates.csv", index=False)
    counts().to_csv(OUTPUT / "recall_counts.csv", index=False)
    a = agreement_bootstrap()
    a.to_csv(OUTPUT / "recall_agreement.csv", index=False)
    overlap().to_csv(OUTPUT / "recall_overlap.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
