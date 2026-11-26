"""Step 5. Memorization audits on the development set (claims with an author-reported outcome).

Within-model analysis: the AUC drop of each model after excluding the claims it flags, compared with the AUC drop
of a reference forecast on the same claims (reference-adjusted AUC drop), under five references. Hidden-recall
contrast: the model-minus-reference AUC difference on passing claims from probe-flagged papers minus that on the
other passing claims. Between-model analysis: every model scored on the claims that pass every self-report (common
set) and on those that also pass every probe (strict set), and the change in the gap between two models from their
own audits to these sets. AUCs use equal paper weights recomputed within every scored set, and intervals come from
a paper-clustered bootstrap in which every drawn copy of a paper counts as a separate paper. The reference
forecasts are held fixed across resamples.

Outputs in output/
  audit_strata.csv             AUC and Brier score of each model on all, flagged and passing claims, and on passing
                               claims by replication year
  audit_reference_strata.csv   AUC of the model and of two references by self-report and probe status
  audit_hidden_recall.csv      reference-adjusted AUC drop and hidden-recall contrast (primary reference)
  audit_references.csv         reference-adjusted AUC drop under all five references
  audit_common_sets.csv        AUC of every model and pairwise differences on all, common, strict and own sets
  audit_gap_change.csv         change in the gap between Opus 5.5 and each other model from own audits to common sets"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DEV_MODELS, MODELS, OUTPUT
from forecasts import PRIMARY_REFERENCE, REFERENCES, audit_frame, dev_frame, probe_flagged_papers
from scoring import Scorer, auc, boot, gaps, interval_rows, paper_draws


def strata():
    rows = []
    for nm, mid in MODELS.items():
        k = dev_frame(mid)
        if k is None:
            continue
        L = k[k.y_reported.notna()].assign(y=lambda x: x.y_reported.astype(int))
        sets = {"all": L, "flagged": L[L.recalled], "passing": L[~L.recalled],
                "passing, replication <= 2022": L[~L.recalled & (L.year_r <= 2022)],
                "passing, replication >= 2024": L[~L.recalled & (L.year_r >= 2024)]}
        for sname, s in sets.items():
            rows.append(Scorer(s).row("mean", "paper", tag={"model": nm, "stratum": sname}))
    return pd.DataFrame(rows)


def reference_strata():
    ref = pd.read_csv(OUTPUT / "reference_forecasts_y_reported.csv")[["claim_uid", "paper_p_n", PRIMARY_REFERENCE]]
    rows = []
    for nm, mid in MODELS.items():
        k = dev_frame(mid)
        if k is None:
            continue
        k["probe"] = k.orig_id.isin(probe_flagged_papers("dev", mid))
        L = k[k.y_reported.notna()].assign(y=lambda x: x.y_reported.astype(int)).merge(ref, on="claim_uid", how="left")
        sets = {"all": L, "self-report flagged": L[L.recalled], "self-report passing": L[~L.recalled],
                "self-report passing, probe flagged": L[~L.recalled & L.probe],
                "self-report passing, probe passing": L[~L.recalled & ~L.probe], "probe flagged": L[L.probe]}
        for sname, d in sets.items():
            sc = Scorer(d)
            r = sc.row("mean", "paper", tag={"model": nm, "stratum": sname})
            for col, short in [("paper_p_n", "ref_p_n"), (PRIMARY_REFERENCE, "ref")]:
                c = sc.row(col, "paper")
                r.update({f"{short}_AUC": c["AUC"], f"{short}_AUC_lo": c["AUC_lo"], f"{short}_AUC_hi": c["AUC_hi"]})
            rows.append({**r, "share_of_labelled": len(d) / len(L)})
    return pd.DataFrame(rows)


def hidden_recall():
    """Reference-adjusted AUC drop (delta) and hidden-recall contrast (h) with the primary reference."""
    ref = pd.read_csv(OUTPUT / "reference_forecasts_y_reported.csv")[["claim_uid", PRIMARY_REFERENCE]].rename(
        columns={PRIMARY_REFERENCE: "ref"})

    def a(d, col):
        return roc_auc_score(d.y, d[col], sample_weight=1 / d.groupby("orig_id").orig_id.transform("size"))

    def stats(d):
        sp, pf, pp = d[~d.rec], d[~d.rec & d.probe], d[~d.rec & ~d.probe]
        return {"drop_model": a(d, "mean") - a(sp, "mean"), "drop_reference": a(d, "ref") - a(sp, "ref"),
                "reference_adjusted_drop": (a(d, "mean") - a(sp, "mean")) - (a(d, "ref") - a(sp, "ref")),
                "excess_probe_flagged": a(pf, "mean") - a(pf, "ref"),
                "excess_probe_passing": a(pp, "mean") - a(pp, "ref"),
                "hidden_recall_contrast": (a(pf, "mean") - a(pf, "ref")) - (a(pp, "mean") - a(pp, "ref"))}

    out = []
    for nm, mid in MODELS.items():
        if nm not in DEV_MODELS:
            continue
        k = dev_frame(mid)
        k["probe"] = k.orig_id.isin(probe_flagged_papers("dev", mid))
        d = k[k.y_reported.notna()].merge(ref, on="claim_uid")
        d = d.assign(y=d.y_reported.astype(int), rec=d.recalled).reset_index(drop=True)
        est = stats(d)
        bs = []
        for ix, cl in paper_draws(d.orig_id.values):
            try:
                bs.append(stats(d.iloc[ix].assign(orig_id=cl)))
            except ValueError:
                pass
        bs = pd.DataFrame(bs)
        for key, v in est.items():
            out.append({"model": nm, "quantity": key, "estimate": v, "lo": np.percentile(bs[key], 2.5),
                        "hi": np.percentile(bs[key], 97.5)})
    return pd.DataFrame(out)


def references(d):
    rows = []
    for m in DEV_MODELS:
        def fn(x):
            sp = x[~x[f"rec_{m}"]]
            drop = auc(x, f"q_{m}") - auc(sp, f"q_{m}")
            out = {"drop_model": drop}
            for rn, col in REFERENCES.items():
                out[f"drop_reference: {rn}"] = auc(x, col) - auc(sp, col)
                out[f"reference_adjusted_drop: {rn}"] = drop - out[f"drop_reference: {rn}"]
            return out

        rows += interval_rows(fn(d), boot(d, fn), model=m)
    return pd.DataFrame(rows)


def common_sets(d):
    any_rec = d[[f"rec_{m}" for m in DEV_MODELS]].any(axis=1)
    any_probe = d[[f"probe_{m}" for m in DEV_MODELS]].any(axis=1)
    sets = {"all labelled claims": d, "passing all self-reports": d[~any_rec],
            "passing all self-reports and probes": d[~any_rec & ~any_probe], "own self-report passing": d}
    pairs = [(a, b) for i, a in enumerate(DEV_MODELS) for b in DEV_MODELS[i + 1:]]
    rows = []
    for sname, s in sets.items():
        s = s.reset_index(drop=True)
        own = sname.startswith("own")

        def fn(x):
            a = {m: auc(x[~x[f"rec_{m}"]] if own else x, f"q_{m}") for m in DEV_MODELS}
            a["reference"] = auc(x, PRIMARY_REFERENCE)
            return {**{f"AUC_{m}": a[m] for m in a}, **{f"{p}_minus_{q}": a[p] - a[q] for p, q in pairs}}

        rows += interval_rows(fn(s), boot(s, fn), set=sname, n_claims=len(s), n_papers=s.orig_id.nunique())
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    strata().to_csv(OUTPUT / "audit_strata.csv", index=False)
    reference_strata().to_csv(OUTPUT / "audit_reference_strata.csv", index=False)
    h = hidden_recall()
    h.to_csv(OUTPUT / "audit_hidden_recall.csv", index=False)
    d = audit_frame()
    c = common_sets(d)
    c.to_csv(OUTPUT / "audit_common_sets.csv", index=False)
    r = references(d)
    r.to_csv(OUTPUT / "audit_references.csv", index=False)
    g = pd.DataFrame(interval_rows(gaps(d), boot(d, gaps)))
    g.to_csv(OUTPUT / "audit_gap_change.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(h.round(3).to_string(index=False))
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
