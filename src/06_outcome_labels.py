"""Step 6. Sensitivity to the scoring outcome on the development claims that carry both outcome labels.

The forecasts, self-report flags, probe flags and claim sets are held fixed, and only the scoring outcome changes
between the author-reported outcome (y) and the significance criterion (y_stat). The reference forecast is
cross-validated separately for each outcome. All quantities, including the differences between the two outcomes,
are computed on the same paper-clustered resamples.

Outputs in output/
  labels_crosstab.csv   cross-tabulation of the two outcomes
  labels_estimates.csv  AUC, reference-adjusted AUC drop, recall agreement and gaps under each outcome
  labels_contrast.csv   author-reported minus significance, for the gap changes and the reference-adjusted drops"""
import pandas as pd

from config import DEV_MODELS, OUTPUT, PROCESSED
from forecasts import PRIMARY_REFERENCE, add_recalled_outcomes, audit_frame
from scoring import auc, boot, delta, gaps, interval_rows


def agreement(x, m, y):
    dd = x[x[f"o_{m}"].isin(["success", "failure"])]
    return ((dd[f"o_{m}"] == "success") == (dd[y] == 1)).mean()


def both_labels(d):
    cl = pd.read_csv(PROCESSED / "claims.csv")[["claim_uid", "y_stat"]]
    rs = pd.read_csv(OUTPUT / "reference_forecasts_y_stat.csv")[["claim_uid", PRIMARY_REFERENCE]].rename(
        columns={PRIMARY_REFERENCE: "ref_stat"})
    b = d.merge(cl, on="claim_uid").merge(rs, on="claim_uid")
    b = b[b.y_stat.notna()].reset_index(drop=True)
    b["y_stat"] = b.y_stat.astype(int)
    return b


def estimates(x):
    out = {}
    for yl, ref in [("y", PRIMARY_REFERENCE), ("y_stat", "ref_stat")]:
        nm = "reported" if yl == "y" else "significance"
        for m in DEV_MODELS:
            out[f"{nm} AUC {m} all"] = auc(x, f"q_{m}", yl)
            out[f"{nm} AUC {m} own"] = auc(x[~x[f"rec_{m}"]], f"q_{m}", yl)
            out[f"{nm} delta {m}"] = delta(x, m, yl, ref)
            out[f"{nm} agreement {m}"] = agreement(x, m, yl)
        for k, v in gaps(x, yl).items():
            out[f"{nm} {k}"] = v
    return out


def contrast(x):
    gr, gs = gaps(x, "y"), gaps(x, "y_stat")
    out = {}
    for o in DEV_MODELS[1:]:
        for sn in ("common", "strict"):
            k = f"opus-{o} {sn} minus own"
            out[f"reported {k}"] = gr[k]
            out[f"significance {k}"] = gs[k]
            out[f"reported minus significance {k}"] = gr[k] - gs[k]
    for m in DEV_MODELS:
        out[f"reported minus significance delta {m}"] = (delta(x, m, "y", PRIMARY_REFERENCE)
                                                         - delta(x, m, "y_stat", "ref_stat"))
    return out


def main():
    OUTPUT.mkdir(exist_ok=True)
    b = both_labels(add_recalled_outcomes(audit_frame()))
    pd.crosstab(b.y, b.y_stat).rename_axis(index="y_reported", columns="y_stat").to_csv(OUTPUT / "labels_crosstab.csv")
    tag = {"n_claims": len(b), "n_papers": b.orig_id.nunique()}
    e = pd.DataFrame(interval_rows(estimates(b), boot(b, estimates), **tag))
    e.to_csv(OUTPUT / "labels_estimates.csv", index=False)
    c = pd.DataFrame(interval_rows(contrast(b), boot(b, contrast), **tag))
    c.to_csv(OUTPUT / "labels_contrast.csv", index=False)
    pd.set_option("display.width", 200)
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
