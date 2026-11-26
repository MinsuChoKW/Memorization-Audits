"""Loading of the prepared claims and of the stored forecasts.

Forecast files are named forecasts/<set>_<condition>_<model>.jsonl, where the set is dev, r1 or r2 (the test set is
released in two rounds) and the condition is blind, stats (the evidence condition), identified, fulltext or probe."""
import pandas as pd

from config import DEV_MODELS, FORECASTS, MODELS, OUTPUT, PROCESSED, RECALLED
from models import logit


def read_forecasts(st, cond, model):
    f = FORECASTS / f"{st}_{cond}_{model}.jsonl"
    return pd.read_json(f, lines=True) if f.exists() else None


def first_responses(st, cond, model):
    """The first of the J responses per claim (all responses of a probe file)."""
    r = read_forecasts(st, cond, model)
    if r is None:
        return None
    return r[r["sample"] == 0] if "sample" in r else r


def mean_forecast(st, cond, model, first_only=False):
    """Mean stated probability per claim over the J samples (or the first sample only)."""
    p = read_forecasts(st, cond, model)
    if p is None:
        return None
    p["probability"] = pd.to_numeric(p.probability, errors="coerce")
    if first_only:
        p = p[p["sample"] == 0]
    return p.groupby("claim_uid").probability.agg(["mean", "std", "size"]).reset_index()


def recall_reports(st, cond, model):
    """Recalled outcome (rec) and recognition (recog) of the first response per claim."""
    r = first_responses(st, cond, model)
    if r is None:
        return None
    return r[["claim_uid", "recalled_replication_outcome", "recognized_study"]].rename(
        columns={"recalled_replication_outcome": "rec", "recognized_study": "recog"})


def on_test(fn, *args):
    fr = [fn(s, *args) for s in ("r1", "r2")]
    fr = [x for x in fr if x is not None]
    return pd.concat(fr, ignore_index=True) if fr else None


def probe_flagged_papers(st, model):
    """Source papers for which the recognition probe reports a replication outcome (first query per paper)."""
    p = read_forecasts(st, "probe", model).drop_duplicates("orig_id")
    return set(p.loc[p.recalled_replication_outcome.isin(RECALLED), "orig_id"])


def load_claims(fill_text=False):
    """Development claims of the database (rows without a database identifier are excluded)."""
    d = pd.read_csv(PROCESSED / "claims.csv")
    if fill_text:
        d["text"] = d.text.fillna("")
    return d[d.source == "forrt"]


def load_test():
    t = pd.concat([pd.read_csv(PROCESSED / f"round{i}.csv") for i in (1, 2)], ignore_index=True)
    t["text"] = t.text.fillna("")
    t["y"] = t.y_stat.astype(int)
    t["w_paper"] = 1 / t.groupby("orig_id").orig_id.transform("size")
    return t


def dev_frame(model):
    """Development claims with the evidence-condition forecast (mean) and the self-report flag (recalled)."""
    f = load_claims().reset_index(drop=True)
    g = mean_forecast("dev", "stats", model)
    r = recall_reports("dev", "stats", model)
    if g is None:
        return None
    k = f.merge(g, on="claim_uid").merge(r, on="claim_uid")
    k["recalled"] = k.rec.isin(RECALLED)
    k["llm"] = logit(k["mean"])
    return k


REFERENCES = {"effect size and N, paper-grouped": "paper_es_n", "p-value and N, paper-grouped": "paper_p_n",
              "boosting on features, paper-grouped": "paper_gbm",
              "features and text, paper-grouped": "paper_features_text",
              "features and text, component-grouped": "component_features_text"}
PRIMARY_REFERENCE = "paper_features_text"


def audit_frame():
    """Labelled development claims (author-reported outcome) with, for every development model, the forecast
    (q_<model>), the self-report flag (rec_<model>) and the probe flag of the source paper (probe_<model>), and the
    cross-validated reference forecasts."""
    ref = pd.read_csv(OUTPUT / "reference_forecasts_y_reported.csv")
    d = None
    for nm in DEV_MODELS:
        mid = MODELS[nm]
        k = dev_frame(mid)[["claim_uid", "orig_id", "y_reported", "mean", "recalled"]]
        k["probe"] = k.orig_id.isin(probe_flagged_papers("dev", mid))
        k = k.rename(columns={"mean": f"q_{nm}", "recalled": f"rec_{nm}", "probe": f"probe_{nm}"})
        d = k if d is None else d.merge(k.drop(columns=["orig_id", "y_reported"]), on="claim_uid")
    d = d[d.y_reported.notna()].merge(ref[["claim_uid"] + list(REFERENCES.values())], on="claim_uid")
    d["y"] = d.y_reported.astype(int)
    return d.reset_index(drop=True)


def add_recalled_outcomes(d):
    """Adds the recalled outcome of the first evidence-condition response of every development model (o_<model>)."""
    for m in DEV_MODELS:
        r = first_responses("dev", "stats", MODELS[m])
        d[f"o_{m}"] = d.claim_uid.map(r.set_index("claim_uid").recalled_replication_outcome)
    return d
