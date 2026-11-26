"""Step 1. Prepares the development and test sets from the raw files in data/raw (see data/README.md).

Outputs in data/processed
  claims.csv              one row per original-replication effect pair of the development set
  round1.csv, round2.csv  claims of the test set with their released outcomes, by release round
and output/data_report.txt with row counts and coverage. Only information about the original study is turned into
predictors; replication-side columns are kept only as outcomes and analysis variables (prefix lab_ / rep_)."""
import re
import sys

import numpy as np
import pandas as pd

from config import OUTPUT, PROCESSED, RAW
from success_criteria import CRITERIA, assess, coded_significance, power_r

OUT = PROCESSED

DISC = {
    "social psychology": "social_psych", "cognitive psychology": "cognitive_psych",
    "psychology": "psych_other", "differential psychology": "psych_other",
    "developmental psychology": "psych_other", "evolutionary psychology": "psych_other",
    "political psychology": "psych_other", "organizational psychology": "psych_other",
    "consumer psychology": "marketing", "judgment and decision making": "jdm_econ",
    "marketing": "marketing", "consumer research": "marketing", "marketing/org behavior": "marketing",
    "management": "management", "economics": "jdm_econ", "experimental philosophy": "xphi",
    "education": "education", "linguistics": "linguistics",
    "health": "health", "medicine": "health", "health psychology": "health",
    "clinical psychology": "health", "psychopathology": "health", "sports science": "health",
    "political science": "polisci_soc", "sociology": "polisci_soc",
}
SUCCESS = {"successful": 1, "yes": 1, "failed": 0, "no": 0}
NUM = r"(-?\d*\.?\d+(?:e-?\d+)?)"


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.extract(NUM, flags=re.I)[0], errors="coerce")


def norm_doi(s):
    s = s.astype(str).str.lower().str.strip().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    return s.where(~s.isin(["nan", "na", ""]))


def effect_to_r(es_type, es_value, n):
    """Return (|r|-equivalent magnitude, family). Magnitude only: sign conventions differ across reports."""
    t = str(es_type).lower().strip()
    v = str(es_value).replace("−", "-").replace(",", "") if pd.notna(es_value) else ""
    x = pd.to_numeric(re.search(NUM, v).group(1), errors="coerce") if re.search(NUM, v) else np.nan
    n = n if pd.notna(n) and n > 3 else np.nan
    if t.startswith("test statistic"):
        m = re.search(r"t\s*\(\s*" + NUM + r"\s*\)\s*=\s*" + NUM, v, flags=re.I)
        if m:
            df, tv = float(m.group(1)), float(m.group(2))
            return np.sqrt(tv**2 / (tv**2 + df)), "t"
        m = re.search(r"f\s*\(\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\)\s*=\s*" + NUM, v, flags=re.I)
        if m:
            d1, d2, fv = map(float, m.groups())
            return np.sqrt(d1 * fv / (d1 * fv + d2)), "F1" if d1 == 1 else "Fk"
        m = re.search(r"(?:chi|χ|x)\s*\^?2?\s*\(\s*" + NUM + r"[^)]*\)\s*=\s*" + NUM, v, flags=re.I)
        if m and pd.notna(n):
            return min(np.sqrt(float(m.group(2)) / n), 1.0), "chi2"
        m = re.search(r"z\s*=\s*" + NUM, v, flags=re.I)
        if m and pd.notna(n):
            return min(abs(float(m.group(1))) / np.sqrt(n), 1.0), "z"
        m = re.search(r"r\s*\(\s*" + NUM + r"\s*\)\s*=\s*" + NUM, v, flags=re.I)
        if m:
            return abs(float(m.group(2))), "r"
        return np.nan, "other"
    if pd.isna(x):
        return np.nan, "other"
    t = {"odds_ratio": "or", "eta_squared": "etasq", "partial_eta_squared": "etasq (partial)"}.get(t, t)
    if t in {"d", "cohen_d", "hedges' g", "cohen's d", "g"}:
        return abs(x) / np.sqrt(x**2 + 4), "d"
    if t in {"r", "pearson_r", "phi", "spearmans_rank_order_correlation", "partial r"}:
        return (abs(x), "r") if abs(x) <= 1 else (np.nan, "other")
    if t in {"etasq", "etasq (partial)", "omega squared", "r-square", "r-squared", "r²", "partial r-square"}:
        return (np.sqrt(x), "eta2_r2") if 0 <= x <= 1 else (np.nan, "other")
    if t in {"or", "odds_ratio"}:
        if x <= 0:
            return np.nan, "other"
        d = np.log(x) * np.sqrt(3) / np.pi
        return abs(d) / np.sqrt(d**2 + 4), "OR"
    if t == "cohen_f_squared":
        return (np.sqrt(x / (1 + x)), "f") if x >= 0 else (np.nan, "other")
    if t == "cohen's f":
        return abs(x) / np.sqrt(1 + x**2), "f"
    if t in {"beta (std)", "beta"}:
        return (abs(x), "beta_std") if abs(x) <= 1 else (np.nan, "other")
    return np.nan, "other"


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build(df, meta=None):
    d = pd.DataFrame(index=df.index)
    d["doi_o"], d["doi_r"] = norm_doi(df.doi_o), norm_doi(df.doi_r)
    d["rep_key"] = d.doi_r.fillna(df.url_r.astype(str).str.lower()).where(lambda s: s != "nan")
    d["rep_key"] = d.rep_key.fillna(df.ref_r.astype(str).str[:80])
    for c in ["entry_id", "effect_id", "fred_id", "ref_o", "ref_r", "study_o", "tags", "description",
              "claim_text_o", "file_o", "file_r"]:
        d[c] = df[c] if c in df else np.nan
    d["year_o"] = df.ref_o.astype(str).str.extract(r"\((\d{4})")[0].astype(float)
    d["year_r"] = df.ref_r.astype(str).str.extract(r"\((\d{4})")[0].astype(float)
    if meta is not None:  # fill from FReD release (Crossref years, journal, author overlap)
        mo = meta.dropna(subset=["doi_o"]).drop_duplicates("doi_o").set_index("doi_o")
        mr = meta.dropna(subset=["doi_r"]).drop_duplicates("doi_r").set_index("doi_r")
        d["year_o"] = d.year_o.fillna(d.doi_o.map(mo.year_o))
        d["year_r"] = d.year_r.fillna(d.doi_r.map(mr.year_r))
        d["journal_o"] = d.doi_o.map(mo.journal_o)
        d["title_o"] = d.doi_o.map(mo.title_o)
    d["discipline_raw"] = df.discipline
    d["disc"] = df.discipline.astype(str).str.lower().str.strip().map(DISC).fillna("other")

    # ---- original-study predictors
    d["n_o"] = to_num(df.n_o)
    d["log_n_o"] = np.log10(d.n_o.clip(lower=2))
    d["p_o"] = to_num(df.pval_value_o).where(lambda s: (s >= 0) & (s <= 1))
    pt = df.pval_type_o.astype(str).str.lower().str.extract(
        r"(less|greater|exact|threshold|<=|>=|<|>|=|\ble\b|\bge\b|\bl\b|\bg\b|\be\b)")[0]
    d["p_o_ineq"] = pt.map({"<": "lt", "l": "lt", "<=": "lt", "le": "lt", ">": "gt", "g": "gt", ">=": "gt",
                            "ge": "gt", "=": "eq", "e": "eq", "less": "lt", "threshold": "lt", "greater": "gt",
                            "exact": "eq"}).fillna("na")
    d["p_o_nonsig"] = ((d.p_o > .05) | ((d.p_o_ineq == "gt") & (d.p_o >= .05))).astype(float).where(d.p_o.notna())
    d["p_o_marginal"] = ((d.p_o > .01) & (d.p_o <= .05) & (d.p_o_ineq != "gt")).astype(float).where(d.p_o.notna())
    d["neglog_p_o"] = -np.log10(d.p_o.clip(lower=1e-6))
    er = [effect_to_r(t, v, n) for t, v, n in zip(df.es_type_o, df.es_value_o, d.n_o)]
    d["absr_o"] = [min(a, .999) if pd.notna(a) else np.nan for a, _ in er]
    if "r_o" in df:  # fall back to the r supplied with the data
        d["absr_o"] = d.absr_o.fillna(pd.to_numeric(df.r_o, errors="coerce").abs().clip(upper=.999))
    d["es_family_o"] = [b for _, b in er]
    d["z_evid_o"] = np.arctanh(d.absr_o) * np.sqrt((d.n_o - 3).clip(lower=1))
    d["nested_o"] = df.nested_o.astype(str).str.lower().str.startswith("y").astype(float)
    d["one_tailed_o"] = df.pval_tails_o.astype(str).str.lower().str.match(r"1|one").astype(float)
    d["prereg_o"] = df.prereg_o.notna().astype(float) if "prereg_o" in df else 0.0
    # Only the verbatim excerpt from the ORIGINAL paper is used as text. The curator-written `description` field
    # can mention the replication (its design, and in a few rows its outcome), so it must not reach any model.
    txt = df.claim_text_o.fillna("").astype(str)
    d["text"] = txt.str.replace(r"\s+", " ", regex=True).str.strip()
    d["interaction"] = txt.str.contains(r"interact|moderat|×|\bx\b condition|two-way|three-way", case=False).astype(float)
    d["claim_len"] = np.log10(df.claim_text_o.fillna("").astype(str).str.len() + 1)

    # ---- replication side: labels and analysis variables only
    d["rep_n"] = to_num(df.n_r)
    d["rep_r_fred"] = pd.to_numeric(df.r_r, errors="coerce") if "r_r" in df else np.nan
    d["orig_r_fred"] = pd.to_numeric(df.r_o, errors="coerce") if "r_o" in df else np.nan
    if "reported_success" in df:
        rs = df.reported_success.astype(str).str.lower().str.strip()
        d["lab_reported_raw"] = rs
        d["lab_reported3"] = rs.map({**{k: ("success" if v else "failure") for k, v in SUCCESS.items()},
                                     "mixed": "mixed"}).fillna("other")
        d["y_reported"] = rs.map(SUCCESS)
    # ---- FReD success criteria on converted correlations (labels only)
    ineq = {"<": "lt", "l": "lt", "<=": "lt", "le": "lt", ">": "gt", "g": "gt", ">=": "gt", "ge": "gt",
            "=": "eq", "e": "eq"}
    rp = to_num(df.pval_value_r)
    rop = df.pval_type_r.astype(str).str.extract(r"(<=|>=|<|>|=|\ble\b|\bge\b|\bl\b|\bg\b|\be\b)")[0].map(ineq)
    sig_o = coded_significance(to_num(df.pval_value_o).values, d.p_o_ineq.values, d.one_tailed_o.values)
    sig_r = coded_significance(rp.values, rop.fillna("na").values,
                               df.pval_tails_r.astype(str).str.startswith("1").astype(float).values)
    crit = assess(d.orig_r_fred.values, d.n_o.values, d.rep_r_fred.values, d.rep_n.values, sig_o, sig_r)
    pair = d.orig_r_fred.notna() & d.rep_r_fred.notna()  # label only where both effects are directional and
    for c in CRITERIA:                                    # convertible, so failures are not over-represented
        d["y_" + c] = crit[c].where(pair.values).values
    d["rep_power"] = power_r(d.orig_r_fred.values, d.rep_n.values)
    ss = df.statistical_success
    d["y_stat"] = ss.map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0, "True": 1.0, "False": 0.0})
    return d


def add_groups(d, prefix):
    uf = UF()
    for o, r in zip(d.doi_o.fillna("o?" + d.ref_o.astype(str).str[:60]), d.rep_key):
        uf.union("o:" + str(o), "r:" + str(r))
    key = d.doi_o.fillna("o?" + d.ref_o.astype(str).str[:60])
    comp = key.map(lambda o: uf.find("o:" + str(o)))
    d["group"] = prefix + pd.factorize(comp)[0].astype(str)
    d["orig_id"] = prefix + "o" + pd.factorize(key)[0].astype(str)
    d["w_orig"] = 1 / d.groupby("orig_id").orig_id.transform("size")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(RAW / "FORRT_Training_Data.csv")
    F = pd.read_excel(RAW / "FReD_2026-11.xlsx")
    F["doi_o"], F["doi_r"] = norm_doi(F.doi_o), norm_doi(F.doi_r)
    d = build(A, F)
    # rows without a database identifier are marked and excluded from all analyses
    d.insert(0, "source", np.where(A.entry_id.isna(), "osf_appended", "forrt"))
    d = add_groups(d, "g")
    d.insert(0, "claim_uid", ["c%04d" % i for i in range(len(d))])
    d.to_csv(OUT / "claims.csv", index=False)

    rounds = {}
    for i in (1, 2):  # test set, released in two rounds with their outcomes
        T = pd.read_csv(RAW / f"rounds/test-set_for-participants_{i}.csv")
        T = T.merge(pd.read_csv(RAW / f"rounds/ground_truth_{i}.csv"), on="claim_id")
        T["statistical_success"] = T.outcome
        for c in ["doi_r", "url_r", "ref_r", "file_r", "n_r", "r_r", "pval_value_r", "pval_type_r", "pval_tails_r"]:
            T[c] = np.nan
        T["ref_r"] = "test set, release round %d" % i
        s = build(T, F)
        s.insert(0, "claim_id", T.claim_id.values)
        s.insert(0, "source", "round%d" % i)
        s["in_dev"] = s.doi_o.isin(set(d.doi_o.dropna()))
        s["rep_key"] = "round"  # replication reports are not identified in the test sets
        key = s.doi_o.fillna(s.ref_o.astype(str).str[:60])
        s["orig_id"] = "p" + pd.util.hash_pandas_object(key, index=False).astype(str).str[:10]
        s["group"] = s.orig_id
        s["w_orig"] = 1 / s.groupby("orig_id").orig_id.transform("size")
        s.insert(0, "claim_uid", ["r%d_%03d" % (i, k) for k in range(len(s))])
        rounds[i] = s
    rounds[2]["in_round1"] = rounds[2].orig_id.isin(set(rounds[1].orig_id))
    for i, s in rounds.items():
        s.to_csv(OUT / f"round{i}.csv", index=False)
    r1, r2 = rounds[1], rounds[2]

    rep = [
        f"development rows={len(d)}, original studies={d.orig_id.nunique()}, leakage groups={d.group.nunique()} "
        f"(largest group rows={d.group.value_counts().iloc[0]})",
        f"y_reported: n={int(d.y_reported.notna().sum())}, success rate={d.y_reported.mean():.3f}; "
        f"3-class={d.lab_reported3.value_counts().to_dict()}",
        f"y_stat: n={int(d.y_stat.notna().sum())}, success rate={d.y_stat.mean():.3f}",
        f"|r|_o coverage={d.absr_o.notna().mean():.3f} (FReD r_o coverage={d.orig_r_fred.notna().mean():.3f}); "
        f"families={d.es_family_o.value_counts().to_dict()}",
        f"p_o coverage={d.p_o.notna().mean():.3f}; original nonsig share={d.p_o_nonsig.mean():.3f}",
        f"year_r>=2024: rows={int((d.year_r >= 2024).sum())}, originals={d[d.year_r >= 2024].orig_id.nunique()}",
        "criteria (n, success rate): " + ", ".join(
            f"{c}={int(d['y_' + c].notna().sum())}/{d['y_' + c].mean():.2f}" for c in CRITERIA),
        f"rep_power: n={int(d.rep_power.notna().sum())}, median={d.rep_power.median():.3f}, "
        f"share>=0.8={(d.rep_power.dropna() >= .8).mean():.3f}",
        f"osf_appended rows={int((d.source == "osf_appended").sum())}, y_stat rate={d[d.source == "osf_appended"].y_stat.mean():.3f}",
        f"round1 claims={len(r1)}, papers={r1.orig_id.nunique()}, success={r1.y_stat.mean():.3f}, "
        f"|r|_o coverage={r1.absr_o.notna().mean():.3f}, papers also in dev={int(r1.in_dev.sum())} claims",
        f"round2 claims={len(r2)}, papers={r2.orig_id.nunique()}, success={r2.y_stat.mean():.3f}, "
        f"claims whose paper is in round1={int(r2.in_round1.sum())}, in dev={int(r2.in_dev.sum())}",
    ]
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "data_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    sys.exit(main())
