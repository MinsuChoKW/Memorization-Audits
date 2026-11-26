"""Supervised forecasts fitted on the development set and the recalibration of elicited forecasts.

Every model is fitted with equal paper weights (w_orig = 1 / claims of the paper)."""
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# predictors from the harmonized record of the original study
P_N = ["neglog_p_o", "log_n_o"]
ES_N = ["z_absr", "log_n_o"]
NUMF = ["neglog_p_o", "p_o_nonsig", "p_o_marginal", "log_n_o", "absr_o", "z_evid_o", "year_o", "nested_o",
        "one_tailed_o", "interaction", "claim_len"]
CATF = ["disc", "es_family_o", "p_o_ineq"]


def num_block():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())


def m_p_n():
    """Logistic regression on -log10 p and log10 n."""
    return make_pipeline(ColumnTransformer([("n", num_block(), P_N)]), LogisticRegression(C=1.0, max_iter=2000))


def m_es_n():
    """Logistic regression on the Fisher-transformed effect size and log10 n."""
    return make_pipeline(ColumnTransformer([("n", num_block(), ES_N)]), LogisticRegression(C=1.0, max_iter=2000))


def m_harmonized():
    """Logistic regression on the full harmonized record."""
    ct = ColumnTransformer([("n", num_block(), NUMF),
                            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=15), CATF)])
    return make_pipeline(ct, LogisticRegression(C=0.3, max_iter=3000))


def m_gbm():
    """Gradient-boosted trees on the harmonized record."""
    ct = ColumnTransformer([("n", "passthrough", NUMF),
                            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=15, sparse_output=False), CATF)])
    return make_pipeline(ct, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                            l2_regularization=1.0, random_state=0))


def m_text():
    """Logistic regression on TF-IDF features of the claim text."""
    ct = ColumnTransformer([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                                  stop_words="english", max_features=20000), "text")])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=3000))


def fit_weighted(mk, x, y, w):
    m = mk()
    m.fit(x, y, **{m.steps[-1][0] + "__sample_weight": w})
    return m


def baselines(tr, te, y):
    """Supervised baselines fitted on tr with target y and applied to te."""
    w = tr.w_orig.values
    out = {name: fit_weighted(mk, tr, tr[y], w).predict_proba(te)[:, 1]
           for name, mk in [("p_n", m_p_n), ("harmonized", m_harmonized), ("gbm", m_gbm), ("text", m_text)]}
    out["features_text"] = (out["gbm"] + out["text"]) / 2
    return out


# ---- recalibration of elicited forecasts
STACK_FEATS = ["neglog_p_o", "log_n_o"]


def logit(p):
    """Logit of a forecast truncated to [0.02, 0.98]."""
    p = np.clip(p, .02, .98)
    return np.log(p / (1 - p))


def recalibrator():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def fit_recalibration(d, cols):
    """Paper-weighted logistic recalibration on the significance criterion."""
    return recalibrator().fit(d[cols], d.y_stat, logisticregression__sample_weight=d.w_orig.values)


def platt_params(m):
    """Intercept and slope of the fitted recalibration on the logit scale."""
    lr, sc = m.named_steps["logisticregression"], m.named_steps["standardscaler"]
    beta = lr.coef_[0][0] / sc.scale_[0]
    return lr.intercept_[0] - lr.coef_[0][0] * sc.mean_[0] / sc.scale_[0], beta
