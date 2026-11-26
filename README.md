# LLM-Replication-Audit

Code, prompts and stored model responses for eliciting probabilistic forecasts of replication outcomes from large language models, auditing the forecasts for recalled outcomes, and scoring them. The forecasts come from five models (Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 4.5, GPT-6.2 Astra and GPT-6.2 Sol) on a development set of published replications and on a test set of replications whose outcomes were released separately in two rounds.

## Repository layout

| Path | Content |
|---|---|
| `src/01_prepare_data.py` … `src/07_stored_responses.py` | Analysis steps, run in numerical order |
| `src/elicit_forecasts.py` | Prompts and elicitation for all information conditions and the recognition probe |
| `src/config.py`, `src/forecasts.py`, `src/models.py`, `src/scoring.py`, `src/success_criteria.py` | Shared paths, loaders, supervised models, scoring and the paper-clustered bootstrap |
| `forecasts/` | All 22,195 stored model responses |
| `data/` | Place for the input files (see `data/README.md`); none are distributed |
| `output/` | Results written by the analysis steps |

## Setup

Python 3.12 or later (tested with 3.13).

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the analyses

1. Place the input files as described in `data/README.md`.
2. Run the steps in order from the repository root:

```sh
python src/01_prepare_data.py
python src/02_reference_forecasts.py
python src/03_test_set.py
python src/04_recall_signals.py
python src/05_development_audit.py
python src/06_outcome_labels.py
python src/07_stored_responses.py
```

The steps read the stored responses in `forecasts/`, so no API access is needed. All folds and bootstrap resamples use fixed seeds, and a rerun gives identical results. The full run takes about 30 minutes on two CPU cores, most of it in the bootstrap of steps 3, 5 and 6.

| Step | What it computes | Outputs in `output/` |
|---|---|---|
| `01_prepare_data.py` | Harmonized records of the original studies, outcome labels and paper groups for both data sets | `data_report.txt` (prepared files in `data/processed/`) |
| `02_reference_forecasts.py` | Reference forecasts under cross-validation grouped by source paper (and by connected component), for the author-reported outcome and the significance criterion | `reference_forecasts_y_reported.csv`, `reference_forecasts_y_stat.csv` |
| `03_test_set.py` | Test-set scores of the supervised baselines, elicited and recalibrated forecasts with equal claim and equal paper weights, elicitation choices, paired model comparisons, scores by release round | `test_performance.csv`, `test_elicitation_choices.csv`, `test_model_pairs.csv`, `test_rounds.csv`, `test_predictions.csv`, `summary_info.csv` |
| `04_recall_signals.py` | Self-report and probe flag rates, agreement of definite recall reports with the observed outcome and the majority-outcome baseline, overlap of flagged claims | `recall_rates.csv`, `recall_counts.csv`, `recall_agreement.csv`, `recall_overlap.csv` |
| `05_development_audit.py` | AUC drop after each model's own exclusions against the reference forecasts, hidden-recall contrast, scores on claim sets shared by all models, change in the gap between models | `audit_strata.csv`, `audit_reference_strata.csv`, `audit_hidden_recall.csv`, `audit_references.csv`, `audit_common_sets.csv`, `audit_gap_change.csv` |
| `06_outcome_labels.py` | The analyses of step 5 rescored under each outcome on the claims that carry both, and their differences | `labels_crosstab.csv`, `labels_estimates.csv`, `labels_contrast.csv` |
| `07_stored_responses.py` | Stored and valid responses by set, condition and model | `stored_responses.csv` |

Every result file holds point estimates with 95% percentile intervals from 2,000 paper-clustered bootstrap resamples, where applicable. Differences between forecasts and between claim sets are computed on the same resamples. Some files in `output/` and `data/processed/` contain test-set outcomes, which are covered by the access agreement, so they are excluded from version control.

## Stored responses

File names follow `forecasts/<set>_<condition>_<model>.jsonl`. The set is `dev` (development) or `r1`/`r2` (test set, release rounds 1 and 2). The condition is `blind`, `stats` (the evidence condition), `identified`, `fulltext` or `probe`. Each line is one response with the fields

| Field | Meaning |
|---|---|
| `key` | Hash of claim, condition, model and sample, used to skip stored calls |
| `claim_uid`, `orig_id` | Claim and source-paper identifiers, as assigned by step 1 |
| `model`, `cond`, `sample` | Model identifier, information condition, sample index |
| `probability` | Stated probability of a successful replication |
| `recognized_study` | Whether the model reports recognizing the study |
| `recalled_replication_outcome` | `success`, `failure`, `mixed` or `none` |
| `confidence` | Probe only |
| `key_factors`, `raw` | Development set only |
| `usage` | Token counts, where the API returned them |

Test-set responses keep only the structured fields, because the free-text rationale and the raw response paraphrase the test claims.

## Eliciting new forecasts

API keys are read from a file given with `--key-file` or from `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, and are never written to the responses. New responses are appended to `forecasts/`, and calls already stored for the same claim, condition, model and sample are skipped.

```sh
python src/elicit_forecasts.py run --cond stats --set dev --model claude-opus-5-5 --key-file <key file>
python src/elicit_forecasts.py run --cond stats --set r1 --model claude-opus-5-5 --k 5 --key-file <key file>
python src/elicit_forecasts.py run --cond probe --set dev --model claude-opus-5-5 --key-file <key file>
OPENAI_REASONING_EFFORT=low python src/elicit_forecasts.py run --cond stats --set r1 --model gpt-6.2-sol --k 5 --key-file <key file>
```

The stored GPT responses were elicited with `OPENAI_REASONING_EFFORT=low`, the lowest reasoning effort these models accept; the Claude models were run without an additional reasoning budget. `--workers` sends calls in parallel. `--dry-run` prints the first prompt without calling an API. Models may be resolved by the vendors to updated snapshots, so new responses can differ from the stored ones.

## License

The code is released under the MIT License (`LICENSE`). The stored responses in `forecasts/` are released under CC BY 4.0.

