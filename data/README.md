# Input data

The input files are not distributed with this repository. Place them under `data/raw/` with the names below. Step 1 (`src/01_prepare_data.py`) writes the prepared files to `data/processed/`.

| File | Content | Source | Access |
|---|---|---|---|
| `raw/FORRT_Training_Data.csv` | Development claims, i.e., original and replication effect pairs from the FORRT Replication Database | Training data released by Stankov et al. (2025), https://osf.io/9uxsv | Public |
| `raw/FReD_2026-11.xlsx` | FORRT Replication Database, used to complete bibliographic metadata (publication years, journals) | https://github.com/forrtproject/FReD-data (commit `0265ac9`) | Public |
| `raw/rounds/test-set_for-participants_1.csv`, `raw/rounds/test-set_for-participants_2.csv` | Test-set claims, release rounds 1 and 2 | Center for Open Science | Access agreement |
| `raw/rounds/ground_truth_1.csv`, `raw/rounds/ground_truth_2.csv` | Test-set outcomes, release rounds 1 and 2 | Center for Open Science | Access agreement |

The full-text condition additionally needs the source papers of the test claims, passed to `src/elicit_forecasts.py` with `--pdf-dir` or `--text-json`. They are not distributed either.

## Checking the prepared data

Step 1 writes row counts and coverage to `output/data_report.txt`. The analyses use

- 1,607 development claims from 566 source papers (rows of the database; the 47 rows without a database identifier are excluded), of which 1,287 carry the outcome reported by the replication authors and 583 the significance criterion;
- 262 test claims from 161 source papers (132 in round 1 and 130 in round 2).
