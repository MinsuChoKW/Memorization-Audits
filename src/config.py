"""Paths, model identifiers and constants shared by all steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
FORECASTS = ROOT / "forecasts"
OUTPUT = ROOT / "output"

# short name -> model identifier used in the forecast file names
MODELS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
          "sol": "gpt-6-2-sol", "astra": "gpt-6-2-astra"}
# models run on the development set
DEV_MODELS = ["opus", "sonnet", "astra", "sol"]
# a recalled outcome other than "none" flags a claim (self-report) or a paper (probe)
RECALLED = ["success", "failure", "mixed"]
# paper-clustered bootstrap resamples
B = 2000
