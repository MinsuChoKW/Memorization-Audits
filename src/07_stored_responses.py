"""Step 7. Stored responses by set, condition and model.

Output output/stored_responses.csv with the number of stored responses, the number that yield a probability (or, for
the probe, a recalled outcome), the samples per unit and the number of units."""
import pandas as pd

from config import FORECASTS, MODELS, OUTPUT


def main():
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    for f in sorted(FORECASTS.glob("*.jsonl")):
        st, cond, mid = f.stem.split("_", 2)
        r = pd.read_json(f, lines=True)
        nm = next((k for k, v in MODELS.items() if v == mid), mid)
        valid = (r.recalled_replication_outcome.notna() if cond == "probe"
                 else pd.to_numeric(r.probability, errors="coerce").notna())
        unit = "orig_id" if cond == "probe" else "claim_uid"
        rows.append({"set": st, "condition": cond, "model": nm, "stored": len(r), "valid": int(valid.sum()),
                     "samples_per_unit": int(r.groupby(unit).size().max()), "units": r[unit].nunique()})
    r = pd.DataFrame(rows)
    r.to_csv(OUTPUT / "stored_responses.csv", index=False)
    print(r.to_string(index=False))
    print("total", r.stored.sum())


if __name__ == "__main__":
    main()
