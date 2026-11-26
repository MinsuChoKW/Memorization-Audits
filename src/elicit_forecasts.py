"""Elicits forecasts from a language model and appends the responses to forecasts/<set>_<condition>_<model>.jsonl.

  python src/elicit_forecasts.py run --cond <condition> --set dev|r1|r2 --model <model-id> [--k 5] [--key-file <file>]
  python src/elicit_forecasts.py run ... --dry-run          prints the first prompt without calling an API
  python src/elicit_forecasts.py models --key-file <file>   lists the OpenAI models the key can use

Conditions (information given to the model)
  blind       discipline and verbatim claim text
  stats       blind + harmonized record of the original study (sample size, p-value, effect size, design flags);
              the evidence condition
  identified  stats + bibliographic reference of the source paper
  fulltext    stats + text of the source paper with its reference list removed (--pdf-dir or --text-json)
  probe       reference of the source paper only, one query per paper; asks whether the model recalls a replication
              attempt and its outcome
Models whose identifier starts with `claude` are queried through the Anthropic API (key from --key-file or
ANTHROPIC_API_KEY), all others through the OpenAI API (key from --key-file or OPENAI_API_KEY). Calls already stored
for the same claim, condition, model and sample are skipped, so an interrupted run can be resumed."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SYSTEM = (
    "You are an expert methodologist forecasting replication outcomes in the social and behavioral sciences. "
    "A replication is successful when an independent study with new data finds a statistically significant "
    "effect (alpha = .05) in the same direction as the original claim. Base your forecast on the evidence "
    "given: strength of the statistical evidence, sample size, effect size plausibility, design complexity, "
    "and prior plausibility of the claim. Historical base rate of success is roughly 50%. "
    "Answer with a single JSON object and nothing else."
)
SCHEMA = ('{"probability": <number between 0 and 1 that the replication succeeds>, '
          '"key_factors": [<up to 4 short phrases>], '
          '"recognized_study": <true|false, whether you recognize this specific study>, '
          '"recalled_replication_outcome": <"success"|"failure"|"mixed"|"none">}')
PROBE = ("Below is the reference of a published study. Do you recall any replication attempt of it? "
         'Answer with JSON only: {"recognized_study": <true|false>, '
         '"recalled_replication_outcome": <"success"|"failure"|"mixed"|"none">, '
         '"confidence": <number between 0 and 1>}')


def fmt(v, nd=3):
    return "not reported" if pd.isna(v) else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


TEXTS = {}


def strip_refs(t, max_chars):
    parts = re.split(r"\n\s*(?:References|REFERENCES|Bibliography|Literature Cited|LITERATURE CITED)\s*\n", t)
    if len(parts) > 1 and len(parts[0]) > 0.4 * len(t):
        t = parts[0]
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars] if len(t) > 2000 else None  # scanned PDFs without a text layer are skipped


def pdf_text(pdf_dir, name, max_chars, doi=None):
    """Paper text from a JSON of extracted texts (--text-json) or from a PDF in pdf_dir named as the file_o field
    or after the DOI (with "/" replaced by "--" or "_")."""
    stems = [str(name)] if isinstance(name, str) else []
    if isinstance(doi, str):
        stems += [doi.replace("/", "--"), doi.replace("/", "_")]
    cands = []
    for st in stems:
        st = re.sub(r'[<>:;"|?*]', "_", st)
        cands += [st, st + ".pdf", re.sub(r"\.pdf$", "", st, flags=re.I) + ".pdf"]
    for c in cands:
        if c in TEXTS:
            return strip_refs(TEXTS[c], max_chars)
    if not pdf_dir:
        return None
    p = next((Path(pdf_dir) / c for c in cands if (Path(pdf_dir) / c).is_file()), None)
    if p is None:
        return None
    return strip_refs(subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout,
                      max_chars)


def build_prompt(r, cond, pdf_dir=None, max_chars=60000):
    if cond == "probe":
        return PROBE + "\n\nReference: " + str(r.ref_o)
    parts = [f"Field: {r.disc}", f"Claim (verbatim from the original report): {r.claim_text_o}"]
    # The database `description` field is deliberately NOT shown: it is written with knowledge of the replication.
    if cond in ("stats", "identified", "fulltext"):
        ineq = {"lt": "<", "gt": ">", "eq": "="}.get(r.p_o_ineq, "=")
        parts += [f"Original sample size: {fmt(r.n_o)}",
                  f"Original p-value: {'not reported' if pd.isna(r.p_o) else ineq + ' ' + fmt(r.p_o)}",
                  f"Original effect size (|r|-equivalent): {fmt(r.absr_o)}",
                  f"Within-subject or nested design: {'yes' if r.nested_o == 1 else 'no'}",
                  f"Interaction or moderation effect: {'yes' if r.interaction == 1 else 'no'}"]
    if cond == "identified":
        parts.append(f"Original study reference: {r.ref_o}")
    if cond == "fulltext":
        t = pdf_text(pdf_dir, r.file_o, max_chars, r.doi_o)
        if t is None:
            return None
        parts.append("Text of the original paper (truncated):\n<paper>\n" + t + "\n</paper>")
    return "\n".join(parts) + "\n\nReturn JSON with this schema:\n" + SCHEMA


def call_openai(model, prompt):
    """OpenAI Responses API over plain HTTPS (no SDK needed). Sampling settings are left at their defaults, as for
    the other vendor; the reasoning effort is set only when OPENAI_REASONING_EFFORT is given (e.g. none, low)."""
    import os, urllib.request, urllib.error
    if "okey" not in call_openai.__dict__:
        call_openai.okey = read_key(call_llm.key_file, r"sk-[A-Za-z0-9_\-]{20,}") if call_llm.key_file else os.environ["OPENAI_API_KEY"]
    req_body = {"model": model, "instructions": SYSTEM, "input": prompt, "max_output_tokens": 16000}
    effort = os.environ.get("OPENAI_REASONING_EFFORT")
    if effort:
        req_body["reasoning"] = {"effort": effort}
    body = json.dumps(req_body).encode()
    err = None
    for attempt in range(6):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
                                     headers={"Authorization": "Bearer " + call_openai.okey, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                m = json.load(r)
            u = m.get("usage") or {}
            call_llm.last_usage.value = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                                         "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                                         "reasoning_effort": effort or "default", "status": m.get("status")}
            return "".join(c.get("text", "") for o in m.get("output", []) if o.get("type") == "message"
                           for c in o.get("content", []) if c.get("type") == "output_text")
        except urllib.error.HTTPError as e:
            err = RuntimeError(f"OpenAI API error {e.code}: {e.read().decode(errors='ignore')[:400]}")
            if e.code not in (429, 500, 502, 503, 504):
                raise err
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        time.sleep(2 ** attempt)
    raise err


def list_openai_models(key_file):
    import urllib.request
    key = read_key(key_file, r"sk-[A-Za-z0-9_\-]{20,}")
    req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=60) as r:
        ids = sorted(x["id"] for x in json.load(r)["data"])
    print("\n".join(i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4", "chat"))))


def call_llm(model, prompt):
    if not model.startswith("claude"):
        return call_openai(model, prompt)
    import anthropic  # pip install anthropic
    if "client" not in call_llm.__dict__:
        call_llm.client = anthropic.Anthropic(api_key=read_key(call_llm.key_file) if call_llm.key_file else None)
    client = call_llm.client
    for attempt in range(5):
        try:
            m = client.messages.create(model=model, max_tokens=1200, system=SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
            call_llm.last_usage.value = {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens}
            return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            time.sleep(2 ** attempt)
            err = e
    raise err


call_llm.key_file = None
import threading
call_llm.last_usage = threading.local()


def read_key(path, pattern=r"sk-ant-[A-Za-z0-9_\-]{20,}"):
    """Pull an API key out of a text/RTF file without echoing it."""
    m = re.search(pattern, Path(path).read_text(errors="ignore"))
    if not m:
        raise SystemExit("no API key found in " + str(path))
    return m.group(0)


def parse(txt):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", txt, flags=re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    out = {}  # truncated or malformed JSON: recover the fields individually
    m = re.search(r'"probability"\s*:\s*([01](?:\.\d+)?)', txt)
    if m:
        out["probability"] = float(m.group(1))
    m = re.search(r'"recognized_study"\s*:\s*(true|false)', txt)
    if m:
        out["recognized_study"] = m.group(1) == "true"
    m = re.search(r'"recalled_replication_outcome"\s*:\s*"(\w+)"', txt)
    if m:
        out["recalled_replication_outcome"] = m.group(1)
    return out


def load(which):
    if which in ("r1", "r2"):
        return pd.read_csv(ROOT / f"data/processed/round{which[1]}.csv")
    d = pd.read_csv(ROOT / "data/processed/claims.csv")
    return d[d.source == "forrt"].reset_index(drop=True)


def run(a):
    d = load(a.set)
    call_llm.key_file = a.key_file
    if a.text_json:
        TEXTS.update(json.load(open(a.text_json)))
    if a.cond == "probe":
        d = d.dropna(subset=["ref_o"]).drop_duplicates("orig_id")
    if a.limit:
        d = d.sample(min(a.limit, len(d)), random_state=0)
    out = ROOT / "forecasts" / f"{a.set}_{a.cond}_{re.sub(r'[^A-Za-z0-9]+', '-', a.model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    jobs = []
    for r in d.itertuples():
        prompt = build_prompt(r, a.cond, a.pdf_dir, a.max_chars)
        if prompt is None:
            continue
        if a.dry_run:
            print("=" * 80 + f"\n[{r.claim_uid}] prompt chars={len(prompt)}\n" + prompt[:1500])
            return
        for j in range(a.k):
            key = hashlib.md5(f"{r.claim_uid}|{a.cond}|{a.model}|{j}".encode()).hexdigest()
            if key not in done:
                jobs.append((key, r.claim_uid, r.orig_id, j, prompt))

    def work(job):
        key, uid, oid, j, prompt = job
        raw = call_llm(a.model, prompt)
        return {"key": key, "claim_uid": uid, "orig_id": oid, "cond": a.cond, "model": a.model, "sample": j,
                "raw": raw, "usage": getattr(call_llm.last_usage, "value", None), **parse(raw)}

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for i, rec in enumerate(ex.map(work, jobs), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print(f"stopped after {i}/{len(jobs)} (time limit); rerun to resume")
                ex.shutdown(wait=False, cancel_futures=True)
                break
    print(f"{len(jobs)} jobs queued")
    if not a.dry_run:
        print("written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cond", required=True, choices=["blind", "stats", "identified", "fulltext", "probe"])
    r.add_argument("--model", default="claude-sonnet-5-5")
    r.add_argument("--set", default="dev", choices=["dev", "r1", "r2"])
    r.add_argument("--pdf-dir")
    r.add_argument("--text-json", help="JSON {file name: extracted text} used instead of --pdf-dir")
    r.add_argument("--key-file", help="file containing the API key (otherwise ANTHROPIC_API_KEY is used)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--max-seconds", type=int, default=0, help="stop early and resume on the next run")
    r.add_argument("--max-chars", type=int, default=100000)
    r.add_argument("--limit", type=int)
    r.add_argument("--k", type=int, default=1, help="independent samples per claim at the default sampling settings")
    r.add_argument("--dry-run", action="store_true")
    mm = sub.add_parser("models", help="list the OpenAI models the key can use")
    mm.add_argument("--key-file", required=True)
    a = ap.parse_args()
    if a.cmd == "models":
        list_openai_models(a.key_file)
    else:
        run(a)
