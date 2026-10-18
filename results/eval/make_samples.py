#!/usr/bin/env python3
"""Rebuild the fixed evaluation samples used in results/ (seed 1234) from the Hugging Face datasets.
usage: make_samples.py [OUT_DIR=samples]      needs: pip install pandas pyarrow huggingface_hub
Writes gsm8k.jsonl (100 of the test split), math500.jsonl (50), humaneval.jsonl (50), mmlupro.jsonl (200 of test),
ceval.jsonl (100 of the val split over all subjects). The datasets keep their own licenses; they are not redistributed here."""
import json, os, random, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = sys.argv[1] if len(sys.argv) > 1 else "samples"
CACHE = os.path.join(OUT, "_hf")
os.makedirs(OUT, exist_ok=True)


def dl(repo, f):
    return hf_hub_download(repo, f, repo_type="dataset", local_dir=CACHE)


def pick(df, n):
    idx = sorted(random.Random(1234).sample(range(len(df)), n))
    return df.iloc[idx]


g = pd.read_parquet(dl("openai/gsm8k", "main/test-00000-of-00001.parquet"))
m = pd.read_json(dl("HuggingFaceH4/MATH-500", "test.jsonl"), lines=True)
h = pd.read_parquet(dl("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"))
p = pd.read_parquet(dl("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"))
fs = [f for f in HfApi().list_repo_files("ceval/ceval-exam", repo_type="dataset") if f.endswith("/val-00000-of-00001.parquet")]
c = pd.concat([pd.read_parquet(dl("ceval/ceval-exam", f)).assign(subject=f.split("/")[0]) for f in fs]).reset_index(drop=True)
out = {
    "gsm8k": [{"id": int(i), "question": r.question, "answer": r.answer.split("####")[-1].strip().replace(",", "")}
              for i, r in pick(g, 100).iterrows()],
    "math500": [{"id": r.unique_id, "problem": r.problem, "answer": r.answer, "level": int(r.level), "subject": r.subject}
                for _, r in pick(m, 50).iterrows()],
    "humaneval": [{"id": r.task_id, "prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}
                  for _, r in pick(h, 50).iterrows()],
    "mmlupro": [{"id": int(r.question_id), "question": r.question, "options": list(r.options), "answer": r.answer,
                 "category": r.category} for _, r in pick(p, 200).iterrows()],
    "ceval": [{"id": f"{r.subject}-{r.id}", "subject": r.subject, "question": r.question, "options": [r.A, r.B, r.C, r.D],
               "answer": r.answer} for _, r in pick(c, 100).iterrows()],
}
for k, v in out.items():
    with open(os.path.join(OUT, f"{k}.jsonl"), "w") as f:
        for x in v:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(k, len(v))
