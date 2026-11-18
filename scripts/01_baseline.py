"""Behavioral baseline: greedy accuracy of a*b for a,b in 1..99."""
import itertools, json, torch
import pandas as pd
from common import load, prompt

tok, model = load()
tok.padding_side = "left"
tok.pad_token = tok.eos_token
print("tokenization:", [tok.decode(t) for t in tok(prompt(23, 47)).input_ids])
print("answer toks 1081:", [tok.decode(t) for t in tok("1081\n", add_special_tokens=False).input_ids])

rows = []
pairs = list(itertools.product(range(1, 100), range(1, 100)))
B = 256
for i in range(0, len(pairs), B):
    batch = pairs[i:i + B]
    enc = tok([prompt(a, b) for a, b in batch], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for (a, b), t in zip(batch, texts):
        pred = t.split("\n")[0].strip()
        rows.append(dict(a=a, b=b, true=a * b, pred=pred, correct=pred == str(a * b)))
df = pd.DataFrame(rows)
df.to_csv("results/baseline.csv", index=False)
df["da"] = df.a.astype(str).str.len(); df["db"] = df.b.astype(str).str.len()
df["single_tok"] = df.true < 1000
print(df.groupby(["da", "db"]).correct.mean())
print(df.groupby("single_tok").correct.mean())
print("overall", df.correct.mean())
