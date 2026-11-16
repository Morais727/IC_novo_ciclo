#!/usr/bin/env python
"""IC de 95% para o ASR final da tabela principal (cenário 8E, sementes 42–45). 13/11/2026.
Com n=4 o bootstrap percentil é grosseiro (só 35 reamostragens distintas), então reportamos também o IC t de Student
(t_{0,975;3}=3,182). Saída: results/fase8e/IC.md e IC.json."""
import json, glob, re
import numpy as np
from pathlib import Path

rows = {}
for f in glob.glob("results/fase8e/raw/laysel_paper__*__a0.9_s4[2-5].json"):
    m = re.match(r".*laysel_paper__(.+)__a0\.9_s(\d+)\.json", f)
    d, s = m.group(1), int(m.group(2))
    if re.search(r"_T\d+$", d):
        continue
    h = json.load(open(f))["history"][-1]
    rows.setdefault(d, {})[s] = (h["backdoor_asr"], h["main_accuracy"])
rng = np.random.default_rng(0)
out, L = {}, ["| defesa | n | ASR média | IC95 t | IC95 bootstrap | MA média |", "|---|---|---|---|---|---|"]
for d, v in sorted(rows.items(), key=lambda kv: np.mean([x[0] for x in kv[1].values()])):
    a = np.array([x[0] for x in v.values()]); ma = np.mean([x[1] for x in v.values()]); n = len(a)
    if n < 2:
        continue
    t = {2: 12.706, 3: 4.303, 4: 3.182}[n] if n <= 4 else 2.776
    half = t * a.std(ddof=1) / np.sqrt(n)
    bs = [rng.choice(a, n).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out[d] = dict(n=n, mean=a.mean(), t_lo=a.mean() - half, t_hi=a.mean() + half, b_lo=lo, b_hi=hi, ma=ma, seeds=sorted(v))
    nome = "sem defesa" if d == "lg_shadow" else d
    L.append(f"| {nome} | {n} | {a.mean():.1f} | [{max(0, a.mean()-half):.1f}, {a.mean()+half:.1f}] | [{lo:.1f}, {hi:.1f}] | {ma:.1f} |")
Path("results/fase8e/IC.md").write_text("# IC 95% do ASR final (8E, sementes 42–45)\n\n" + "\n".join(L) + "\n")
json.dump(out, open("results/fase8e/IC.json", "w"), indent=1)
print("\n".join(L))
