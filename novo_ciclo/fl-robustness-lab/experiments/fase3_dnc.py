#!/usr/bin/env python
"""Adendo A3 (fase3_repair.py) — busca diferencial por cliente (src/defenses/dnc.py) nos dumps (s44, s45; rounds
10, 20, 30). Fixado antes: escore do cliente = max_c G_i(c), 100 passos, lam 1e-2. Critério para ciclo fechado:
os 2 atacantes nos 2 maiores escores em >= 4 dos 6 pares (semente, round) e AUC >= 0,9 em cada semente.
Contexto (não é critério): com o gatilho verdadeiro, o efeito descontado separa nos rounds 10–20 (oráculo).
Saída: results/fase3/dnc.json / dnc.md. Uso: python experiments/fase3_dnc.py
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from experiments.fase3_clientnc import auc
from src.defenses.dnc import gain
from src.defenses.nc import _tensors
from src.federated.utils import load_vector
from src.models.cnn import build_model

D, OUT = Path("results/fase3/dumps"), Path("results/fase3")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf = OUT / "dnc.json"; cache = json.load(open(cf)) if cf.exists() else {}
    files = [D / f"lp_full_indiv_nu1.0_s{s}_r{r}.pt" for r in (20, 10, 30) for s in (44, 45)]
    for f in files:
        if f.stem in cache or not f.exists():
            continue
        t0 = time.time(); d = torch.load(f, weights_only=False)
        seed = int(f.stem.split("_s")[-1].split("_")[0])
        hold, _ = holdout_for(cfg_from_name(f"lp_full_indiv_nu1.0__fedavg__a0.1_s{seed}"))
        X, Y = _tensors(hold)
        fg, fi = build_model("smallcnn").to(dev), build_model("smallcnn").to(dev)
        load_vector(fg, d["global"].to(dev)); fg.eval(); fi.eval()
        for p in list(fg.parameters()) + list(fi.parameters()):
            p.requires_grad_(False)
        G = np.zeros((d["deltas"].shape[0], 10))
        for i in range(d["deltas"].shape[0]):
            load_vector(fi, (d["global"] + d["deltas"][i]).to(dev))
            for p in fi.parameters():
                p.requires_grad_(False)
            for c in range(10):
                G[i, c] = gain(fi, fg, X, Y, c, dev)[0]
        cache[f.stem] = {"G": G.tolist(), "malicious": d["malicious"]}
        json.dump(cache, open(cf, "w")); print(f.stem, f"{time.time()-t0:.0f}s", flush=True)
    md = ["# Busca diferencial por cliente (A3)", "", "| caso | postos dos atacantes | AUC | classe do máx. (atac.) |", "|---|---|---|---|"]
    per = {}
    for k, v in sorted(cache.items()):
        G, mal = np.array(v["G"]), v["malicious"]; s = G.max(1); hon = [i for i in range(len(s)) if i not in mal]
        order = np.argsort(-s).tolist(); seed = k.split("_s")[-1].split("_")[0]
        per.setdefault(seed, [[], []]); per[seed][0] += s[mal].tolist(); per[seed][1] += s[hon].tolist()
        md.append(f"| {k} | {sorted(order.index(a) + 1 for a in mal)} | {auc(s[mal], s[hon]):.2f} | {[int(G[a].argmax()) for a in mal]} |")
    md += ["", "AUC por semente: " + ", ".join(f"s{s} {auc(*v):.2f}" for s, v in per.items())]
    (OUT / "dnc.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
