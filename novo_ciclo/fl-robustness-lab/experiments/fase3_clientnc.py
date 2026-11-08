#!/usr/bin/env python
"""Diagnóstico D1 (adendo A2 em experiments/fase3_repair.py): NC enxuto (src/defenses/nc.py::invert, 100 passos)
no modelo de cada cliente (global + Δ_i) nos rounds salvos por fedavg_dump. Escores por cliente:
  S1  índice de anomalia dentro do modelo (como o NC): max_c (mediana_c L1 - L1_c) / MAD
  S2  comparação entre clientes: para cada classe c, z robusto de -L1_{i,c} entre os clientes; escore = max_c
Critério (fixado antes): AUC >= 0,9 em cada semente e os 2 atacantes nos 2 maiores escores em >= 4 de 6 casos.
Saída: results/fase3/clientnc.json e clientnc.md. Uso: python experiments/fase3_clientnc.py
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from src.defenses.nc import invert, _tensors
from src.federated.utils import load_vector
from src.models.cnn import build_model

D, OUT = Path("results/fase3/dumps"), Path("results/fase3")


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def rz(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (v - med) / mad


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf = OUT / "clientnc.json"
    cache = json.load(open(cf)) if cf.exists() else {}
    for f in sorted(D.glob("*.pt")):
        if f.stem in cache:
            continue
        t0 = time.time()
        d = torch.load(f, weights_only=False)
        seed = int(f.stem.split("_s")[-1].split("_")[0])
        hold, _ = holdout_for(cfg_from_name(f"lp_full_indiv_nu1.0__fedavg__a0.1_s{seed}"))
        X, Y = _tensors(hold); X, Y = X.to(dev), Y.to(dev)
        m = build_model("smallcnn").to(dev)
        L1 = np.zeros((d["deltas"].shape[0], 10))
        for i in range(d["deltas"].shape[0]):
            load_vector(m, (d["global"] + d["deltas"][i]).to(dev))
            for c in range(10):
                L1[i, c] = invert(m, X, Y, c, dev, steps=100, seed=0)[2]
        cache[f.stem] = {"L1": L1.tolist(), "malicious": d["malicious"]}
        json.dump(cache, open(cf, "w"))
        print(f.stem, f"{time.time()-t0:.0f}s", flush=True)
    md = ["# NC no modelo de cada cliente (D1)", "", "| caso | atacantes | S1 AUC | S1 postos atac. | S2 AUC | S2 postos atac. | classe S2 dos atac. |", "|---|---|---|---|---|---|---|"]
    per_seed = {}
    for k, v in sorted(cache.items()):
        L1, mal = np.array(v["L1"]), v["malicious"]
        hon = [i for i in range(len(L1)) if i not in mal]
        med = np.median(L1, 1, keepdims=True); mad = 1.4826 * np.median(np.abs(L1 - med), 1, keepdims=True) + 1e-9
        s1 = ((med - L1) / mad).max(1)
        z = np.stack([rz(-L1[:, c]) for c in range(10)], 1); s2 = z.max(1)
        rank = lambda s: sorted((np.argsort(-s).tolist().index(a) + 1 for a in mal))
        seed = k.split("_s")[-1].split("_")[0]
        for nm, s in (("S1", s1), ("S2", s2)):
            per_seed.setdefault((seed, nm), [[], []]); per_seed[(seed, nm)][0] += s[mal].tolist(); per_seed[(seed, nm)][1] += s[hon].tolist()
        md.append(f"| {k} | {mal} | {auc(s1[mal], s1[hon]):.2f} | {rank(s1)} | {auc(s2[mal], s2[hon]):.2f} | {rank(s2)} | {[int(z[a].argmax()) for a in mal]} |")
    md += ["", "AUC por semente (rounds juntos): " + ", ".join(f"s{s} {nm} {auc(*v):.2f}" for (s, nm), v in sorted(per_seed.items()))]
    (OUT / "clientnc.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
