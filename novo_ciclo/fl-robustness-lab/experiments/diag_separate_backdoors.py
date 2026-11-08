#!/usr/bin/env python
"""Diagnóstico (16/10): que sinal separa backdoors treinados em separado (pixel, PGD, DBA) dos honestos com
α = 0,1? Motivação: LayerGuard v1 não os barrou e o Multi-Krum barrou o pixel. Agregação oráculo, 10 rounds,
semente 42. Sinais por cliente (↑ = malicioso), todos com z robusto entre clientes no round:
  krum         soma das distâncias quadradas aos n-f-2 vizinhos mais próximos (modelo inteiro, f = 4)
  norm         norma do delta (modelo inteiro)
  sim_t, norm_t, sn_t   máximo por tensor de z_sim, z_norm e max(z_sim,0)+max(z_norm,0) (LayerGuard)
  krum_t       máximo por tensor do z do escore Krum calculado por tensor
Saída: results/diag_separate/{raw.csv, summary.md}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.layer_profile import robust_z
from experiments.layer_similarity import max_other_cos
from experiments.ragnar_eval import ATTACKS, COMMON, OPTIONAL_ATTACKS
from src.attacks.recent import tensor_groups
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig
from src.federated.simulate import _build_data, _train_round
from src.federated.utils import flatten
from src.models.cnn import build_model


def krum_scores(x, f=4):
    d = torch.cdist(x, x) ** 2
    k = x.shape[0] - f - 2
    return d.sort(dim=1).values[:, 1:k + 1].sum(dim=1).numpy()


def signals(deltas, groups):
    c = deltas - deltas.median(dim=0).values
    names = list(groups)
    zs = robust_z(np.stack([max_other_cos(c[:, groups[g]]) for g in names], 1))
    zn = robust_z(np.stack([c[:, groups[g]].norm(dim=1).numpy() for g in names], 1))
    zk = robust_z(np.stack([krum_scores(deltas[:, groups[g]]) for g in names], 1))
    return {"krum": robust_z(krum_scores(deltas)), "norm": robust_z(deltas.norm(dim=1).numpy()),
            "sim_t": zs.max(1), "norm_t": zn.max(1), "sn_t": (np.maximum(zs, 0) + np.maximum(zn, 0)).max(1),
            "krum_t": zk.max(1), "top_krum_t": [names[i] for i in zk.argmax(1)]}


def collect(attack, rounds=10, alpha=0.1, seed=42, device="cuda"):
    spec = {**ATTACKS, **OPTIONAL_ATTACKS}[attack]
    cfg = ScenarioConfig(name=f"diag_{attack}", attack=attack, num_malicious=spec["num_malicious"],
                         attack_params=dict(spec["attack_params"]), defense="fedavg", seed=seed, device=device,
                         **{**COMMON, "rounds": rounds, "dirichlet_alpha": alpha})
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ctor = lambda: build_model(cfg.model)
    data, _, _, _ = _build_data(cfg, rng)
    mal = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())
    groups = tensor_groups(ctor()); benign = [c for c in range(cfg.num_clients) if c not in mal]
    gv = flatten(ctor()).cpu(); rows = []
    for t in range(1, rounds + 1):
        deltas, w = _train_round(ctor, gv, data, torch.device(device), cfg, mal, ATTACK_REGISTRY[attack], cfg.attack_params)
        deltas = deltas.cpu(); s = signals(deltas, groups)
        for c in range(cfg.num_clients):
            rows.append({"attack": attack, "round": t, "client": c, "malicious": int(c in mal), **{k: v[c] for k, v in s.items()}})
        ww = w[benign].float(); gv = gv + (deltas[benign] * (ww / ww.sum()).unsqueeze(1)).sum(0)
        print(f"  [{attack}] round {t}/{rounds}", flush=True)
    return pd.DataFrame(rows)


def main():
    out = Path("results/diag_separate"); out.mkdir(parents=True, exist_ok=True)
    p = out / "raw.csv"
    df = pd.read_csv(p) if p.exists() else pd.concat([collect(a) for a in ("backdoor_pixel", "backdoor_pgd", "dba")])
    df.to_csv(p, index=False)
    S = ["krum", "norm", "sim_t", "norm_t", "sn_t", "krum_t"]
    md = ["# Que sinal separa backdoors treinados em separado (α = 0,1)?", "",
          "| ataque | " + " | ".join(S) + " |", "|---" * (len(S) + 1) + "|"]
    for a, g in df.groupby("attack", sort=False):
        aucs = [np.mean([roc_auc_score(r.malicious, r[s]) for _, r in g.groupby("round")]) for s in S]
        md.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in aucs) + " |")
    md += ["", "Recall / FP com z > 3:", ""]
    for a, g in df.groupby("attack", sort=False):
        m, h = g[g.malicious == 1], g[g.malicious == 0]
        md.append(f"- {a}: " + "; ".join(f"{s} {(m[s] > 3).mean():.2f}/{(h[s] > 3).mean():.2f}" for s in S)
                  + f"; tensor krum_t dos maliciosos: {m.top_krum_t.value_counts().head(2).to_dict()}")
    (out / "summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
