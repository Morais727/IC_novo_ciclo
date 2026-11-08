#!/usr/bin/env python
"""Fase 3, H7 — consistência da vizinhança entre camadas (fixado ANTES de olhar os números; só CPU, sobre os
registros do lg_shadow em results/fase2/raw, que guardam por round e tensor a matriz de cossenos entre clientes das
atualizações centradas C_i = Δ_i - mediana).

Hipótese: a atualização do LP é montada por camada (L* vem do modelo envenenado, o resto do treino limpo do próprio
atacante). Num honesto, quem é parecido com ele num tensor é parecido em todos (mesmos dados). No atacante, a
vizinhança nos tensores de L* não bate com a dos outros tensores — e o ruído em L* (ν = 1) só piora isso para ele.
Escore por (round, tensor g, cliente i): s = 1 - pearson(cmat_g[i, j≠i], média_{h≠g} cmat_h[i, j≠i]); z robusto
entre os clientes no (round, g). Variantes: V1 média sobre os outros 9 tensores; V2 só os outros tensores de peso.
Critérios (dev = 42, 44, 45; teste = 46, 47, 48, só se dev passar):
  H7a AUC >= 0,9 em cada semente; positivos = (atacante, g ∈ L* do round), negativos = (honesto, g) em todo g;
  H7b regra z > 3 marca >= 50% da massa de L* dos atacantes e <= 5% dos (honesto, tensor) — a mesma exigência
      da Fase 2. Se H7a e H7b passarem em dev com uma variante, ela vai ao teste e, passando, para o ciclo fechado.
Uso: python experiments/fase3_xlayer.py [dev|test]
"""
import json, sys
import numpy as np
from pathlib import Path

RAW = Path("results/fase2/raw")
SEEDS = {"dev": (42, 44, 45), "test": (46, 47, 48)}


def rz(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (v - med) / mad


def auc(pos, neg):
    pos, neg = np.sort(np.asarray(pos)), np.asarray(neg)
    return float((np.searchsorted(pos, neg, side="left").sum() + 0.5 * (np.searchsorted(pos, neg, side="right") - np.searchsorted(pos, neg, side="left")).sum()) / (len(pos) * len(neg)) * -1 + 1) if len(pos) and len(neg) else float("nan")


def analyze(seed):
    d = json.load(open(RAW / f"lp_full_indiv_nu1.0__lg_shadow__a0.1_s{seed}.json"))
    log, mal = d["diag_log"], d["malicious_ids"]
    lp = [set(e["layers"]) for e in d["attack_params"]["_lp_log"]]
    names = list(log[0]); N = len(log[0][names[0]]["cmat"])
    sizes = None
    out = {}
    for var in ("V1", "V2"):
        pos, neg, flag_mal, flag_hon, n_hon = [], [], [], 0, 0
        for r, rl in enumerate(log):
            cm = {g: np.array(rl[g]["cmat"]) for g in names}
            Lr = lp[r] if r < len(lp) else set()
            for g in names:
                others = [h for h in names if h != g and (var == "V1" or h.endswith("weight"))]
                ref = np.mean([cm[h] for h in others], 0)
                s = np.zeros(N)
                for i in range(N):
                    m = np.arange(N) != i
                    a, b = cm[g][i, m], ref[i, m]
                    s[i] = 1 - (np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0)
                z = rz(s)
                for i in range(N):
                    if i in mal:
                        if g in Lr:
                            pos.append(z[i]); flag_mal.append(z[i] > 3)
                    else:
                        neg.append(z[i]); flag_hon += z[i] > 3; n_hon += 1
        out[var] = {"auc": auc(pos, neg), "L*_marcado": float(np.mean(flag_mal)) if flag_mal else float("nan"),
                    "hon_marcado": flag_hon / n_hon, "n_pos": len(pos)}
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    res = {s: analyze(s) for s in SEEDS[mode]}
    for s, v in res.items():
        print(s, {k: {kk: round(vv, 3) for kk, vv in x.items()} for k, x in v.items()})
    json.dump(res, open(f"results/fase3/xlayer_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
