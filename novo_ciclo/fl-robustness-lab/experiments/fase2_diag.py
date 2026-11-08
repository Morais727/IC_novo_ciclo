#!/usr/bin/env python
"""Fase B (fase2_lab.py): lê os registros do lg_shadow e mede, por semente, onde o escore do layer_guard_sn
deixa os atacantes passar. Fora do circuito: avalia regras de marcação candidatas sobre os mesmos registros.

Métricas por regra (rounds 1-40):
  mal_L*   fração da massa de parâmetros de L* (por round, por atacante) marcada  -> quanto do backdoor sai
  hon      fração dos parâmetros honestos marcados                                -> custo
Uso: python experiments/fase2_diag.py [arquivos...]   (padrão: sementes de desenvolvimento 42, 44, 45)
"""
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path("results/fase2/raw")
DEV_FILES = [RAW / f"lp_full_indiv_nu1.0__lg_shadow__a0.1_s{s}.json" for s in (42, 44, 45)] + \
            [RAW / "lp_full_indiv__lg_shadow__a0.1_s44.json"]


def load(f):
    d = json.load(open(f))
    log = d["diag_log"]
    names = list(log[0])
    sizes = None
    T, N = len(log), len(log[0][names[0]]["cos"])
    arr = lambda key: np.array([[log[r][g][key] for g in names] for r in range(T)])   # [T, G, N]
    lp_layers = [e["layers"] for e in d["attack_params"].get("_lp_log", [])]
    cm = arr("cmat") if "cmat" in log[0][names[0]] else None   # [T, G, N, N]
    return dict(cm=cm, d=d, names=names, T=T, N=N, zs=arr("z_sim"), zn=arr("z_norm"), cos=arr("cos"), nrm=arr("norm"),
                mal=d["malicious_ids"], lp=lp_layers)


SIZES = {"features.0.weight": 864, "features.0.bias": 32, "features.3.weight": 18432, "features.3.bias": 64,
         "features.6.weight": 36864, "features.6.bias": 64, "fc1.weight": 131072, "fc1.bias": 128,
         "fc2.weight": 1280, "fc2.bias": 10}


def evaluate(L, flags):
    """flags: bool [T, G, N] (True = tensor do cliente sai)."""
    sz = np.array([SIZES[g] for g in L["names"]])
    hon = [i for i in range(L["N"]) if i not in L["mal"]]
    hon_frac = (flags[:, :, hon] * sz[None, :, None]).sum() / (sz.sum() * len(hon) * L["T"])
    mal_tot = mal_hit = 0
    for r in range(L["T"]):
        gi = [L["names"].index(g) for g in L["lp"][r]] if r < len(L["lp"]) else []
        for m in L["mal"]:
            mal_tot += sz[gi].sum()
            mal_hit += (flags[r, gi, m] * sz[gi]).sum()
    return mal_hit / max(mal_tot, 1), hon_frac


def score(L):
    return np.maximum(L["zs"], 0) + np.maximum(L["zn"], 0)


def describe(L, tag):
    s = score(L)
    print(f"\n## {tag}  atacantes {L['mal']}")
    lp_count = {}
    for ls in L["lp"]:
        for g in ls:
            lp_count[g] = lp_count.get(g, 0) + 1
    print("L* (rounds em que cada tensor entrou):", lp_count)
    hon = [i for i in range(L["N"]) if i not in L["mal"]]
    print("tensor | escore médio atac. A / B (z_sim, z_norm) nos rounds em que o tensor está em L* | p95 honestos")
    for gi, g in enumerate(L["names"]):
        rs = [r for r in range(L["T"]) if r < len(L["lp"]) and g in L["lp"][r]]
        if not rs:
            continue
        cells = []
        for m in L["mal"]:
            cells.append(f"{s[rs, gi, m].mean():.1f} ({L['zs'][rs, gi, m].mean():.1f}, {L['zn'][rs, gi, m].mean():.1f}) "
                         f">3 em {np.mean(s[rs, gi, m] > 3):.0%}")
        print(f"  {g:18s} n={len(rs):2d} | " + " | ".join(cells) + f" | {np.percentile(s[rs][:, gi][:, hon], 95):.1f}")
    # cosseno entre atacantes vs melhor honesto
    for gi, g in enumerate(L["names"]):
        rs = [r for r in range(L["T"]) if r < len(L["lp"]) and g in L["lp"][r]]
        if len(rs) >= 10:
            print(f"  cos máx {g}: atac. {np.round(L['cos'][rs][:, gi][:, L['mal']].mean(0), 2)}, "
                  f"honestos mediana {np.median(L['cos'][rs][:, gi][:, hon]):.2f} p95 {np.percentile(L['cos'][rs][:, gi][:, hon], 95):.2f}")


def rules():
    """Regras candidatas (fora do circuito). Retorna {nome: f(L) -> flags [T, G, N]}."""
    def r0(L, thr=3.0):
        return score(L) > thr

    def ewma(L, beta=0.5, thr=3.0):
        s, e = score(L), np.zeros_like(score(L))
        for t in range(L["T"]):
            e[t] = s[t] if t == 0 else beta * e[t - 1] + (1 - beta) * s[t]
        return e > thr

    def pooled(L, thr=3.0, base=1.0):
        # evidência do cliente somada nos tensores: marca TODOS os tensores com escore > base de um cliente
        # cuja soma de excessos passa thr (o LP espalhado em vários tensores soma evidência)
        s = score(L)
        tot = np.maximum(s - base, 0).sum(axis=1, keepdims=True)
        return (s > thr) | ((tot > thr) & (s > base))

    def rz(v, axis):
        med = np.median(v, axis=axis, keepdims=True)
        return (v - med) / (1.4826 * np.median(np.abs(v - med), axis=axis, keepdims=True) + 1e-12)

    def pair_acc(L, beta=0.8, thr=3.0, with_norm=True):
        # evidência acumulada POR PAR e por tensor: A = média exponencial (no tempo) da matriz de cossenos;
        # escore do cliente = z robusto de max_{j != i} A_ij (+ z_norm do round)
        cm = L["cm"].copy()
        T, G, N, _ = cm.shape
        A = np.zeros_like(cm)
        for t in range(T):
            A[t] = cm[t] if t == 0 else beta * A[t - 1] + (1 - beta) * cm[t]
        idx = np.arange(N)
        A[:, :, idx, idx] = -2
        z = rz(A.max(axis=3), axis=2)
        if with_norm:
            z = np.maximum(z, 0) + np.maximum(L["zn"], 0)
        return z > thr

    out = {"R0 por tensor > 3": r0}
    if all(L is not None for L in [None]) or True:
        for b in (0.5, 0.8, 0.9):
            out[f"par acumulado β={b} (+norma) > 3"] = (lambda b: lambda L: pair_acc(L, b) if L["cm"] is not None else np.zeros_like(L["zs"], bool))(b)
        out["par acumulado β=0.8 sem norma > 3"] = lambda L: pair_acc(L, 0.8, with_norm=False) if L["cm"] is not None else np.zeros_like(L["zs"], bool)
    for thr in (2.0, 2.5):
        out[f"R0 limiar {thr}"] = (lambda t: lambda L: r0(L, t))(thr)
    for b, t in ((0.5, 3.0), (0.7, 3.0), (0.5, 2.0)):
        out[f"EWMA β={b} > {t}"] = (lambda b, t: lambda L: ewma(L, b, t))(b, t)
    for base, t in ((1.0, 3.0), (1.5, 3.0), (1.0, 4.0), (2.0, 4.0)):
        out[f"soma no cliente (base {base}) > {t}"] = (lambda b, t: lambda L: pooled(L, t, b))(base, t)
    return out


def main():
    files = [Path(f) for f in sys.argv[1:]] or [f for f in DEV_FILES if f.exists()]
    Ls = {f.stem.replace("__lg_shadow__a0.1", ""): load(f) for f in files}
    for k, L in Ls.items():
        describe(L, k)
    print("\n## Regras candidatas: fração de L* dos atacantes marcada / fração de parâmetros honestos marcada")
    print("| regra | " + " | ".join(Ls) + " |")
    for name, f in rules().items():
        cells = [f"{a:.0%} / {h:.1%}" for a, h in (evaluate(L, f(L)) for L in Ls.values())]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
