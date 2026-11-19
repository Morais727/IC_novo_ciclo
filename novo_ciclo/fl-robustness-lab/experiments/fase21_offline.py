#!/usr/bin/env python
"""Fase 21 (18/11/2026) — teste SEM TREINO de regras de confirmação que não excluam honestos quando a detecção fica
ruidosa no fim do treino (achado do diagnóstico da fase 20). Usa os escores por tensor (z_sim, z_norm) que o
agregador-sombra (lg_shadow) gravou em todas as rodadas: ataque padrão (fase 8E, 6 sementes), atacante adaptativo
(fase 8G, 6 sementes) e sem ataque (fase 11, α 0,9 e 0,1). Para cada rodada: contagem de acusações por cliente
(nota > 3), e cada regra decide quem é confirmado. Mede revocação (atacantes confirmados) e honestos confirmados.
Regras: fixa (T = 8); adaptativa  T_r = max(T, mediana(c) + k·1,4826·MAD(c)), k ∈ {2, 3}."""
import glob, json, re, sys
import numpy as np

def counts(rec):
    names = [k for k in rec if isinstance(rec[k], dict) and "z_sim" in rec[k]]
    zs = np.array([rec[k]["z_sim"] for k in names]); zn = np.array([rec[k]["z_norm"] for k in names])
    return ((np.maximum(zs, 0) + np.maximum(zn, 0)) > 3).sum(0)

def rule(c, name, T=8):
    if name == "fixa":
        Tr = T
    else:
        k = float(name.split("k")[1]); med = np.median(c); mad = np.median(np.abs(c - med))
        Tr = max(T, med + k * 1.4826 * mad)
    return c >= Tr, Tr

RULES = ["fixa", "adapt_k2", "adapt_k3"]
WIN = {"pré-ataque (1–79)": (1, 79), "início do ataque (80–129)": (80, 129), "fim do ataque (130–200)": (130, 200), "todas": (1, 205)}
out = {}
sets = {"padrão": "results/fase8e/raw/laysel_paper__lg_shadow__a0.9_s*.json", "adaptativo": "results/fase8g/raw/laysel_adapt_T8__lg_shadow__a0.9_s*.json",
        "sem ataque α0,9": "results/fase11/raw/noattack__lg_shadow__a0.9_s*.json", "sem ataque α0,1": "results/fase11/raw/noattack__lg_shadow__a0.1_s*.json"}
for sname, pat in sets.items():
    for f in sorted(glob.glob(pat)):
        r = json.load(open(f)); mal = set(r.get("malicious_ids") or [])
        noatk = "noattack" in f
        seed = int(re.search(r"_s(\d+)\.json", f).group(1))
        per = []
        for t, rec in enumerate(r["diag_log"], 1):
            c = counts(rec)
            row = {"t": t, "c": c.tolist()}
            for rn in RULES:
                conf, Tr = rule(c, rn)
                atk = [i for i in mal] if (not noatk and 80 <= t <= 200) else []
                hon = [i for i in range(len(c)) if (noatk or i not in mal)]
                row[rn] = {"rec": (sum(conf[i] for i in atk) / len(atk)) if atk else None, "fp": int(sum(conf[i] for i in hon)), "Tr": float(Tr)}
            per.append(row)
        out[f"{sname}|{seed}"] = per
        print(sname, seed, "ok", flush=True)
        del r
json.dump(out, open("results/fase21/offline.json", "w"))
# resumo
lines = ["| cenário | janela | regra | revocação | honestos confirmados por rodada |", "|---|---|---|---|---|"]
for sname in sets:
    keys = [k for k in out if k.startswith(sname + "|")]
    for wn, (a, b) in WIN.items():
        for rn in RULES:
            recs = [x[rn]["rec"] for k in keys for x in out[k] if a <= x["t"] <= b and x[rn]["rec"] is not None]
            fps = [x[rn]["fp"] for k in keys for x in out[k] if a <= x["t"] <= b]
            if not fps: continue
            lines.append(f"| {sname} ({len(keys)} sem.) | {wn} | {rn} | {np.mean(recs):.2f} | {np.mean(fps):.2f} |" if recs else f"| {sname} ({len(keys)} sem.) | {wn} | {rn} | — | {np.mean(fps):.2f} |")
open("results/fase21/RESULTADO_offline.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
