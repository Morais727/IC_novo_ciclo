#!/usr/bin/env python
"""Fase 21b (18/11/2026) — duas ideias para não confirmar honestos quando a régua fica ruidosa, testadas sem treino nos
logs do agregador-sombra (cos e norma brutos por tensor e por cliente, em todas as rodadas):
  piso:     z robusto com PISO na escala: z = (x − med) / max(1,4826·MAD, piso); piso_cos = 0,05 (cosseno) e
            piso_norma = 0,25·mediana (norma). Evita que diferenças minúsculas virem z enorme quando todos se parecem.
  memoria:  confirma só quem passou de T em pelo menos 2 das últimas 3 rodadas (o atacante insiste; o falso alarme
            honesto tende a ser esporádico).
  piso+memoria: as duas juntas.
Mede revocação (atacantes confirmados por rodada) e honestos confirmados por rodada, por janela."""
import glob, json, re
import numpy as np
T = 8

def rz(v, floor):
    med = np.median(v); mad = np.median(np.abs(v - med))
    return (v - med) / max(1.4826 * mad, floor, 1e-12)

def counts(rec, piso):
    names = [k for k in rec if isinstance(rec[k], dict) and "cos" in rec[k]]
    c = 0
    for k in names:
        cs = np.array(rec[k]["cos"]); nr = np.array(rec[k]["norm"])
        if piso:
            zs, zn = rz(cs, 0.05), rz(nr, 0.25 * np.median(nr))
        else:
            zs, zn = np.array(rec[k]["z_sim"]), np.array(rec[k]["z_norm"])
        c = c + ((np.maximum(zs, 0) + np.maximum(zn, 0)) > 3)
    return np.asarray(c)

sets = {"padrão": "results/fase8e/raw/laysel_paper__lg_shadow__a0.9_s*.json", "adaptativo": "results/fase8g/raw/laysel_adapt_T8__lg_shadow__a0.9_s*.json",
        "sem ataque α0,9": "results/fase11/raw/noattack__lg_shadow__a0.9_s*.json", "sem ataque α0,1": "results/fase11/raw/noattack__lg_shadow__a0.1_s*.json"}
RULES = ["fixa", "piso", "memoria", "piso+memoria"]
WIN = {"início do ataque (80–129)": (80, 129), "fim do treino (130–200)": (130, 200), "todas": (1, 205)}
acc = {}
for sname, pat in sets.items():
    for f in sorted(glob.glob(pat)):
        r = json.load(open(f)); mal = set(r.get("malicious_ids") or []); noatk = "noattack" in f
        hist = {"fixa": [], "piso": []}
        for t, rec in enumerate(r["diag_log"], 1):
            cf = counts(rec, False); cp = counts(rec, True)
            hist["fixa"].append(cf >= T); hist["piso"].append(cp >= T)
            n = len(cf)
            dec = {"fixa": hist["fixa"][-1], "piso": hist["piso"][-1]}
            for base, nm in (("fixa", "memoria"), ("piso", "piso+memoria")):
                last = hist[base][-3:]
                dec[nm] = (np.sum(last, axis=0) >= 2) if len(last) >= 2 else np.zeros(n, bool)
            atk = sorted(mal) if (not noatk and 80 <= t <= 200) else []
            hon = [i for i in range(n) if noatk or i not in mal]
            for rn in RULES:
                for wn, (a, b) in WIN.items():
                    if a <= t <= b:
                        d = acc.setdefault((sname, wn, rn), {"rec": [], "fp": []})
                        d["fp"].append(int(dec[rn][hon].sum()))
                        if atk: d["rec"].append(float(dec[rn][atk].mean()))
        print(sname, f, "ok", flush=True); del r
lines = ["| cenário | janela | regra | revocação | honestos confirmados por rodada |", "|---|---|---|---|---|"]
for (s, w, rn), d in acc.items():
    lines.append(f"| {s} | {w} | {rn} | {np.mean(d['rec']):.2f} | {np.mean(d['fp']):.2f} |" if d["rec"] else f"| {s} | {w} | {rn} | — | {np.mean(d['fp']):.2f} |")
open("results/fase21/RESULTADO_offline2.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
