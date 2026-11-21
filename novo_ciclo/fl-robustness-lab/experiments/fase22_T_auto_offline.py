#!/usr/bin/env python
"""Fase 22 (19/11/2026) — teste SEM TREINO de regras que calculam o limiar de corroboração T sozinhas, a cada rodada,
sem conhecer o ataque (resposta à crítica "T é escolhido à mão por arquitetura").

Usa os escores por tensor (z_sim, z_norm) que o agregador-sombra (lg_shadow) gravou em todas as rodadas, nas
arquiteturas onde o T certo muda: ResNet18 (T=8 funciona; CIFAR-10 6 sementes, FMNIST, CIFAR-100), VGG19 (só T<=4
contém) e EfficientNet (só T=2 contém); e sem ataque (ResNet18, α 0,9 e 0,1) para medir honestos excluídos.

Contagem do cliente i na rodada: c_i = #tensores com max(z_sim,0)+max(z_norm,0) > 3 (igual à defesa).
Regras:
  fixa_T       T fixo (2, 4, 6, 8), para calibrar que revocação basta para conter em cada arquitetura;
  frac_q       T = ceil(q · nº de tensores sensíveis);
  zc_k         confirmado se c_i >= 2 e c_i > med(c) + k · max(1,4826·MAD(c), 1)  (z robusto da contagem, sem T);
  zc_k_T8      zc_k, mas nunca exige mais que 8 (teto: não fica mais frouxa que a regra atual da ResNet18).
Mede: revocação (fração dos atacantes confirmados, rodadas 80–200) e honestos confirmados por rodada."""
import glob, json, math, re, sys
import numpy as np

def counts(rec):
    names = [k for k in rec if isinstance(rec[k], dict) and "z_sim" in rec[k]]
    zs = np.array([rec[k]["z_sim"] for k in names]); zn = np.array([rec[k]["z_norm"] for k in names])
    return ((np.maximum(zs, 0) + np.maximum(zn, 0)) > 3).sum(0), len(names)

def zc_thr(c, k):
    med = np.median(c); s = max(1.4826 * np.median(np.abs(c - med)), 1.0)
    return max(2.0, med + k * s + 1e-9)          # c > thr  <=>  c >= floor(thr)+1

RULES = {}
for T in (2, 4, 6, 8):
    RULES[f"fixa_T{T}"] = (lambda T: lambda c, n: float(T))(T)
for q in (0.05, 0.10, 0.15):
    RULES[f"frac_{q:.2f}"] = (lambda q: lambda c, n: float(math.ceil(q * n)))(q)
for k in (2, 3, 4):
    RULES[f"zc_{k}"] = (lambda k: lambda c, n: math.floor(zc_thr(c, k)) + 1.0)(k)
    RULES[f"zc_{k}_T8"] = (lambda k: lambda c, n: min(8.0, math.floor(zc_thr(c, k)) + 1.0))(k)

# 2ª família (19/11, depois da 1ª rodada): confirmar os ⌈β·n⌉ clientes mais acusados (com c >= 2) — usa só a hipótese
# padrão de robustez (fração máxima de atacantes β, como Krum/média aparada/FLAME), não o T.
for b in (0.2, 0.25, 0.3, 0.4, 0.5):
    RULES[f"top_{b:.2f}"] = (lambda b: lambda c, n: float(max(2, np.sort(c)[-int(math.ceil(b * len(c)))])))(b)

SETS = {
    "ResNet18 · CIFAR-10 (T=8 contém)": "results/fase8e/raw/laysel_paper__lg_shadow__a0.9_s*.json",
    "ResNet18 · FMNIST": "results/fase8f/raw/laysel_resnet18gn_fmnist__lg_shadow__a0.9_s42.json",
    "ResNet18 · CIFAR-100": "results/fase13/raw/cifar100__lg_shadow__s4[23].json",
    "VGG19 · CIFAR-10 (só T<=4 contém)": "results/fase8f/raw/laysel_vgg19gn_cifar10__lg_shadow__a0.9_s42.json",
    "EfficientNet · CIFAR-10 (só T=2 contém)": "results/fase8f/raw/laysel_efficientnetgn_cifar10__lg_shadow__a0.9_s42.json",
    "ResNet18 · sem ataque α0,9": "results/fase11/raw/noattack__lg_shadow__a0.9_s*.json",
    "ResNet18 · sem ataque α0,1": "results/fase11/raw/noattack__lg_shadow__a0.1_s*.json",
}
res = {}
for sname, pat in SETS.items():
    fs = sorted(glob.glob(pat))
    agg = {r: {"rec": [], "fp": [], "Tr": []} for r in RULES}
    for f in fs:
        r = json.load(open(f)); mal = set(r.get("malicious_ids") or []); noatk = "noattack" in f
        print("   clientes maliciosos:", sorted(mal), flush=True)
        for t, rec in enumerate(r.get("diag_log") or [], 1):
            c, n = counts(rec)
            if n == 0: continue
            atk = sorted(mal) if (not noatk and 80 <= t <= 200) else []
            hon = [i for i in range(len(c)) if noatk or i not in mal]
            for rn, fn in RULES.items():
                Tr = fn(c, n); conf = c >= Tr
                if atk: agg[rn]["rec"].append(float(np.mean([conf[i] for i in atk])))
                if noatk or 80 <= t <= 200: agg[rn]["fp"].append(int(sum(conf[i] for i in hon)))
                agg[rn]["Tr"].append(Tr)
        print(sname, f.split("/")[-1], "ok", flush=True); del r
    res[sname] = {"n_arquivos": len(fs), **{rn: {"rec": float(np.mean(v["rec"])) if v["rec"] else None,
                   "fp": float(np.mean(v["fp"])) if v["fp"] else None,
                   "T_med": float(np.median(v["Tr"])) if v["Tr"] else None} for rn, v in agg.items()}}
json.dump(res, open("results/fase22/offline_T_auto.json", "w"), indent=1)

L = ["# Fase 22 — T automático, teste sem treino", "",
     "Revocação = fração dos atacantes confirmados (rodadas 80–200; maior é melhor). Honestos = honestos confirmados por rodada (menor é melhor). T mediano = limiar efetivo.", ""]
for sname, d in res.items():
    L += [f"## {sname} ({d['n_arquivos']} arquivo(s))", "", "| regra | revocação | honestos/rodada | T mediano |", "|---|---|---|---|"]
    for rn in RULES:
        x = d[rn]; fm = lambda v, p: "—" if v is None else format(v, p)
        L.append("| %s | %s | %s | %s |" % (rn, fm(x["rec"], ".2f"), fm(x["fp"], ".2f"), fm(x["T_med"], ".0f")))
    L.append("")
open("results/fase22/RESULTADO_offline.md", "w").write("\n".join(L)); print("\n".join(L))
