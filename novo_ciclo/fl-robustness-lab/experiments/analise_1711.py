#!/usr/bin/env python
"""Análises de 17/11/2026 (sem GPU): sensibilidade a T (corr, s42–44), falsos positivos sem ataque (fase 11) e
CIFAR-100 (fase 13, s42–43). Escreve RESULTADO*.md no lab e os .csv do artigo (../artigo/dados)."""
import csv, glob, json, os, statistics as st
from pathlib import Path

R = Path("results"); ART = Path("../artigo/dados")


def load(f):
    h = json.load(open(f))["history"]
    att = [x for x in h if 80 <= x["round"] <= 200 and x.get("backdoor_asr") is not None]
    return {"asr": h[-1]["backdoor_asr"] or 0.0, "ma": h[-1]["main_accuracy"],
            "peak": max((x["backdoor_asr"] for x in att), default=None),
            "fp": [x["detection"]["fp"] for x in h if x.get("detection")]}


def m(v): return st.mean(v)
def sd(v): return st.stdev(v) if len(v) > 1 else 0.0
def br(x, n=1): return f"{x:.{n}f}".replace(".", ",")


# --- 1) sensibilidade a T
rows, md = [], ["| T | ASR final (s42 · s43 · s44) | média | contém (≤ 20%) | pico médio | acurácia média |", "|---|---|---|---|---|---|"]
ref = [load(f"{R}/fase8e/raw/laysel_paper__lg_shadow__a0.9_s{s}.json") for s in (42, 43, 44)]
for T, tag in [(4, "layer_guard_corr_T4"), (8, "layer_guard_corr"), (11, "layer_guard_corr_T11"), (14, "layer_guard_corr_T14")]:
    d = [load(f"{R}/fase8e/raw/laysel_paper__{tag}__a0.9_s{s}.json") for s in (42, 43, 44)]
    a = [x["asr"] for x in d]
    rows.append({"T": T, "asr_s42": a[0], "asr_s43": a[1], "asr_s44": a[2], "asr_media": m(a), "asr_dp": sd(a),
                 "contem": sum(x <= 20 for x in a), "pico_medio": m([x["peak"] for x in d]), "ma_media": m([x["ma"] for x in d])})
    md.append(f"| {T} | {' · '.join(br(x) for x in a)} | {br(m(a))} ± {br(sd(a))} | {sum(x <= 20 for x in a)}/3 | {br(m([x['peak'] for x in d]))} | {br(m([x['ma'] for x in d]))} |")
md.append(f"| sem defesa | {' · '.join(br(x['asr']) for x in ref)} | {br(m([x['asr'] for x in ref]))} | 0/3 | {br(m([x['peak'] for x in ref]))} | {br(m([x['ma'] for x in ref]))} |")
with open(ART / "sens_T.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(R / "fase8e" / "TSWEEP.md").write_text("# Sensibilidade a T (corr, cenário fiel, s42–44)\n\n" + "\n".join(md) + "\n")

# --- 2) falsos positivos sem ataque
fp_rows, md2 = [], ["| α | semente | agregação | acurácia | Δ vs sem defesa (pp) | benignos descartados por avaliação (média · máx.) | ASR do gatilho |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{R}/fase11/raw/noattack__*.json")):
    nm = os.path.basename(f)[len("noattack__"):-5]; dfn, rest = nm.rsplit("__a", 1); al, s = rest.split("_s")
    x = load(f); b = f"{R}/fase11/raw/noattack__lg_shadow__a{al}_s{s}.json"
    base = load(b)["ma"] if os.path.exists(b) else None
    dl = (x["ma"] - base) if base is not None else None
    fp_rows.append({"alpha": al, "seed": s, "agregacao": dfn, "ma": x["ma"], "delta_pp": dl, "fp_medio": m(x["fp"]) if x["fp"] else 0,
                    "fp_max": max(x["fp"]) if x["fp"] else 0, "asr_gatilho": x["asr"]})
    md2.append(f"| {al} | {s} | {dfn} | {br(x['ma'])} | {br(dl) if dl is not None else 'sem referência'} | {br(m(x['fp']) if x['fp'] else 0)} · {max(x['fp']) if x['fp'] else 0} | {br(x['asr'])} |")
with open(ART / "fp_sem_ataque.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fp_rows[0])); w.writeheader(); w.writerows(fp_rows)
(R / "fase11" / "RESULTADO.md").write_text("# Falsos positivos sem ataque (fase 11; 20 clientes, todos honestos)\n\n" + "\n".join(md2) + "\n")

# --- 3) CIFAR-100
c_rows, md3 = [], ["| agregação | ASR final (s42 · s43) | pico (s42 · s43) | acurácia (s42 · s43) | contém |", "|---|---|---|---|---|"]
for dfn in ["lg_shadow", "layer_guard_corr", "layer_guard_corr_soft", "layer_guard_client", "lasa", "fedsurrogate", "flame"]:
    d = [load(f"{R}/fase13/raw/cifar100__{dfn}__s{s}.json") for s in (42, 43)]
    c_rows.append({"agregacao": dfn, **{f"asr_s{s}": x["asr"] for s, x in zip((42, 43), d)}, **{f"pico_s{s}": x["peak"] for s, x in zip((42, 43), d)},
                   **{f"ma_s{s}": x["ma"] for s, x in zip((42, 43), d)}})
    md3.append(f"| {dfn} | {' · '.join(br(x['asr']) for x in d)} | {' · '.join(br(x['peak']) for x in d)} | {' · '.join(br(x['ma']) for x in d)} | {sum(x['asr'] <= 20 for x in d)}/2 |")
with open(ART / "cifar100.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(c_rows[0])); w.writeheader(); w.writerows(c_rows)
(R / "fase13" / "RESULTADO_cifar100.md").write_text("# CIFAR-100 (ResNet18-GN, cenário fiel; s42–43)\n\nModelo limpo (fase 18, s42): acurácia 44,6%, ASR do gatilho 0,1%.\n\n" + "\n".join(md3) + "\n")
print("\n".join(md), "\n\n" + "\n".join(md2), "\n\n" + "\n".join(md3))
