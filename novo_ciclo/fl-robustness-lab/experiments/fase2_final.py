#!/usr/bin/env python
"""Avaliação final da Fase 2 (critérios F1/F2 do adendo, experiments/fase2_lab.py). Roda o NC-U (fase2_nc.py) nos
modelos que faltam e monta results/fase2/FINAL.md. ASR/acurácia: modelo do round 40 (teste inteiro); para as
defesas sem modelo salvo em fases anteriores, média dos rounds 35/40 do registro.
Uso: python experiments/fase2_final.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.fase2_analyze import metrics

OUT = Path("results/fase2")
OLD = [Path("results/final_lab/raw"), Path("results/layer_guard_sn/raw"), Path("results/layer_guard/raw")]
TEST, DEV = (46, 47, 48), (42, 44, 45)


def raw(attack, defense, seed):
    f = OUT / "raw" / f"{attack}__{defense}__a0.1_s{seed}.json"
    if f.exists():
        return metrics(json.load(open(f)))
    for d in OLD:   # semente 42 das fases anteriores (sem sufixo) e réplicas _sNN
        for g in (d / f"{attack}__{defense}__a0.1_s{seed}.json", d / f"{attack}__{defense}__a0.1.json" if seed == 42 else None):
            if g and g.exists():
                return metrics(json.load(open(g)))
    return None


def nc(attack, defense, seed):
    c = json.load(open(OUT / "nc.json")) if (OUT / "nc.json").exists() else {}
    return c.get(f"{attack}__{defense}__a0.1_s{seed}")


def f(v, acc=False):
    return "—" if v is None else f"{v:.1f}%"


def main():
    subprocess.run([sys.executable, "experiments/fase2_nc.py"], check=False, capture_output=True)
    md = ["# Fase 2 — avaliação final (critérios fixados antes; adendos em experiments/fase2_lab.py)", "",
          "Defesa final: **LayerGuard v2 no treino + NC-U no modelo final** (LG+NC-U). ASR (acurácia).", ""]

    # 1. LP + ruído nas seis sementes
    md += ["## LP evasivo + ruído ν = 1, α = 0,1, por semente", "",
           "| semente | FedAvg | Multi-Krum | FLAME | FedSurrogate | LayerGuard v2 | FedAvg + NC-U | LG + NC-U |", "|---|---|---|---|---|---|---|---|"]
    for s in DEV + TEST:
        cells = []
        for d in ("fedavg", "multi_krum", "flame", "fedsurrogate", "layer_guard_sn"):
            m = raw("lp_full_indiv_nu1.0", d, s)
            cells.append("—" if m is None else f"{m['asr']:.1f}% ({m['acc']:.0f}%)")
        for d in ("fedavg", "layer_guard_sn"):
            r = nc("lp_full_indiv_nu1.0", d, s)
            cells.append("—" if r is None else f"{r['NC-U']['asr']:.1f}% ({r['NC-U']['acc']:.0f}%)")
        md.append(f"| {s}{' (teste)' if s in TEST else ''} | " + " | ".join(cells) + " |")

    # 2. Critérios nas sementes de teste
    md += ["", "## Critérios F1/F2 (sementes de teste 46, 47, 48)", "",
           "| semente | ataque | ASR FedAvg | ASR LG v2 | ASR LG+NC-U | ASR FedAvg+NC-U | F1 (LG+NC-U <= 10%) |", "|---|---|---|---|---|---|---|"]
    f1, f2 = [], []
    for s in TEST:
        for a in ("lp_full_indiv_nu1.0", "backdoor_pixel", "dba"):
            fa, lg = raw(a, "fedavg", s), raw(a, "layer_guard_sn", s)
            rl, rf = nc(a, "layer_guard_sn", s), nc(a, "fedavg", s)
            if fa is None or rl is None:
                md.append(f"| {s} | {a} | {f(fa and fa['asr'])} | {f(lg and lg['asr'])} | pendente | | |")
                continue
            applies = fa["asr"] >= 40
            ok = rl["NC-U"]["asr"] <= 10
            if applies:
                f1.append(ok)
            md.append(f"| {s} | {a} | {fa['asr']:.1f}% | {lg['asr']:.1f}% | {rl['NC-U']['asr']:.1f}% "
                      f"(alvo {rl['det']['target']}) | {rf['NC-U']['asr'] if rf else float('nan'):.1f}% | "
                      f"{('OK' if ok else 'FALHA') if applies else 'n/a (ataque < 40% no FedAvg)'} |")
        fa, rl = raw("none", "fedavg", s), nc("none", "layer_guard_sn", s)
        if fa and rl:
            ok = rl["NC-U"]["acc"] >= fa["acc"] - 2
            f2.append(ok)
            md.append(f"| {s} | sem ataque (acurácia) | {fa['acc']:.1f}% | {raw('none', 'layer_guard_sn', s)['acc']:.1f}% | "
                      f"{rl['NC-U']['acc']:.1f}% (alvo {rl['det']['target']}) | | F2 {'OK' if ok else 'FALHA'} |")
    md += ["", f"**F1:** {sum(f1)} de {len(f1)} casos aplicáveis com ASR <= 10% → **{'OK' if f1 and all(f1) else 'FALHA'}**"
           f"{' (incompleto)' if len(f1) < 9 else ''}",
           f"**F2:** {sum(f2)} de {len(f2)} sementes → **{'OK' if f2 and all(f2) else 'FALHA'}**{' (incompleto)' if len(f2) < 3 else ''}"]

    # 3. NC-U nos modelos de desenvolvimento (semente 42)
    md += ["", "## NC-U sobre o FedAvg, semente 42 (desenvolvimento)", "", "| ataque | sem reparo | NC-U | alvo (índice) |", "|---|---|---|---|"]
    for a in ("none", "lp_full", "lp_full_indiv", "lp_full_indiv_nu1.0", "lp_full_indiv_nu1.0k", "backdoor_pixel", "backdoor_pgd", "dba"):
        r = nc(a, "fedavg", 42)
        if r:
            cell = lambda v: f"{v['acc']:.1f}%" if v["asr"] is None else f"{v['asr']:.1f}% ({v['acc']:.0f}%)"
            md.append(f"| {a} | {cell(r['sem reparo'])} | {cell(r['NC-U'])} | {r['det']['target']} ({max(r['det']['ai']):.1f}) |")
    (OUT / "FINAL.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
