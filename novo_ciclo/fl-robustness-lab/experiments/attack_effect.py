#!/usr/bin/env python
"""Efeito de cada ataque sem defesa (FedAvg, 40 rounds, semente 42): o ataque é "funcional"?

Compara com a linha de base sem atacantes, do mesmo protocolo:
  MA   acurácia final vs. clean__fedavg
  ASR  sucesso do ataque direcionado vs. o que um modelo limpo já prevê como alvo
       (asr_baseline__trigger / __edge, de experiments/asr_baseline.py)

    python experiments/attack_effect.py      -> results/attack_effect/effect.{md,tex,csv}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from experiments.ragnar_eval import ATTACKS, RECENT_ATTACKS
from src.attacks.registry import ATTACK_REGISTRY

RES = Path("results/ragnar_eval")
LABEL = {"gaussian_noise": "Ruído gaussiano", "ipm": "IPM", "lie": "LIE", "minmax": "MinMax", "minsum": "MinSum",
         "label_flip": "Label flip", "backdoor_pixel": "Backdoor (pixel)", "backdoor_pgd": "Backdoor (PGD)",
         "mimic": "Mimic", "lf": "LF", "dba": "DBA", "lp": "LP", "edge_case_pgd": "Edge-case PGD"}


def last(path: Path, key: str):
    if not path.exists():
        return None
    return json.loads(path.read_text())["history"][-1][key]


def main():
    clean_ma = last(RES / "clean__fedavg_s42.json", "main_accuracy")
    base = {k: last(RES / f"asr_baseline__{k}_s42.json", "backdoor_asr") for k in ("trigger", "edge")}
    rows = []
    for a in ATTACKS:
        f = RES / f"{a}__fedavg_s42.json"
        if not f.exists():
            continue
        ev = ATTACK_REGISTRY[a].get("eval")
        asr, asr0 = last(f, "backdoor_asr"), base.get(ev) if ev else None
        rows.append({"ataque": a, "novo": a in RECENT_ATTACKS, "maliciosos": ATTACKS[a]["num_malicious"],
                     "ma": last(f, "main_accuracy"), "delta_ma": last(f, "main_accuracy") - clean_ma,
                     "asr": asr, "asr_base": asr0})
    df = pd.DataFrame(rows)
    out = Path("results/attack_effect")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "effect.csv", index=False)

    f1 = lambda v: "--" if v is None or pd.isna(v) else f"{v:.1f}"
    md = [f"Sem ataque: MA = {clean_ma:.1f}%. ASR de base (modelo limpo): gatilho {f1(base['trigger'])}%, "
          f"edge-case {f1(base['edge'])}%.", "",
          "| Ataque | Maliciosos | MA (%) | ΔMA (p.p.) | ASR (%) | ASR base (%) |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcrrrr}", r"\toprule",
           r"Ataque & Maliciosos & MA (\%) & $\Delta$MA (p.p.) & ASR (\%) & ASR base (\%) \\", r"\midrule"]
    for new in (False, True):
        part = df[df.novo == new]
        if new and len(part):
            tex.append(r"\midrule")
        for _, r in part.iterrows():
            cells = [LABEL[r.ataque] + (" *" if new else ""), str(r.maliciosos), f1(r.ma), f"{r.delta_ma:+.1f}",
                     f1(r.asr), f1(r.asr_base)]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells).replace("*", r"$^\ast$") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "effect.md").write_text("\n".join(md) + "\n")
    (out / "effect.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
