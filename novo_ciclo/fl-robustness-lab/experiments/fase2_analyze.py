"""Resumo da Fase 2 (experiments/fase2_lab.py): tabela de ASR, acurácia e aceitação por treino."""
import json
from pathlib import Path

import numpy as np

OUT = Path("results/fase2")


def metrics(d):
    h = d["history"][-2:]
    a = d["acceptance"]
    return {"acc": np.mean([e["main_accuracy"] for e in h]),
            "asr": np.mean([e["backdoor_asr"] for e in h]) if h[-1]["backdoor_asr"] is not None else None,
            "hon": a["benign"], "mal": a["malicious"], "hon_par": a.get("benign_params"), "mal_par": a.get("malicious_params")}


def summarize():
    md = ["# Fase 2 — resumo automático", "", "| treino | acurácia | ASR | honestos aceitos | maliciosos aceitos | params mal. aceitos |",
          "|---|---|---|---|---|---|"]
    for f in sorted((OUT / "raw").glob("*.json")):
        m = metrics(json.load(open(f)))
        p = lambda v: "—" if v is None else f"{v:.0%}"
        md.append(f"| {f.stem} | {m['acc']:.1f}% | {'—' if m['asr'] is None else f'{m[chr(97)+chr(115)+chr(114)]:.1f}%'} | "
                  f"{p(m['hon'])} | {p(m['mal'])} | {p(m['mal_par'])} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)
