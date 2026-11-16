#!/usr/bin/env python
"""Painel de UM COMANDO para os experimentos da fila (10/11/2026): resume o que já caiu de
  (1) A3FL × {FMNIST, CIFAR} × {corr, client, corr-soft}  (results/fase8f/raw/laysel_a3fl_*.json)
  (2) adversário adaptativo ciente da corroboração — fase8g  (results/fase8g/raw/laysel_adapt_*.json)
  (3) evidência do limite de 1 atacante — Fase 9A  (results/fase9/evidencia/corr_m*.jsonl)
Robusto a arquivos faltando (diz "ainda não rodou"). Só leitura; não roda treino.
Uso: python experiments/analise_pendentes.py
"""
import json, sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))


def _final(p):
    p = Path(p)
    if not p.exists():
        return None
    h = json.load(open(p))["history"][-1]
    return h["backdoor_asr"], h["main_accuracy"]


def _a3fl():
    out = ["## 1. A3FL (gatilho adaptativo) × família — contenção?", ""]
    raw = LAB / "results/fase8f/raw"
    for ds, lbl in [("cifar10", "CIFAR-10"), ("fmnist", "FMNIST")]:
        g = _final(raw / f"laysel_a3fl_resnet18gn_{ds}__lg_shadow__a0.9_s42.json")
        out.append(f"### {lbl}")
        out.append("portão (sem defesa): " + ("ainda não rodou" if g is None
                   else f"ASR {g[0]:.1f}% / MA {g[1]:.1f}% → {'PEGOU' if g[0] >= 20 else 'não pegou'}"))
        out += ["", "| defesa | ASR | MA | contém? |", "|---|---|---|---|"]
        for d in ("layer_guard_corr", "layer_guard_client", "layer_guard_corr_soft"):
            r = _final(raw / f"laysel_a3fl_resnet18gn_{ds}__{d}__a0.9_s42.json")
            if r is None:
                out.append(f"| {d} | — | — | ainda não rodou |")
            else:
                out.append(f"| {d} | {r[0]:.1f}% | {r[1]:.1f}% | {'SIM' if r[0] < 20 else 'não'} |")
        out.append("")
    return out


def _adapt():
    out = ["## 2. Adversário adaptativo ciente da corroboração (fase8g, T=8)", ""]
    raw = LAB / "results/fase8g/raw"
    g = raw / "laysel_adapt_T8__lg_shadow__a0.9_s42.json"
    gf = _final(g)
    out.append("portão (ataque concentrado, sem defesa): " + ("ainda não rodou" if gf is None
               else f"ASR {gf[0]:.1f}% / MA {gf[1]:.1f}% → {'ainda pega' if gf[0] >= 20 else 'a própria evasão já contém'}"))
    if g.exists():
        ev = json.load(open(g)).get("attack_params", {}).get("_evade_corr_log", [])
        ev = [e for e in ev if e.get("m_kept") is not None]
        if ev:
            ms = [e["m_kept"] for e in ev]
            evd = sum(e["evades"] for e in ev)
            out.append(f"evasão: m mantido médio {sum(ms)/len(ms):.1f}/{ev[0]['n_sens']} sensíveis; "
                       f"evadiu em {evd}/{len(ev)} rodadas (T={ev[0]['T']}).")
    out += ["", "| defesa | ASR | MA | a corr cede? |", "|---|---|---|---|"]
    for d in ("layer_guard_corr", "layer_guard_client", "layer_guard_corr_soft"):
        r = _final(raw / f"laysel_adapt_T8__{d}__a0.9_s42.json")
        if r is None:
            out.append(f"| {d} | — | — | ainda não rodou |")
        else:
            out.append(f"| {d} | {r[0]:.1f}% | {r[1]:.1f}% | {'CEDE' if r[0] >= 20 else 'contém'} |")
    out.append("")
    return out


def _evid():
    out = ["## 3. Evidência do limite de 1 atacante (Fase 9A)", ""]
    d = LAB / "results/fase9/evidencia"
    files = sorted(d.glob("corr_m*.jsonl")) if d.exists() else []
    if not files:
        out += ["ainda não rodou (precisa do cv2 + GPU livre).", ""]
        return out
    out += ["Contagem de corroboração (nº de tensores que acusam) por round — atacante vs. benigno:", "",
            "| maliciosos | nº atacantes | rounds | atacante (média) | benigno (mediana) | rounds c/ atacante≥? |",
            "|---|---|---|---|---|---|"]
    import numpy as np
    for f in files:
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        cmal = np.array([np.mean(r["count_mal"]) for r in recs if r["count_mal"]])
        cben = np.array([r["count_ben_med"] for r in recs if r["count_ben_med"] is not None])
        benmax = np.array([r["count_ben_max"] for r in recs if r["count_ben_max"] is not None])
        # "destaca-se" = contagem do atacante acima do pior benigno do round
        above = sum(np.mean(r["count_mal"]) > (r["count_ben_max"] or 0) for r in recs if r["count_mal"])
        frac = f.stem.replace("corr_m", "").split("_")[0]
        out.append(f"| {frac} | {recs[0]['nmal']} | {len(recs)} | {cmal.mean():.1f} | {cben.mean():.1f} | {above}/{len(recs)} |")
    out += ["", "Leitura esperada: com 1 atacante a média dele ≈ benigno (não se destaca → corr não purga); "
                "com 2, destaca-se na maioria das rodadas (corroboram-se).", ""]
    return out


def main():
    lines = ["# Análise dos pendentes — A3FL, adaptativo e evidência (gerado por analise_pendentes.py)", ""]
    lines += _a3fl() + _adapt() + _evid()
    txt = "\n".join(lines)
    (LAB / "results/ANALISE_PENDENTES.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
