#!/usr/bin/env python
"""17/11/2026 — custo que a defesa impõe ao atacante adaptativo (sem treino novo). Lê results/fase8g/raw.
Métricas por agregação (sementes 42–47): ASR final, pico, ASR médio na janela de ataque (rod. 90–200), primeira
rodada com ASR >= 20% e >= 50%, e o que o atacante precisou ceder (rodadas em que evadiu, tensores mantidos, ruído)."""
import csv, glob, json, re, statistics as st
from pathlib import Path

R = Path("results/fase8g/raw"); ART = Path("../artigo/dados")
DEFS = ["lg_shadow", "layer_guard_corr", "layer_guard_corr_soft", "layer_guard_client", "layer_guard_corr_soft_clip", "layer_guard_corr_soft_tm"]
br = lambda x, n=1: "—" if x is None else f"{x:.{n}f}".replace(".", ",")


def first(h, thr):
    for x in h:
        if 80 <= x["round"] <= 205 and (x["backdoor_asr"] or 0) >= thr:
            return x["round"]
    return None


rows = []
for d in DEFS:
    per = []
    for f in sorted(glob.glob(str(R / f"laysel_adapt_T8__{d}__a0.9_s*.json"))):
        s = int(re.search(r"_s(\d+)\.json", f).group(1)); r = json.load(open(f)); h = r["history"]
        win = [x["backdoor_asr"] for x in h if 90 <= x["round"] <= 200]
        lg = r.get("attack_params", {}).get("_evade_corr_log", [])
        ev = [e for e in lg if e.get("evades")]
        per.append({"seed": s, "final": h[-1]["backdoor_asr"], "peak": max(win), "mean_win": st.mean(win), "ma": h[-1]["main_accuracy"],
                    "r20": first(h, 20), "r50": first(h, 50), "evade_frac": (len(ev) / len(lg)) if lg else None,
                    "m_evade": st.mean([e["m_kept"] for e in ev]) if ev else None, "n_sens": st.mean([e["n_sens"] for e in lg]) if lg else None,
                    "nu_evade": st.mean([e["nu"] for e in ev]) if ev else None,
                    # fração de pares (atacante, rodada) com contagem < T: o atacante que passa sem ser confirmado
                    "pass_frac": st.mean([st.mean([c < e["T"] for c in e["confirm_count_mal"]]) for e in lg]) if lg else None})
    if not per:
        continue
    agg = lambda k: [p[k] for p in per if p[k] is not None]
    mean = lambda k: st.mean(agg(k)) if agg(k) else None
    rows.append({"agregacao": d, "n": len(per), "final": mean("final"), "pico": mean("peak"), "media_janela": mean("mean_win"),
                 "acuracia": mean("ma"), "rod20_media": mean("r20"), "nao_chega_20": sum(p["r20"] is None for p in per),
                 "rod50_media": mean("r50"), "nao_chega_50": sum(p["r50"] is None for p in per),
                 "evade_frac": mean("evade_frac"), "m_evade": mean("m_evade"), "n_sens": mean("n_sens"), "nu_evade": mean("nu_evade"), "passa_frac": mean("pass_frac"),
                 "sementes": " ".join(str(p["seed"]) for p in per)})
with open(ART / "custo_adaptativo.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
md = ["| agregação | n | ASR final | pico | ASR médio no ataque | 1ª rodada ≥ 20% | 1ª rodada ≥ 50% | rodadas em que evadiu | atacante-rodada abaixo de T | tensores mantidos ao evadir | ruído ν | acurácia |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['agregacao']} | {r['n']} | {br(r['final'])} | {br(r['pico'])} | {br(r['media_janela'])} | {br(r['rod20_media'],0)} ({r['nao_chega_20']} não chega) | "
              f"{br(r['rod50_media'],0)} ({r['nao_chega_50']} não chega) | {br(100*r['evade_frac'] if r['evade_frac'] is not None else None,0)}% | "
              f"{br(100*r['passa_frac'],0)}% | {br(r['m_evade'],0)} de {br(r['n_sens'],0)} | {br(r['nu_evade'],2)} | {br(r['acuracia'])} |")
Path("results/fase8g/CUSTO_ADAPTATIVO.md").write_text("# Custo imposto ao atacante adaptativo (fase 8G/16, sem treino novo)\n\n" + "\n".join(md) + "\n")
print("\n".join(md))
