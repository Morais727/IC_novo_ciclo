#!/usr/bin/env python
"""Gera tabelas (.tex) e figuras (.png) da apresentação a partir dos resultados.

Rodar de novo sempre que novos resultados chegarem; a apresentação
(`../entregavel_03/apresentacao/apresentacao.tex`) só faz \\input/\\includegraphics destes arquivos.

    python experiments/make_slides_assets.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "entregavel_03" / "apresentacao"
TAB, FIG = OUT / "tabelas", OUT / "figuras"

ATTACK_ORDER = ["gaussian_noise", "ipm", "lie", "minmax", "minsum", "label_flip", "backdoor_pixel", "backdoor_pgd"]
ATTACK_LABEL = {"gaussian_noise": "Ruído gaussiano", "ipm": "IPM", "lie": "LIE", "minmax": "MinMax",
                "minsum": "MinSum", "label_flip": "Label flip", "backdoor_pixel": "Backdoor (pixel)",
                "backdoor_pgd": "Backdoor (PGD)"}

# paleta de referência (skill dataviz, modo claro)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2 = "#071f4a", "#fd9b1f"  # identidade visual H.IAAC (azul-marinho, laranja)


def ragnar_eval_rows() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "results" / "ragnar_eval").glob("*_s42.json")):
        r = json.loads(f.read_text())
        attack, arm = r["scenario"].split("__")
        # o braço "seen" antigo sorteava o bootstrap do pool inteiro (ataque podia não ser visto): descartado
        if arm == "ragnar_seen" and not r["defense_params"].get("bootstrap_only_eval_attack"):
            continue
        last = r["history"][-1]
        dets = [h["detection"] for h in r["history"] if h["detection"]]
        tp, fp, fn = (sum(d[k] for d in dets) for k in ("tp", "fp", "fn"))
        n_honest = (r["num_clients"] - r["num_malicious"]) * max(len(dets), 1)
        rows.append({"attack": attack, "arm": arm, "ma": last["main_accuracy"], "ba": last["backdoor_asr"],
                     "recall": tp / max(tp + fn, 1) if dets else np.nan,
                     "fpr": fp / n_honest if dets else np.nan})
    return pd.DataFrame(rows)


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.1f}" if pct else f"{v:.2f}"


def table_ragnar(df: pd.DataFrame):
    clean = df[(df.attack == "clean")]
    arms = [a for a in ["fedavg", "ragnar_unseen", "ragnar_seen"] if a in set(df.arm)]
    get = lambda a, arm, col: (df[(df.attack == a) & (df.arm == arm)][col].iloc[0]
                               if len(df[(df.attack == a) & (df.arm == arm)]) else np.nan)
    has_seen = "ragnar_seen" in arms
    head = r"Ataque & MA FedAvg & MA RAGNAR$_{nv}$" + (r" & MA RAGNAR$_{v}$" if has_seen else "") + \
           r" & BA FedAvg & BA RAGNAR$_{nv}$ & Recall$_{nv}$ & FP$_{nv}$ \\"
    lines = [r"\begin{tabular}{l" + "r" * (6 + has_seen) + "}", r"\toprule", head, r"\midrule"]
    for a in ATTACK_ORDER:
        if not len(df[df.attack == a]):
            continue
        cells = [ATTACK_LABEL[a], fmt(get(a, "fedavg", "ma")), fmt(get(a, "ragnar_unseen", "ma"))]
        if has_seen:
            cells.append(fmt(get(a, "ragnar_seen", "ma")))
        cells += [fmt(get(a, "fedavg", "ba")), fmt(get(a, "ragnar_unseen", "ba")),
                  fmt(get(a, "ragnar_unseen", "recall"), False), fmt(get(a, "ragnar_unseen", "fpr"), False)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(7 + has_seen) + r"}{l}{\footnotesize Sem ataque (FedAvg): MA = "
              + fmt(clean.ma.iloc[0] if len(clean) else np.nan) + r"\%} \\", r"\bottomrule", r"\end{tabular}"]
    (TAB / "ragnar_eval.tex").write_text("\n".join(lines) + "\n")


def figure_ragnar(df: pd.DataFrame):
    attacks = [a for a in ATTACK_ORDER if len(df[(df.attack == a) & (df.arm == "ragnar_unseen")])]
    if not attacks:
        return
    get = lambda a, arm, col: df[(df.attack == a) & (df.arm == arm)][col].iloc[0]
    y = np.arange(len(attacks))
    h = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(attacks) + 2.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    for off, arm, color, label in ((-h / 2, "fedavg", S1, "FedAvg (sem defesa)"),
                                   (h / 2, "ragnar_unseen", S2, "RAGNAR (ataque não visto)")):
        vals = [get(a, arm, "ma") for a in attacks]
        ax.barh(y + off, vals, height=h - 0.04, color=color, label=label)
        for yi, v in zip(y + off, vals):
            ax.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=7, color=INK_2)
    clean = df[df.attack == "clean"]
    if len(clean):
        ax.axvline(clean.ma.iloc[0], color=MUTED, lw=1, ls="--", label=f"sem ataque ({clean.ma.iloc[0]:.0f}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("MA final (%)", color=INK_2)
    ax.set_title("Acurácia do modelo global", loc="left", fontsize=10, color=INK)
    ax.set_yticks(y, [ATTACK_LABEL[a] for a in attacks], color=INK_2)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3)

    ax = axes[1]
    for off, col, color, label in ((-h / 2, "recall", S1, "Recall (atacantes pegos)"),
                                   (h / 2, "fpr", S2, "Falsos positivos (honestos excluídos)")):
        vals = [get(a, "ragnar_unseen", col) for a in attacks]
        ax.barh(y + off, vals, height=h - 0.04, color=color, label=label)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("taxa", color=INK_2)
    ax.set_title("Detecção do RAGNAR (ataque não visto)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ragnar_vs_fedavg.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def table_cross_phase():
    src = next((p for p in (ROOT / "results" / "signals_v2" / "cross_phase_classifier.csv",
                            ROOT / "results" / "signals" / "cross_phase_classifier.csv") if p.exists()), None)
    if src is None:
        return
    cp = pd.read_csv(src)
    cp = cp.rename(columns={"norm": "normalizacao"}) if "norm" in cp.columns else cp
    m = cp[cp.protocolo == "nao_visto"].groupby(["sinais", "normalizacao"], sort=False)[["auc", "recall", "fpr"]].mean()
    label = {"raw": "nenhuma", "robust_z": "z robusto", "rank": "rank"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Sinais & Normalização & AUC & Recall & Falsos pos. \\", r"\midrule"]
    prev = None
    for (s, n), r in m.iterrows():
        if prev is not None and s != prev:
            lines.append(r"\addlinespace")
        lines.append(f"{s if s != prev else ''} & {label.get(n, n)} & {r.auc:.2f} & {r.recall:.2f} & {r.fpr:.2f} \\\\")
        prev = s
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "cross_phase.tex").write_text("\n".join(lines) + "\n")
    (TAB / "cross_phase_fonte.tex").write_text(src.parent.name.replace("_", r"\_") + "\n")


def copy_heatmaps():
    for name in ("signals", "signals_v2"):
        src = ROOT / "results" / name / "auc_heatmap.png"
        if src.exists():
            shutil.copy(src, FIG / f"heatmap_{name}.png")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = ragnar_eval_rows()
    table_ragnar(df)
    figure_ragnar(df)
    table_cross_phase()
    copy_heatmaps()
    print(f"Tabelas e figuras atualizadas em {OUT}/ ({len(df)} execuções do ragnar_eval)")


if __name__ == "__main__":
    main()
