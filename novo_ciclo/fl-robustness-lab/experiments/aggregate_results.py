#!/usr/bin/env python
"""Consolida os JSONs de results/ em tabelas e figuras para o relatório.

Para cada execução extrai: MA final, MA máxima, BA final (backdoor) e
precisão/revocação/F1 médios de detecção ao longo das avaliações. Agrupa por
cenário (média ± desvio-padrão entre sementes).

Saídas em results/summary/:
  runs.csv          uma linha por execução (cenário x semente)
  summary.csv       uma linha por cenário (média/desvio entre sementes)
  summary.md        mesma tabela em Markdown
  summary.tex       mesma tabela em LaTeX (booktabs), pronta para \\input
  final_ma.png      MA final por cenário
  backdoor_ba.png   BA final nos cenários de backdoor
  detection_f1.png  F1 de detecção por cenário (defesas que excluem clientes)
  curves.png        MA (e BA) por round, um painel por cenário

Uso:
    python experiments/aggregate_results.py                 # ignora execuções --quick
    python experiments/aggregate_results.py --include-quick
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta de referência (skill dataviz, modo claro)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_MA = "#2a78d6"   # slot 1 (azul)
SERIES_BA = "#eb6834"   # slot 2 (laranja)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def load_runs(results_dir: Path, include_quick: bool) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "history" not in data or not data["history"]:
            continue
        if data.get("quick", path.stem.endswith("_quick")) and not include_quick:
            continue
        data["_file"] = path.name
        runs.append(data)
    return runs


def summarize_run(run: dict) -> dict:
    hist = run["history"]
    last = hist[-1]
    dets = [h["detection"] for h in hist if h.get("detection")]
    row = {
        "scenario": run["scenario"],
        "attack": run["attack"],
        "defense": run["defense"],
        "alpha": run.get("dirichlet_alpha"),
        "seed": run.get("seed"),
        "file": run["_file"],
        "final_ma": last["main_accuracy"],
        "max_ma": max(h["main_accuracy"] for h in hist),
        "final_ba": last.get("backdoor_asr"),
        "det_precision": np.mean([d["precision"] for d in dets]) if dets else np.nan,
        "det_recall": np.mean([d["recall"] for d in dets]) if dets else np.nan,
        "det_f1": np.mean([d["f1"] for d in dets]) if dets else np.nan,
        "sec_per_round": run.get("sec_per_round", np.nan),
    }
    return row


def summarize_scenarios(runs_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["final_ma", "max_ma", "final_ba", "det_precision", "det_recall", "det_f1", "sec_per_round"]
    grouped = runs_df.groupby(["scenario", "attack", "defense", "alpha"], dropna=False)
    out = grouped[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = grouped.size()
    return out.reset_index().sort_values(["attack", "defense"])


def _fmt(mean, std, pct=True):
    if pd.isna(mean):
        return "—"
    s = f"{mean:.1f}" if pct else f"{mean:.2f}"
    if not pd.isna(std):
        s += f" ± {std:.1f}" if pct else f" ± {std:.2f}"
    return s


def display_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Cenário": summary["scenario"],
        "Ataque": summary["attack"],
        "Defesa": summary["defense"],
        "α": summary["alpha"],
        "MA final (%)": [_fmt(m, s) for m, s in zip(summary["final_ma_mean"], summary["final_ma_std"])],
        "BA final (%)": [_fmt(m, s) for m, s in zip(summary["final_ba_mean"], summary["final_ba_std"])],
        "F1 detecção": [_fmt(m, s, pct=False) for m, s in zip(summary["det_f1_mean"], summary["det_f1_std"])],
        "Sementes": summary["n_seeds"],
    })


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row.values) + " |" for _, row in table.iterrows()]
    return "\n".join(lines) + "\n"


def to_latex(table: pd.DataFrame) -> str:
    esc = lambda v: str(v).replace("_", r"\_").replace("±", r"$\pm$").replace("α", r"$\alpha$").replace("—", "--")
    cols = list(table.columns)
    lines = [r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in table.iterrows():
        lines.append(" & ".join(esc(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def barh(ax, labels, means, stds, color, xlabel, xlim=None, fmt="{:.1f}"):
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=np.nan_to_num(stds), color=color, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    for yi, m in zip(y, means):
        ax.text(m, yi, "  " + fmt.format(m), va="center", ha="left", color=INK_2, fontsize=8)


def plot_bars(summary: pd.DataFrame, out_dir: Path):
    s = summary.sort_values("final_ma_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    barh(ax, s["scenario"], s["final_ma_mean"], s["final_ma_std"], SERIES_MA, "MA final (%)", (0, 100))
    ax.set_title("Acurácia principal final por cenário (média ± desvio entre sementes)", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "final_ma.png", dpi=200)
    plt.close(fig)

    b = summary.dropna(subset=["final_ba_mean"]).sort_values("final_ba_mean", ascending=False)
    if len(b):
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(b) + 1.2))
        barh(ax, b["scenario"], b["final_ba_mean"], b["final_ba_std"], SERIES_BA, "BA final (%) — menor é melhor", (0, 100))
        ax.set_title("Taxa de sucesso do backdoor por cenário", loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / "backdoor_ba.png", dpi=200)
        plt.close(fig)

    d = summary.dropna(subset=["det_f1_mean"]).sort_values("det_f1_mean", ascending=False)
    if len(d):
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(d) + 1.2))
        barh(ax, d["scenario"], d["det_f1_mean"], d["det_f1_std"], SERIES_MA, "F1 médio de detecção", (0, 1), "{:.2f}")
        ax.set_title("Detecção de clientes maliciosos (F1 médio nas avaliações)", loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / "detection_f1.png", dpi=200)
        plt.close(fig)


def plot_curves(runs: list[dict], out_dir: Path):
    by_scenario: dict[str, list[dict]] = {}
    for r in runs:
        by_scenario.setdefault(r["scenario"], []).append(r)
    names = sorted(by_scenario)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.4 * nrows), sharey=True, squeeze=False)

    for ax, name in zip(axes.flat, names):
        group = by_scenario[name]
        rounds = [h["round"] for h in group[0]["history"]]
        for key, color, label in (("main_accuracy", SERIES_MA, "MA"), ("backdoor_asr", SERIES_BA, "BA")):
            vals = [[h.get(key) for h in r["history"]] for r in group if len(r["history"]) == len(rounds)]
            if not vals or vals[0][0] is None:
                continue
            arr = np.array(vals, dtype=float)
            mean = arr.mean(axis=0)
            ax.plot(rounds, mean, color=color, lw=2, marker="o", ms=3, label=label)
            if len(arr) > 1:
                std = arr.std(axis=0)
                ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_ylim(0, 100)
        ax.set_xlabel("round", fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("%")

    handles = [plt.Line2D([], [], color=SERIES_MA, lw=2, marker="o", ms=3, label="MA — acurácia principal"),
               plt.Line2D([], [], color=SERIES_BA, lw=2, marker="o", ms=3, label="BA — sucesso do backdoor")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Evolução por round (média entre sementes; faixa = ±1 desvio)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "curves.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None, help="default: <results-dir>/summary")
    parser.add_argument("--include-quick", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "summary"
    runs = load_runs(results_dir, args.include_quick)
    if not runs:
        print(f"Nenhum resultado em {results_dir} (use --include-quick para incluir execuções rápidas).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame([summarize_run(r) for r in runs])
    runs_df.to_csv(out_dir / "runs.csv", index=False)
    summary = summarize_scenarios(runs_df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    table = display_table(summary)
    (out_dir / "summary.md").write_text(to_markdown(table))
    (out_dir / "summary.tex").write_text(to_latex(table))

    plot_bars(summary, out_dir)
    plot_curves(runs, out_dir)

    print(table.to_string(index=False))
    print(f"\n{len(runs)} execuções, {len(summary)} cenários -> {out_dir}/")


if __name__ == "__main__":
    main()
