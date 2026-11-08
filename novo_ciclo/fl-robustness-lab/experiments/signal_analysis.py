#!/usr/bin/env python
"""Análise de discriminabilidade dos sinais observáveis (entregável do mês 3).

Para cada ataque, simula `--rounds` rounds federados e registra, para cada
cliente em cada round, todos os sinais de `src/signals.py` e o rótulo
(malicioso/honesto). Depois mede, para cada sinal x ataque, o quanto o sinal
separa maliciosos de honestos: AUC-ROC calculada **dentro de cada round**
(o que o servidor precisa decidir) e depois média entre rounds.

Agregação oráculo: o modelo global é atualizado só com os deltas honestos, para
que todos os ataques sejam observados sobre a mesma trajetória realista de
treino — mede-se a detectabilidade do ataque, não o estrago que ele causaria.

Saídas em results/signals/:
  raw/<ataque>.csv     sinais por (round, cliente) — retomável: ataques já coletados são pulados
  auc_by_round.csv     AUC por sinal x ataque x round (bruta, na direção declarada e bilateral)
  auc_summary.csv      médias por sinal x ataque
  auc_table.md / .tex  AUC na direção declarada a priori (src/signals.py::DIRECTION); < 0.5 = o
                       ataque inverte o sinal. Não usa max(AUC, 1-AUC), que escolhe a direção por
                       ataque depois de ver o resultado e superestima o que um detector consegue
  auc_bilateral.md/.tex AUC do escore bilateral |z robusto| (desvio da mediana do round, sem direção)
  auc_heatmap.png      mapa sinais x ataques (direção declarada)
  cross_phase_*.csv/md classificador estilo RAGNAR (XGBoost, limiar 0.5) treinado nos rounds
                       iniciais e testado nos finais, por conjunto de sinais x normalização,
                       com o ataque visto ou não visto no treino

Uso (CPU, sem disputar a GPU):
    CUDA_VISIBLE_DEVICES="" python experiments/signal_analysis.py --threads 3
    python experiments/signal_analysis.py --analyze-only     # só recalcula tabelas/figuras
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.ragnar_eval import ATTACKS, COMMON
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig
from src.federated.simulate import _build_data, _train_round
from src.federated.utils import flatten
from src.models.cnn import build_model
from src.federated.simulate import attack_family
from src.signals import DIRECTION, ROBUST_SIGNALS, SIGNAL_FAMILIES, SIGNALS, SignalExtractor

RAGNAR_FEATURES = ["l1_mean", "l2_mean", "l3_mean", "cos_mean"]
FEATURE_SETS = {
    "RAGNAR-4": RAGNAR_FEATURES,
    "ampliado-10": RAGNAR_FEATURES + ["cos_median", "fc1_cos_median", "hist_cos_self", "holdout_loss_gain",
                                      "mahalanobis", "cos_global"],
    "robusto": ROBUST_SIGNALS + ["fc1_cos_median", "hist_cos_self", "holdout_loss_gain"],
    "todos": SIGNALS,
}
NORMALIZATIONS = ["raw", "robust_z", "rank"]


def collect(attack: str, rounds: int, seed: int, device: torch.device, quick: bool = False) -> pd.DataFrame:
    spec = ATTACKS[attack]
    cfg = ScenarioConfig(name=f"signals_{attack}", attack=attack, num_malicious=spec["num_malicious"],
                         attack_params=dict(spec["attack_params"]), defense="fedavg", seed=seed,
                         device=str(device), quick=quick, **{**COMMON, "rounds": rounds})
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_ctor = lambda: build_model(cfg.model)
    client_data, server_holdout, _, _ = _build_data(cfg, rng)
    malicious_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())
    extractor = SignalExtractor(model_ctor, server_holdout, device, num_malicious_assumed=cfg.num_malicious,
                                batch_size=cfg.batch_size, lr=cfg.lr)

    global_vector = flatten(model_ctor()).cpu()
    benign = [c for c in range(cfg.num_clients) if c not in malicious_ids]
    rows = []
    t0 = time.time()
    for r in range(1, rounds + 1):
        deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg,
                                       malicious_ids, ATTACK_REGISTRY[attack], cfg.attack_params)
        sig = extractor.compute(deltas, global_vector)
        for c in range(cfg.num_clients):
            rows.append({"attack": attack, "round": r, "client": c, "malicious": int(c in malicious_ids),
                         **{k: sig[k][c] for k in SIGNALS}})
        w = weights[benign].float()
        global_vector = global_vector + (deltas[benign] * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        print(f"  [{attack}] round {r}/{rounds} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def auc_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """Por round: AUC bruta, AUC na direção declarada (DIRECTION) e AUC do escore bilateral |z robusto|."""
    out = []
    signals = [s for s in SIGNALS if s in df.columns]
    for (attack, r), g in df.groupby(["attack", "round"]):
        y = g["malicious"].values
        for s in signals:
            v = g[s].values
            ok = ~np.isnan(v)
            if ok.sum() < 2 or len(set(y[ok])) < 2:
                continue
            v, yy = v[ok], y[ok]
            med = np.median(v)
            z_abs = np.abs(v - med) / (1.4826 * np.median(np.abs(v - med)) + 1e-12)
            auc = roc_auc_score(yy, v)
            out.append({"attack": attack, "round": r, "signal": s, "auc": auc,
                        "auc_dir": auc if DIRECTION[s] > 0 else 1 - auc,
                        "auc_bilateral": roc_auc_score(yy, z_abs)})
    return pd.DataFrame(out)


def summarize(by_round: pd.DataFrame) -> pd.DataFrame:
    g = by_round.groupby(["signal", "attack"])
    summ = g["auc"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "auc_mean", "count": "rounds"})
    summ["auc_dir"] = g["auc_dir"].mean().values
    summ["auc_dir_min_round"] = g["auc_dir"].min().values
    summ["auc_bilateral"] = g["auc_bilateral"].mean().values
    # separabilidade pós-hoc, mantida só para referência (NÃO usar como evidência de detecção)
    summ["separability_posthoc"] = np.maximum(summ["auc_mean"], 1 - summ["auc_mean"])
    summ["direction_declared"] = summ["signal"].map(lambda s: "↑" if DIRECTION[s] > 0 else "↓")
    summ["family"] = summ["signal"].map(SIGNAL_FAMILIES)
    return summ


def normalize(df: pd.DataFrame, cols: list[str], how: str) -> pd.DataFrame:
    """Normalização dentro de cada (ataque, round): nenhuma, z robusto (mediana/MAD) ou rank."""
    X = df[cols].copy()
    if how == "raw":
        return X
    g = df.groupby(["attack", "round"])
    for c in cols:
        if how == "robust_z":
            med = g[c].transform("median")
            mad = (df[c] - med).abs().groupby([df["attack"], df["round"]]).transform("median")
            X[c] = (df[c] - med) / (1.4826 * mad + 1e-12)
        else:
            X[c] = g[c].rank(pct=True)
    return X


def cross_phase(df: pd.DataFrame, train_rounds: tuple[int, int], test_from: int) -> pd.DataFrame:
    """Treina um classificador estilo RAGNAR (XGBoost, limiar 0.5) nos rounds `train_rounds` e
    testa nos rounds >= `test_from`. Protocolo "visto": treino só com o ataque avaliado;
    "nao_visto": treino com todos os ataques exceto a família do avaliado."""
    import xgboost as xgb
    df = df[df["round"] >= 2].reset_index(drop=True)  # sinais de histórico começam no round 2
    rows = []
    for protocol in ["visto", "nao_visto"]:
        for attack in sorted(df["attack"].unique()):
            tr = df[df["round"].between(*train_rounds)]
            tr = tr[tr["attack"] == attack] if protocol == "visto" else \
                tr[tr["attack"].map(attack_family) != attack_family(attack)]
            te = df[(df["round"] >= test_from) & (df["attack"] == attack)]
            if tr["malicious"].nunique() < 2 or te.empty:
                continue
            for fs, cols in FEATURE_SETS.items():
                if not set(cols) <= set(df.columns):  # sinais v2 ainda não coletados
                    continue
                for how in NORMALIZATIONS:
                    clf = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.2, eval_metric="logloss")
                    clf.fit(normalize(tr, cols, how), tr["malicious"])
                    p = clf.predict_proba(normalize(te, cols, how))[:, 1]
                    y, rnd = te["malicious"].values, te["round"].values
                    aucs = [roc_auc_score(y[rnd == r], p[rnd == r]) for r in np.unique(rnd)
                            if len(set(y[rnd == r])) == 2]
                    rows.append({"protocolo": protocol, "ataque": attack, "sinais": fs, "normalizacao": how,
                                 "auc": np.mean(aucs), "recall": (p[y == 1] >= 0.5).mean(),
                                 "fpr": (p[y == 0] >= 0.5).mean()})
    return pd.DataFrame(rows)


def write_cross_phase(cp: pd.DataFrame, out: Path, train_rounds, test_from):
    cp.to_csv(out / "cross_phase_classifier.csv", index=False)
    mean = cp.groupby(["protocolo", "sinais", "normalizacao"], sort=False)[["auc", "recall", "fpr"]].mean()
    unseen = cp[cp["protocolo"] == "nao_visto"].pivot_table(
        index="ataque", columns=["sinais", "normalizacao"], values="recall", sort=False)
    lines = [f"# Classificador entre fases (treino rounds {train_rounds[0]}–{train_rounds[1]}, "
             f"teste rounds ≥ {test_from}, limiar 0.5)", "", "## Média entre ataques", "",
             "| Protocolo | Sinais | Normalização | AUC | Recall | Falsos positivos |", "|---|---|---|---|---|---|"]
    lines += [f"| {p} | {s} | {n} | {r.auc:.2f} | {r.recall:.2f} | {r.fpr:.2f} |"
              for (p, s, n), r in mean.iterrows()]
    lines += ["", "## Recall por ataque (ataque não visto)", "",
              "| Ataque | " + " | ".join(f"{s} / {n}" for s, n in unseen.columns) + " |",
              "|---|" + "---|" * len(unseen.columns)]
    lines += [f"| {a} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |" for a, row in unseen.iterrows()]
    (out / "cross_phase_summary.md").write_text("\n".join(lines) + "\n")
    return mean


def pivot(summ: pd.DataFrame, attacks: list[str], value: str) -> pd.DataFrame:
    signals = [s for s in SIGNALS if s in set(summ["signal"])]
    tab = summ.pivot(index="signal", columns="attack", values=value).reindex(index=signals, columns=attacks)
    tab["média"] = tab[attacks].mean(axis=1)
    tab["pior"] = tab[attacks].min(axis=1)
    return tab


def write_tables(tab: pd.DataFrame, out: Path, stem: str, with_dir: bool):
    """Tabela sinal x ataque (+ média e pior caso). `with_dir` põe a direção declarada ao lado do nome."""
    cell = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    cols = list(tab.columns)
    name = lambda s: f"{s} {'↑' if DIRECTION[s] > 0 else '↓'}" if with_dir else s
    md = ["| Sinal | Família | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    tex = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
           "Sinal & Família & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for s in tab.index:
        vals = [cell(tab.loc[s, c]) for c in cols]
        md.append(f"| {name(s)} | {SIGNAL_FAMILIES[s]} | " + " | ".join(vals) + " |")
        tex_name = name(s).replace("_", r"\_").replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
        tex_vals = [(r"\textbf{" + v + "}" if v != "—" and float(v) < 0.5 else v).replace("—", "--") for v in vals]
        tex.append(tex_name + f" & {SIGNAL_FAMILIES[s]} & " + " & ".join(tex_vals) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n")
    (out / f"{stem}.tex").write_text("\n".join(tex) + "\n")


def plot_heatmap(tab: pd.DataFrame, out: Path, fname: str = "auc_heatmap.png",
                 title: str = "AUC na direção declarada a priori (média dos rounds)"):
    """Mapa sinal x ataque (+ média e pior caso). Divergente em 0.5: azul = detecta na direção
    declarada, vermelho = o ataque inverte o sinal (AUC < 0.5), cinza = não separa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # divergente azul <-> vermelho com ponto médio neutro (paleta de referência, skill dataviz)
    ramp = ["#9e2a2b", "#e34948", "#f0a39f", "#f0efec", "#9ec5f4", "#3987e5", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("auc_div", ramp)
    vals = tab.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.0 * tab.shape[1] + 3.4, 0.36 * len(tab) + 1.7))
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(vals, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=30, ha="right", fontsize=8, color="#52514e")
    ax.set_yticks(range(len(tab.index)),
                  [f"{s} {'↑' if DIRECTION[s] > 0 else '↓'}  ({SIGNAL_FAMILIES[s]})" for s in tab.index],
                  fontsize=8, color="#52514e")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(v - 0.5) >= 0.3 else "#0b0b0b")
    n_att = tab.shape[1] - 2
    ax.axvline(n_att - 0.5, color="#0b0b0b", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, tab.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tab.index)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("AUC (0,5 = não separa; < 0,5 = sinal invertido)", fontsize=8, color="#52514e")
    cb.outline.set_visible(False)
    ax.set_title(f"{title}\n↑/↓ = direção declarada antes de ver os dados · vermelho = o ataque inverte o sinal",
                 loc="left", fontsize=9, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=list(ATTACKS))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=3, help="Threads de CPU do PyTorch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="results/signals")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--train-until", type=int, default=5, help="Classificador entre fases: treino nos rounds 2..N")
    parser.add_argument("--test-from", type=int, default=8, help="Classificador entre fases: teste nos rounds >= N")
    parser.add_argument("--quick", action="store_true", help="Subamostra os dados (só para validar o script)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        t0 = time.time()
        for i, attack in enumerate(args.attacks, 1):
            path = raw_dir / f"{attack}.csv"
            if path.exists():
                print(f"[{i}/{len(args.attacks)}] {attack}: já coletado — pulando")
                continue
            print(f"[{i}/{len(args.attacks)}] {attack} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
            collect(attack, args.rounds, args.seed, torch.device(args.device), args.quick).to_csv(path, index=False)

    frames = [pd.read_csv(raw_dir / f"{a}.csv") for a in args.attacks if (raw_dir / f"{a}.csv").exists()]
    if not frames:
        print("Nenhum dado coletado.")
        return
    df = pd.concat(frames)
    attacks = [a for a in args.attacks if a in set(df["attack"])]
    by_round = auc_by_round(df)
    by_round.to_csv(out / "auc_by_round.csv", index=False)
    summ = summarize(by_round)
    summ.to_csv(out / "auc_summary.csv", index=False)
    tab_dir = pivot(summ, attacks, "auc_dir")
    tab_bi = pivot(summ, attacks, "auc_bilateral")
    # sinais medidos em todos os ataques vs. só em parte deles (v2, coletados só nos ataques recentes):
    # tabelas separadas, para que média e pior caso comparem o mesmo conjunto de ataques
    complete = tab_dir[attacks].notna().all(axis=1)
    write_tables(tab_dir[complete], out, "auc_table", with_dir=True)
    write_tables(tab_bi[complete], out, "auc_bilateral", with_dir=False)
    plot_heatmap(tab_dir[complete], out)
    plot_heatmap(tab_bi[complete], out, "auc_bilateral_heatmap.png", "AUC do escore bilateral |z robusto| (sem direção)")
    if (~complete).any():
        part = [a for a in attacks if tab_dir.loc[~complete, a].notna().all()]
        tab_part = pivot(summ[summ["signal"].isin(tab_dir.index[~complete])], part, "auc_dir")
        write_tables(tab_part, out, "auc_table_v2", with_dir=True)
        plot_heatmap(tab_part, out, "auc_heatmap_v2.png",
                     f"Sinais v2: AUC na direção declarada (medidos só em {', '.join(part)})")
    print("AUC na direção declarada:")
    print(tab_dir.round(2).to_string())
    print("\nAUC bilateral |z robusto|:")
    print(tab_bi.round(2).to_string())

    train_rounds, test_from = (2, args.train_until), args.test_from
    if df["round"].max() >= test_from:
        mean = write_cross_phase(cross_phase(df, train_rounds, test_from), out, train_rounds, test_from)
        print("\nClassificador entre fases (média entre ataques):")
        print(mean.round(2).to_string())
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
