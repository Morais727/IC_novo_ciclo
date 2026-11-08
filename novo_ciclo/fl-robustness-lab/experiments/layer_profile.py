#!/usr/bin/env python
"""Perfil por camada: teste das hipóteses H1 e H2 da proposta (apresentação de 11/10/2026).

Escores e critérios FIXADOS ANTES DE VER OS DADOS (todos com direção ↑ = malicioso):

  Por round t, cliente i e camada l (módulos da CNN: conv1, conv2, conv3, fc1, fc2):
    d[i,l] = || delta[i,l] - mediana_coord_j(delta[j,l]) ||       (distância à mediana, na camada)
    r[i,l] = d[i,l] / mediana_j d[j,l]                              (relativa ao típico da camada)

  Linhas de base
    global_l2_median  || delta_i - mediana ||, modelo inteiro (o sinal l2_median do M3)
    layer_max_z       max_l z_robusto_l(d[i,l]): detector camada a camada (análogo ao
                      Multi-Krum por camada do Apêndice 14 do artigo do LP)
  Propostos (H1)
    conc_top2         (soma dos 2 maiores r[i,l]^2) / (soma_l r[i,l]^2)   <- escore PRIMÁRIO de H1
    layer_excess      max_l log r[i,l] - mediana_l log r[i,l]              <- secundário
                      (autonormalizado: um honesto não-IID que desvia em TODAS as camadas não pontua)
  Variante H2 (desconto do subespaço de heterogeneidade, à la BOBA)
    *_resid           os mesmos escores após remover de cada delta a projeção nos c-1 = 9 primeiros
                      componentes principais dos deltas centrados do round

Métricas: AUC por round (média entre rounds) e decisão por z robusto do escore no round > 3
(recall = fração de pares maliciosos excluídos; FP = fração de pares honestos excluídos).

Critérios
  H1 sustentada se AUC(conc_top2) >= 0,8 no LP (lp e lp_full) com alpha = 0,5; refutada se < 0,8.
  H2 sustentada se, em alpha = 0,1, conc_top2_resid tem FP menor que conc_top2 e perde no máximo
     5 p.p. de recall.

Agregação oráculo (só honestos), como em signal_analysis.py: mede detectabilidade sobre a mesma
trajetória de treino.

Uso:
    python experiments/layer_profile.py --attacks lp lp_full --alphas 0.5 0.1 --rounds 15
    python experiments/layer_profile.py --analyze-only
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

from experiments.ragnar_eval import ATTACKS, COMMON, OPTIONAL_ATTACKS
from src.attacks.recent import layer_groups
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig
from src.federated.simulate import _build_data, _train_round
from src.federated.utils import flatten
from src.models.cnn import build_model

LAYER_NAMES = {"features.0": "conv1", "features.3": "conv2", "features.6": "conv3", "fc1": "fc1", "fc2": "fc2"}
SCORES = ["global_l2_median", "layer_max_z", "conc_top2", "layer_excess",
          "global_l2_median_resid", "layer_max_z_resid", "conc_top2_resid", "layer_excess_resid"]
N_PC = 9  # c - 1 classes


def robust_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0)
    return (v - med) / (1.4826 * mad + 1e-12)


def layer_scores(deltas: torch.Tensor, groups: dict) -> tuple[dict, np.ndarray]:
    med = deltas.median(dim=0).values
    names = list(groups)
    d = torch.stack([(deltas[:, groups[g]] - med[groups[g]]).norm(dim=1) for g in names], dim=1).numpy()
    r = d / (np.median(d, axis=0, keepdims=True) + 1e-12)
    r2 = np.sort(r ** 2, axis=1)
    logr = np.log(r + 1e-12)
    return {
        "global_l2_median": (deltas - med).norm(dim=1).numpy(),
        "layer_max_z": robust_z(d).max(axis=1),
        "conc_top2": r2[:, -2:].sum(axis=1) / r2.sum(axis=1),
        "layer_excess": logr.max(axis=1) - np.median(logr, axis=1),
    }, r


def remove_heterogeneity(deltas: torch.Tensor, k: int = N_PC) -> torch.Tensor:
    centered = deltas - deltas.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:k]                                   # [k, D]
    return deltas - (deltas @ basis.T) @ basis


def collect(attack: str, alpha: float, rounds: int, seed: int, device: torch.device) -> pd.DataFrame:
    spec = {**ATTACKS, **OPTIONAL_ATTACKS}[attack]
    cfg = ScenarioConfig(name=f"layer_profile_{attack}", attack=attack, num_malicious=spec["num_malicious"],
                         attack_params=dict(spec["attack_params"]), defense="fedavg", seed=seed,
                         device=str(device), **{**COMMON, "rounds": rounds, "dirichlet_alpha": alpha})
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_ctor = lambda: build_model(cfg.model)
    client_data, _, _, _ = _build_data(cfg, rng)
    malicious_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())
    groups = layer_groups(model_ctor())
    benign = [c for c in range(cfg.num_clients) if c not in malicious_ids]

    global_vector = flatten(model_ctor()).cpu()
    rows, t0 = [], time.time()
    for t in range(1, rounds + 1):
        deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg,
                                       malicious_ids, ATTACK_REGISTRY[attack], cfg.attack_params)
        deltas = deltas.cpu()
        s, r = layer_scores(deltas, groups)
        s_res, r_res = layer_scores(remove_heterogeneity(deltas), groups)
        s.update({f"{k}_resid": v for k, v in s_res.items()})
        bc = cfg.attack_params.get("_bc_layers", [])
        for c in range(cfg.num_clients):
            rows.append({"attack": attack, "alpha": alpha, "round": t, "client": c,
                         "malicious": int(c in malicious_ids), "bc_layers": "|".join(bc),
                         **{k: float(s[k][c]) for k in SCORES},
                         **{f"r_{LAYER_NAMES[g]}": float(r[c, j]) for j, g in enumerate(groups)},
                         **{f"rres_{LAYER_NAMES[g]}": float(r_res[c, j]) for j, g in enumerate(groups)}})
        w = weights[benign].float()
        global_vector = global_vector + (deltas[benign] * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        print(f"  [{attack} a={alpha}] round {t}/{rounds} BC={bc} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (attack, alpha), g in df.groupby(["attack", "alpha"]):
        for s in SCORES:
            aucs, rec, fp = [], [], []
            for _, gr in g.groupby("round"):
                y, v = gr["malicious"].values, gr[s].values
                if len(set(y)) == 2:
                    aucs.append(roc_auc_score(y, v))
                flag = robust_z(v) > 3
                rec.append(flag[y == 1])
                fp.append(flag[y == 0])
            out.append({"ataque": attack, "alpha": alpha, "escore": s, "auc": np.mean(aucs),
                        "recall": np.concatenate(rec).mean(), "fp": np.concatenate(fp).mean()})
    return pd.DataFrame(out)


def verdicts(ev: pd.DataFrame) -> list[str]:
    get = lambda a, al, s, m: ev[(ev.ataque == a) & (ev.alpha == al) & (ev.escore == s)][m].iloc[0]
    lines = []
    for a in sorted(ev.ataque.unique()):
        if 0.5 in set(ev[ev.ataque == a].alpha):
            auc = get(a, 0.5, "conc_top2", "auc")
            lines.append(f"- H1 ({a}, α=0,5): AUC(conc_top2) = {auc:.2f} → "
                         f"{'sustentada' if auc >= 0.8 else 'REFUTADA'} (critério ≥ 0,80)")
        if 0.1 in set(ev[ev.ataque == a].alpha):
            fp0, fp1 = get(a, 0.1, "conc_top2", "fp"), get(a, 0.1, "conc_top2_resid", "fp")
            r0, r1 = get(a, 0.1, "conc_top2", "recall"), get(a, 0.1, "conc_top2_resid", "recall")
            ok = fp1 < fp0 and (r0 - r1) <= 0.05
            lines.append(f"- H2 ({a}, α=0,1): FP {fp0:.2f} → {fp1:.2f}, recall {r0:.2f} → {r1:.2f} → "
                         f"{'sustentada' if ok else 'REFUTADA'}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", nargs="+", default=["lp", "lp_full"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/layer_profile")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        for attack in args.attacks:
            for alpha in args.alphas:
                path = raw / f"{attack}_a{alpha}.csv"
                if path.exists():
                    print(f"{path.name}: já coletado — pulando")
                    continue
                collect(attack, alpha, args.rounds, args.seed, torch.device(args.device)).to_csv(path, index=False)

    df = pd.concat([pd.read_csv(p) for p in sorted(raw.glob("*.csv"))])
    ev = evaluate(df)
    ev.to_csv(out / "summary.csv", index=False)
    tab = ev.pivot_table(index="escore", columns=["ataque", "alpha"], values=["auc", "recall", "fp"], sort=False)
    md = ["# Perfil por camada — H1 e H2", "", "Critérios fixados antes da coleta (ver docstring).", "",
          *verdicts(ev), "", "## AUC / recall / FP (z > 3) por escore", ""]
    for m in ["auc", "recall", "fp"]:
        t = tab[m].reindex(SCORES)
        md += [f"### {m}", "", "| escore | " + " | ".join(f"{a} α={al}" for a, al in t.columns) + " |",
               "|---|" + "---|" * t.shape[1]]
        md += [f"| {s} | " + " | ".join(f"{v:.2f}" for v in t.loc[s]) + " |" for s in t.index]
        md.append("")
    rcols = [c for c in df.columns if c.startswith("r_")]
    prof = df.groupby(["attack", "alpha", "malicious"])[rcols].median().round(2)
    md += ["## Perfil mediano r[l] (desvio da camada / típico da camada)", "",
           "| ataque | α | malicioso | " + " | ".join(c[2:] for c in rcols) + " |", "|---|---|---|" + "---|" * len(rcols)]
    md += [f"| {a} | {al} | {m} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |"
           for (a, al, m), row in prof.iterrows()]
    md.append("")
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
