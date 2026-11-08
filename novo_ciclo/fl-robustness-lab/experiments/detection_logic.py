#!/usr/bin/env python
"""Avaliação offline da lógica de detecção especificada no entregável do mês 3.

Reaproveita os sinais já coletados por experiments/signal_analysis.py (results/signals/raw/*.csv);
não treina nada e roda em segundos. Implementa a regra sem treino da especificação
(especificacao-sinais-deteccao.tex, seção "Lógica de detecção"):

  1. normalização no round:   z_k(i,t) = (s_k - mediana_t) / (1.4826 · MAD_t)
  2. evidência por sinal:     unilateral  e_k = max(0, d_k · z_k)   (d_k = direção declarada)
                              bilateral   e_k = |z_k|               (desvio em qualquer sentido)
  3. escore do cliente:       S(i,t) = max_k e_k      e o sinal que atinge o máximo = explicação
  4. acúmulo entre rounds:    C(i,t) = max(0, C(i,t-1) + S(i,t) - mediana_i S(·,t) - kappa)
                              (desconta o nível típico de S no round: o máximo de K escores |z| de um
                              honesto já passa de 1, e um kappa fixo faria o CUSUM derivar para cima)
  5. decisão:                 exclui se S > tau (instantânea) ou C > h (acumulada), no máximo
                              floor((n-1)/2) clientes por round (os de maior escore) — nunca
                              "aceita todos" quando muitos são marcados

Parâmetros fixados a priori (não ajustados nestes dados): tau = 3, kappa = 0.5, h = 6.
--kappa-mode fixed reproduz a variante inicial (sem descontar a mediana, kappa = 1.5).
A escolha uni/bilateral por sinal usa a análise de sinais destes mesmos dados: os números
aqui são evidência de viabilidade, não validação (que exige outra semente, meses 7-9).

Saídas em results/detection_logic/: per_attack.csv, summary.md, summary.tex, explanations.csv

    python experiments/detection_logic.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.signals import DIRECTION

warnings.filterwarnings("ignore", message="All-NaN slice")

UNI, BI = "uni", "bi"
RULES = {
    # linha de base: as 4 features do RAGNAR v1, na direção declarada
    "RAGNAR-4": {"l1_mean": UNI, "l2_mean": UNI, "l3_mean": UNI, "cos_mean": UNI},
    # proposta sem os sinais de conluio (que dependem de atacantes enviarem vetores quase iguais)
    "proposta": {"norm": UNI, "mahalanobis": UNI, "cos_median": BI, "fc1_cos_median": BI,
                 "holdout_loss_gain": UNI, "holdout_acc": UNI, "hist_cos_self": BI},
    "proposta+conluio": {"norm": UNI, "mahalanobis": UNI, "cos_median": BI, "fc1_cos_median": BI,
                         "holdout_loss_gain": UNI, "holdout_acc": UNI, "hist_cos_self": BI,
                         "max_cos_peer": UNI, "hist_max_cos_cum": UNI},
}
ATTACK_ORDER = ["gaussian_noise", "ipm", "lie", "minmax", "minsum", "label_flip", "backdoor_pixel", "backdoor_pgd",
                "mimic", "lf", "dba", "lp", "edge_case_pgd"]
# ataques vistos ao desenhar a regra (modos uni/bilateral escolhidos com eles) vs. implementados depois,
# com a regra já congelada: estes são o teste fora da amostra
DESIGN_ATTACKS = ATTACK_ORDER[:8]
NEW_ATTACKS = ATTACK_ORDER[8:]


def evidence(g: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Evidência e_k por cliente num round (NaN -> 0: sinal indisponível não acusa ninguém)."""
    out = {}
    for s, mode in rule.items():
        v = g[s].to_numpy(dtype=float)
        med = np.nanmedian(v)
        z = (v - med) / (1.4826 * np.nanmedian(np.abs(v - med)) + 1e-12)
        e = np.abs(z) if mode == BI else np.maximum(0.0, DIRECTION[s] * z)
        out[s] = np.nan_to_num(e, nan=0.0)
    return pd.DataFrame(out, index=g.index)


def run_rule(df: pd.DataFrame, rule: dict, tau: float, kappa: float, h: float,
             kappa_mode: str = "median") -> pd.DataFrame:
    rows = []
    for attack, ga in df.groupby("attack"):
        cusum = {}
        for r, g in ga.sort_values("round").groupby("round"):
            ev = evidence(g, rule)
            S = ev.max(axis=1).to_numpy()
            top = ev.idxmax(axis=1).to_numpy()
            clients = g["client"].to_numpy()
            ref = np.median(S) + kappa if kappa_mode == "median" else kappa
            C = np.array([max(0.0, cusum.get(c, 0.0) + S[j] - ref) for j, c in enumerate(clients)])
            cusum.update(zip(clients, C))
            flag = (S > tau) | (C > h)
            cap = (len(clients) - 1) // 2
            if flag.sum() > cap:  # limite de exclusões: fica com os `cap` de maior escore
                keep = np.argsort(-np.maximum(S / tau, C / h))[:cap]
                flag = np.zeros_like(flag); flag[keep] = True
            for j, c in enumerate(clients):
                rows.append({"attack": attack, "round": r, "client": c, "malicious": g["malicious"].iloc[j],
                             "S": S[j], "C": C[j], "inst": S[j] > tau, "flag": bool(flag[j]), "top_signal": top[j]})
    return pd.DataFrame(rows)


def metrics(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, g in res.groupby("attack"):
        aucs = [roc_auc_score(x["malicious"], x["S"]) for _, x in g.groupby("round") if x["malicious"].nunique() == 2]
        m, b = g[g.malicious == 1], g[g.malicious == 0]
        rows.append({"attack": attack, "auc_S": np.mean(aucs),
                     "recall_inst": m["inst"].mean(), "fpr_inst": b["inst"].mean(),
                     "recall": m["flag"].mean(), "fpr": b["flag"].mean()})
    return pd.DataFrame(rows).set_index("attack").reindex([a for a in ATTACK_ORDER if a in set(res.attack)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/signals/raw")
    parser.add_argument("--out-dir", default="results/detection_logic")
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--kappa-mode", choices=["median", "fixed"], default="median")
    parser.add_argument("--h", type=float, default=6.0)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(f) for f in sorted(Path(args.raw_dir).glob("*.csv"))], ignore_index=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables, expl = [], []
    for name, rule in RULES.items():
        res = run_rule(df, rule, args.tau, args.kappa, args.h, args.kappa_mode)
        m = metrics(res)
        m.insert(0, "regra", name)
        tables.append(m.reset_index())
        # explicação: qual sinal deu a maior evidência nos maliciosos marcados
        hit = res[(res.malicious == 1) & res.flag]
        e = hit.groupby("attack")["top_signal"].agg(lambda x: x.value_counts(normalize=True).round(2).head(2).to_dict())
        expl.append(e.rename(name))
    per = pd.concat(tables, ignore_index=True)
    per.to_csv(out / "per_attack.csv", index=False)
    pd.concat(expl, axis=1).to_csv(out / "explanations.csv")

    label = {"auc_S": "AUC", "recall": "Recall", "fpr": "FP"}
    wide = per.pivot(index="attack", columns="regra", values=list(label)).reindex(
        [a for a in ATTACK_ORDER if a in set(per.attack)])
    design = [a for a in DESIGN_ATTACKS if a in wide.index]
    new = [a for a in NEW_ATTACKS if a in wide.index]
    if new:
        wide.loc["média (desenho)"] = wide.loc[design].mean()
        wide.loc["média (novos)"] = wide.loc[new].mean()
    else:
        wide.loc["média"] = wide.loc[design].mean()
    rules = list(RULES)
    md = ["| Ataque | " + " | ".join(f"{label[k]} {r}" for k in label for r in rules) + " |",
          "|---|" + "---|" * (len(label) * len(rules))]
    tex = [r"\begin{tabular}{l" + "ccc" * len(label) + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{" + str(len(rules)) + "}{c}{" + label[k] + "}" for k in label) + r" \\",
           " & ".join([""] + [r.replace("+", "+") for _ in label for r in rules]) + r" \\", r"\midrule"]
    for a, row in wide.iterrows():
        vals = [f"{row[(k, r)]:.2f}" for k in label for r in rules]
        md.append(f"| {a} | " + " | ".join(vals) + " |")
        if a.startswith("média") and not tex[-1].startswith(r"\midrule") and "média" not in tex[-1]:
            tex.append(r"\midrule")
        if new and a == new[0]:
            tex.append(r"\midrule")
        tex.append(a.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    head = (f"Regra sem treino, tau={args.tau}, kappa={args.kappa} ({args.kappa_mode}), h={args.h}; recall/FP = fração de "
            "(cliente, round) excluídos; AUC = do escore instantâneo S, por round.\n\n")
    (out / "summary.md").write_text(head + "\n".join(md) + "\n")
    (out / "summary.tex").write_text("\n".join(tex) + "\n")
    print(head + "\n".join(md))
    print("\nSinal que mais explica as detecções (maliciosos marcados):")
    print(pd.concat(expl, axis=1).to_string())


if __name__ == "__main__":
    main()
