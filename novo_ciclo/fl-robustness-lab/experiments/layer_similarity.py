#!/usr/bin/env python
"""Item 2 do plano de 13/10: similaridade entre clientes na camada crítica.

Ideia: com não-IID severo, honestos têm dados diferentes e não ficam quase idênticos em nenhum
tensor; atacantes que injetam o mesmo backdoor compartilham uma componente nos tensores de L*.
A H2 tentava remover essa componente comum e acabou apagando os atacantes; aqui ela vira o sinal.

Desenho FIXADO ANTES DE VER OS DADOS (↑ = malicioso):

  Por round: C_i = delta_i - mediana_coord_j(delta_j) (tira a direção comum a todos os clientes).
  Para cada tensor g (pesos e vieses separados, a granularidade do L* do LP):
      s_g(i) = max_{j != i} cos(C_i[g], C_j[g])
  Escores
      layer_sim       max_g z_robusto_g(s_g(i))         <- PRIMÁRIO (z entre clientes, por tensor)
      layer_sim_raw   max_g s_g(i)                      (secundário, sem padronizar)
      global_sim      max_{j != i} cos(C_i, C_j)        (linha de base: modelo inteiro, centrado)
      foolsgold_sim   max_{j != i} cos(delta_i, delta_j) (linha de base: como o FoolsGold, sem centrar)

  Ataques (α = 0,1 é o caso decisivo; α = 0,5 como controle):
      lp_full         LP dos autores: todos os atacantes enviam o MESMO delta (checagem de sanidade)
      lp_full_indiv   variante evasiva: cada atacante usa o próprio delta limpo fora de L*, então só
                      L* carrega a parte em comum. É o teste que importa.
      backdoor_pixel, dba   atacantes treinam separados (sem delta compartilhado)

  Critério do item 2: sustentado se, em lp_full_indiv com α = 0,1,
      AUC(layer_sim) >= 0,80  E  AUC(layer_sim) >= AUC(global_sim) + 0,05
  (a segunda parte exige que olhar por tensor acrescente algo à similaridade do modelo inteiro).
  Também reportados: recall e FP com z robusto do escore > 3 no round.

Agregação oráculo (só honestos), 15 rounds, semente 42, como em layer_profile.py.

Uso:
    python experiments/layer_similarity.py
    python experiments/layer_similarity.py --analyze-only
Saídas: results/layer_similarity/{raw/*.csv, summary.md, summary.csv}
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

from experiments.layer_profile import robust_z
from experiments.ragnar_eval import ATTACKS, COMMON, OPTIONAL_ATTACKS
from src.attacks.recent import tensor_groups
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig
from src.federated.simulate import _build_data, _train_round
from src.federated.utils import flatten
from src.models.cnn import build_model

SCORES = ["layer_sim", "layer_sim_raw", "global_sim", "foolsgold_sim"]
RUNS = [("lp_full_indiv", 0.1), ("lp_full", 0.1), ("backdoor_pixel", 0.1), ("dba", 0.1),
        ("lp_full_indiv", 0.5), ("lp_full", 0.5)]


def max_other_cos(x: torch.Tensor) -> np.ndarray:
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c = u @ u.T
    c.fill_diagonal_(-2.0)
    return c.max(dim=1).values.numpy()


def similarity_scores(deltas: torch.Tensor, groups: dict) -> tuple[dict, list]:
    centered = deltas - deltas.median(dim=0).values
    names = list(groups)
    s = np.stack([max_other_cos(centered[:, groups[g]]) for g in names], axis=1)   # [n, G]
    z = robust_z(s)
    return {"layer_sim": z.max(axis=1), "layer_sim_raw": s.max(axis=1),
            "global_sim": max_other_cos(centered), "foolsgold_sim": max_other_cos(deltas)}, \
        [names[k] for k in z.argmax(axis=1)]


def collect(attack, alpha, rounds, seed, device, extra_params=None):
    spec = {**ATTACKS, **OPTIONAL_ATTACKS}[attack]
    cfg = ScenarioConfig(name=f"layer_sim_{attack}", attack=attack, num_malicious=spec["num_malicious"],
                         attack_params={**spec["attack_params"], **(extra_params or {})}, defense="fedavg", seed=seed,
                         device=str(device), **{**COMMON, "rounds": rounds, "dirichlet_alpha": alpha})
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_ctor = lambda: build_model(cfg.model)
    client_data, _, _, _ = _build_data(cfg, rng)
    malicious_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())
    groups = tensor_groups(model_ctor())
    benign = [c for c in range(cfg.num_clients) if c not in malicious_ids]

    global_vector = flatten(model_ctor()).cpu()
    rows, t0 = [], time.time()
    for t in range(1, rounds + 1):
        deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg,
                                       malicious_ids, ATTACK_REGISTRY[attack], cfg.attack_params)
        deltas = deltas.cpu()
        s, top = similarity_scores(deltas, groups)
        cen = deltas - deltas.median(dim=0).values   # secundário (H4): norma por tensor, z robusto, máx
        s["layer_norm"] = robust_z(np.stack([cen[:, groups[g]].norm(dim=1).numpy() for g in groups], axis=1)).max(axis=1)
        bc = cfg.attack_params.get("_bc_layers", [])
        for c in range(cfg.num_clients):
            rows.append({"attack": attack, "alpha": alpha, "round": t, "client": c,
                         "malicious": int(c in malicious_ids), "bc_layers": "|".join(bc), "top_tensor": top[c],
                         **{k: float(v[c]) for k, v in s.items()}})
        w = weights[benign].float()
        global_vector = global_vector + (deltas[benign] * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        print(f"  [{attack} a={alpha}] round {t}/{rounds} BC={bc} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def evaluate(df):
    out = []
    for (attack, alpha), g in df.groupby(["attack", "alpha"], sort=False):
        for s in [s for s in SCORES + ["layer_norm"] if s in g.columns and g[s].notna().all()]:
            aucs, rec, fp = [], [], []
            for _, gr in g.groupby("round"):
                y, v = gr["malicious"].values, gr[s].values
                if len(set(y)) == 2:
                    aucs.append(roc_auc_score(y, v))
                flag = robust_z(v) > 3
                rec.append(flag[y == 1]); fp.append(flag[y == 0])
            out.append({"ataque": attack, "alpha": alpha, "escore": s, "auc": np.mean(aucs),
                        "recall": np.concatenate(rec).mean(), "fp": np.concatenate(fp).mean()})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/layer_similarity")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        for attack, alpha in RUNS:
            p = raw / f"{attack}_a{alpha}.csv"
            if p.exists():
                print(f"{p.name}: já coletado — pulando"); continue
            collect(attack, alpha, args.rounds, args.seed, torch.device(args.device)).to_csv(p, index=False)

    df = pd.concat([pd.read_csv(raw / f"{a}_a{al}.csv") for a, al in RUNS if (raw / f"{a}_a{al}.csv").exists()])
    ev = evaluate(df)
    ev.to_csv(out / "summary.csv", index=False)
    md = ["# Similaridade na camada crítica (item 2)", "", "Desenho e critério fixados antes da coleta (ver docstring).", ""]
    get = lambda a, al, s: ev[(ev.ataque == a) & (ev.alpha == al) & (ev.escore == s)].auc
    if len(get("lp_full_indiv", 0.1, "layer_sim")):
        a1, a0 = get("lp_full_indiv", 0.1, "layer_sim").iloc[0], get("lp_full_indiv", 0.1, "global_sim").iloc[0]
        ok = a1 >= 0.8 and a1 >= a0 + 0.05
        md.append(f"- Item 2 (lp_full_indiv, α=0,1): AUC layer_sim = {a1:.2f}, global_sim = {a0:.2f} → "
                  f"{'SUSTENTADO' if ok else 'REFUTADO'} (critério: ≥ 0,80 e ≥ global + 0,05)")
    md += ["", "| ataque | α | escore | AUC | recall z>3 | FP z>3 |", "|---|---|---|---|---|---|"]
    md += [f"| {r.ataque} | {r.alpha} | {r.escore} | {r.auc:.2f} | {r.recall:.2f} | {r.fp:.2f} |" for r in ev.itertuples()]
    mal = df[df.malicious == 1]
    md += ["", "Tensor com maior z nos maliciosos (L* real na coluna bc_layers):", ""]
    for (a, al), g in mal.groupby(["attack", "alpha"], sort=False):
        top = ", ".join(f"{k}: {v:.0%}" for k, v in g.top_tensor.value_counts(normalize=True).head(3).items())
        bc = g.bc_layers.dropna().value_counts().head(1)
        md.append(f"- {a} α={al}: {top}" + (f" (L* mais comum: {bc.index[0]})" if len(bc) else ""))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
