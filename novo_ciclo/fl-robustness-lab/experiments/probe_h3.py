#!/usr/bin/env python
"""Sonda sem gatilho: teste da hipótese H3 da proposta (apresentação de 11/10/2026).

Desenho FIXADO ANTES DE VER OS DADOS:

  Sonda: K = 64 manchas aleatórias (semente fixa, as mesmas em todos os rounds), lado de 3 a 6 px,
  posição uniforme na imagem 32x32; metade de cor sólida (valor por canal ~ U[-3, 3] no espaço
  normalizado) e metade de ruído (N(0, 1,5) por pixel). Cada mancha é colada nas 100 imagens do holdout.

  Para um modelo m, mancha p e classe c:
      dP_m(p, c) = média no holdout de [ softmax_c(m(x com p)) - softmax_c(m(x)) ]
  Escore do cliente i (↑ = malicioso), sem conhecer gatilho nem classe-alvo:
      probe       = max_{p,c} [ dP_{w+delta_i}(p, c) - dP_w(p, c) ]       <- escore PRIMÁRIO de H3
  Subtrair o efeito no modelo global w desconta o que todo modelo já faz com aquela mancha. Como dP
  é uma diferença com/sem mancha no mesmo modelo, o viés de classe do cliente não-IID se cancela.

  Variantes (secundárias):
      probe_far   o mesmo, só com manchas que não tocam o quadrante 8x8 inferior direito, onde fica
                  o gatilho real do laboratório (controle: a sonda não depende de acertar o lugar)
      probe_solid / probe_noise   só manchas sólidas / só ruído (o gatilho real é um quadrado sólido)
      probe_x5    o mesmo com w + 5*delta_i (amplifica o efeito de um round)

  Métricas: AUC por round (média entre rounds) e decisão por z robusto > 3 no round.
  Critério (slide 15): H3 refutada se AUC(probe) <= 0,7 no lp_full com alpha = 0,1.

Agregação oráculo (só honestos), como em layer_profile.py e signal_analysis.py.

Uso:
    python experiments/probe_h3.py                     # lp_full, lp, backdoor_pixel, dba (a=0,1) + lp_full (a=0,5)
    python experiments/probe_h3.py --analyze-only
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
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig
from src.federated.simulate import _build_data, _train_round
from src.federated.utils import flatten, load_vector
from src.models.cnn import build_model

K = 64
SCORES = ["probe", "probe_far", "probe_solid", "probe_noise", "probe_x5"]
ORACLE = False
RUNS = [("lp_full", 0.1), ("lp", 0.1), ("backdoor_pixel", 0.1), ("dba", 0.1), ("lp_full", 0.5)]


def oracle_patches():
    """DIAGNÓSTICO, não detector: só o gatilho real do laboratório (4x4 no canto inferior direito,
    valor TRIGGER_VALUE). Dá o teto do que uma sonda por manchas consegue ver numa atualização de um round."""
    from src.data.datasets import TRIGGER_VALUE
    mask = torch.zeros(1, 32, 32); mask[:, -4:, -4:] = 1
    return [(mask, torch.full((3, 32, 32), TRIGGER_VALUE), True, False)]


def make_patches(seed: int = 1234):
    """Lista de (máscara [1,32,32], valores [3,32,32], é_sólida, longe_do_canto)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(K):
        s = int(torch.randint(3, 7, (1,), generator=g))
        r, c = (int(v) for v in torch.randint(0, 32 - s + 1, (2,), generator=g))
        mask = torch.zeros(1, 32, 32)
        mask[:, r:r + s, c:c + s] = 1
        solid = k % 2 == 0
        if solid:
            vals = (torch.rand(3, 1, 1, generator=g) * 6 - 3).expand(3, 32, 32).clone()
        else:
            vals = torch.randn(3, 32, 32, generator=g) * 1.5
        far = not (r + s > 24 and c + s > 24)
        out.append((mask, vals, solid, far))
    return out


@torch.no_grad()
def patch_effect(model, x_clean, x_patched):
    """dP [K, C]: aumento médio de probabilidade de cada classe causado por cada mancha."""
    p0 = torch.softmax(model(x_clean), 1).mean(0)                       # [C]
    # em blocos de 8 manchas (800 imagens): a MX450 tem 2 GB
    p = torch.cat([torch.softmax(model(x_patched[i:i + 8].flatten(0, 1)), 1).view(-1, x_clean.shape[0], p0.numel()).mean(1)
                   for i in range(0, K, 8)])                                  # [K, C]
    return p - p0


def collect(attack, alpha, rounds, seed, device):
    spec = {**ATTACKS, **OPTIONAL_ATTACKS}[attack]
    cfg = ScenarioConfig(name=f"probe_{attack}", attack=attack, num_malicious=spec["num_malicious"],
                         attack_params=dict(spec["attack_params"]), defense="fedavg", seed=seed,
                         device=str(device), **{**COMMON, "rounds": rounds, "dirichlet_alpha": alpha})
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_ctor = lambda: build_model(cfg.model)
    client_data, holdout, _, _ = _build_data(cfg, rng)
    malicious_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())
    benign = [c for c in range(cfg.num_clients) if c not in malicious_ids]

    x = torch.stack([holdout[i][0] for i in range(len(holdout))]).to(device)
    patches = oracle_patches() if ORACLE else make_patches()
    global K
    K = len(patches)
    masks = torch.stack([m for m, _, _, _ in patches]).to(device)          # [K,1,32,32]
    vals = torch.stack([v for _, v, _, _ in patches]).to(device)           # [K,3,32,32]
    x_p = x.unsqueeze(0) * (1 - masks.unsqueeze(1)) + (vals * masks).unsqueeze(1)   # [K,n,3,32,32]
    far = torch.tensor([f for *_, f in patches])
    solid = torch.tensor([s for _, _, s, _ in patches])
    subsets = {"probe": torch.ones(K, dtype=torch.bool), "probe_far": far, "probe_solid": solid, "probe_noise": ~solid}

    model = model_ctor().to(device).eval()
    def effect(vec):
        load_vector(model, vec.to(device))
        return patch_effect(model, x, x_p).cpu()

    global_vector = flatten(model_ctor()).cpu()
    rows, t0 = [], time.time()
    for t in range(1, rounds + 1):
        deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg,
                                       malicious_ids, ATTACK_REGISTRY[attack], cfg.attack_params)
        deltas = deltas.cpu()
        base = effect(global_vector)
        for c in range(cfg.num_clients):
            diff = effect(global_vector + deltas[c]) - base                  # [K, C]
            diff5 = effect(global_vector + 5 * deltas[c]) - base
            row = {"attack": attack, "alpha": alpha, "round": t, "client": c, "malicious": int(c in malicious_ids)}
            for name, sel in subsets.items():
                if not sel.any():
                    continue
                d = diff[sel]
                row[name] = float(d.max())
                if name == "probe":
                    k, cls = divmod(int(d.argmax()), d.shape[1])
                    row["probe_class"] = cls
                    row["probe_patch"] = k
            row["probe_x5"] = float(diff5.max())
            rows.append(row)
        w = weights[benign].float()
        global_vector = global_vector + (deltas[benign] * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        print(f"  [{attack} a={alpha}] round {t}/{rounds} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def evaluate(df):
    out = []
    for (attack, alpha), g in df.groupby(["attack", "alpha"], sort=False):
        for s in SCORES:
            if s not in g or g[s].isna().all():
                continue
            aucs, rec, fp = [], [], []
            for _, gr in g.groupby("round"):
                y, v = gr["malicious"].values, gr[s].values
                if len(set(y)) == 2:
                    aucs.append(roc_auc_score(y, v))
                flag = robust_z(v) > 3
                rec.append(flag[y == 1]); fp.append(flag[y == 0])
            out.append({"ataque": attack, "alpha": alpha, "escore": s, "auc": np.mean(aucs),
                        "auc_rounds_8_15": np.mean(aucs[7:]) if len(aucs) > 7 else np.nan,
                        "recall": np.concatenate(rec).mean(), "fp": np.concatenate(fp).mean()})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/probe_h3")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="diagnóstico: só o gatilho real (teto, não detector)")
    parser.add_argument("--runs", nargs="+", default=None, help="ex.: lp_full:0.1 backdoor_pixel:0.1")
    args = parser.parse_args()
    global ORACLE, RUNS
    ORACLE = args.oracle
    if args.runs:
        RUNS = [(r.split(":")[0], float(r.split(":")[1])) for r in args.runs]

    out = Path(args.out_dir); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        for attack, alpha in RUNS:
            path = raw / f"{attack}_a{alpha}.csv"
            if path.exists():
                print(f"{path.name}: já coletado — pulando"); continue
            collect(attack, alpha, args.rounds, args.seed, torch.device(args.device)).to_csv(path, index=False)

    df = pd.concat([pd.read_csv(raw / f"{a}_a{al}.csv") for a, al in RUNS if (raw / f"{a}_a{al}.csv").exists()])
    ev = evaluate(df)
    ev.to_csv(out / "summary.csv", index=False)
    md = ["# Sonda sem gatilho — H3", "", "Desenho e critério fixados antes da coleta (ver docstring).", ""]
    m = ev[(ev.ataque == "lp_full") & (ev.alpha == 0.1) & (ev.escore == "probe")]
    if ORACLE:
        md[0] = "# DIAGNÓSTICO ORÁCULO da sonda (gatilho real; teto, não detector; NÃO é o teste de H3)"
    elif len(m):
        auc = m.auc.iloc[0]
        md.append(f"- H3 (lp_full, α=0,1): AUC(probe) = {auc:.2f} → {'sustentada' if auc > 0.7 else 'REFUTADA'} (refutada se ≤ 0,70)")
    md += ["", "| ataque | α | escore | AUC | AUC rounds 8–15 | recall z>3 | FP z>3 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r.ataque} | {r.alpha} | {r.escore} | {r.auc:.2f} | {r.auc_rounds_8_15:.2f} | {r.recall:.2f} | {r.fp:.2f} |"
           for r in ev.itertuples()]
    mal = df[df.malicious == 1]
    md += ["", "Classe apontada pela sonda nos maliciosos (alvo real = 0 em todos os backdoors):", ""]
    md += [f"- {a} α={al}: " + ", ".join(f"{k}: {v:.0%}" for k, v in g["probe_class"].value_counts(normalize=True).head(3).items())
           for (a, al), g in mal.groupby(["attack", "alpha"], sort=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
