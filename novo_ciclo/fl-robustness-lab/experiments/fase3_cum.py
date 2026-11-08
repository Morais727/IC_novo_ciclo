#!/usr/bin/env python
"""Fase 3, H8 — atualização ACUMULADA por cliente (fixado ANTES de olhar; dumps de fedavg_dump, sementes 44 e 45 de
desenvolvimento, rounds 10/20/30/40). Hipótese: o ruído do atacante é independente a cada round e se cancela na
soma (cresce ~√R), enquanto a parte do backdoor é igual em todo round e nos dois atacantes (cresce ~R). Então, na
soma dos deltas desde o round 1, os dois atacantes ficam parecidos entre si nos tensores de L* bem acima do que
dois honestos com dados parecidos ficam.
Escores (por round salvo): C_i = cum_i - mediana_coord(cum); por tensor g: cos(C_i[g], C_j[g]);
  P  posto do par de atacantes entre os 190 pares, por tensor (o melhor tensor) e na média dos tensores de peso;
  S  escore por cliente = max_g z robusto de max_{j≠i} cos (como a nossa proposta, mas na soma).
Critério H8 (para virar defesa em ciclo fechado): nos rounds 20 e 30 das duas sementes (4 casos), os 2 atacantes
são os 2 maiores S em >= 3 casos, OU o par de atacantes é o 1º de 190 em algum tensor de peso fixo (o mesmo nos
4 casos). Também reportado para os deltas de um round só (controle, mesmo cálculo com 'deltas').
Uso: python experiments/fase3_cum.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attacks.recent import tensor_groups
from src.models.cnn import build_model

D = Path("results/fase3/dumps")


def rz(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (v - med) / mad


def analyze(vecs, mal, groups):
    C = vecs - vecs.median(0).values
    n = C.shape[0]; a, b = mal
    out = {"S": None, "pair_rank": {}}
    zs = []
    for g, sl in groups.items():
        x = C[:, sl]; x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
        cm = (x @ x.T).numpy(); np.fill_diagonal(cm, -np.inf)
        iu = np.triu_indices(n, 1); vals = cm[iu]
        out["pair_rank"][g] = int((vals > cm[a, b]).sum() + 1)
        zs.append(rz(cm.max(1)))
    S = np.max(zs, 0)
    order = np.argsort(-S).tolist()
    out["S_ranks"] = sorted([order.index(a) + 1, order.index(b) + 1])
    return out


def main():
    groups = {g: sl for g, sl in tensor_groups(build_model("smallcnn")).items() if g.endswith("weight")}
    for f in sorted(D.glob("*.pt"), key=lambda p: (p.stem.split("_s")[-1].split("_")[0], int(p.stem.split("_r")[-1]))):
        d = torch.load(f, weights_only=False)
        for kind in ("cum", "deltas"):
            r = analyze(d[kind].float(), d["malicious"], groups)
            print(f"{f.stem:40s} {kind:6s} atac. nos postos S {r['S_ranks']} | posto do par (de 190) por tensor {r['pair_rank']}")


if __name__ == "__main__":
    main()
