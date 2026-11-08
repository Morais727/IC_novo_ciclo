#!/usr/bin/env python
"""Fase 8C, piloto (27/10/2026) — o LP pega numa rede grande? Desenho e critério FIXADOS ANTES DE RODAR
(results/fase8/PLANO.md). Saídas: results/fase8/raw/<nome>.json e o log de cada treino em results/fase8/<nome>.log.

ResNet18 com GroupNorm (src/models/cnn.py::ResNet18GN, 11,2 M parâmetros; uma das redes do artigo do LP), CIFAR-10,
20 clientes, 2 atacantes, 40 rodadas, α = 0,5, semente 42 (desenvolvimento) — o resto igual à Fase 7 (fase6.cfg).
Treinos (3, em paralelo):
  1. LP original (lp_full), sem defesa (lg_shadow), λ = 1 (o valor do artigo para o CIFAR-10);
  2. o mesmo com λ = 0,5 (o nosso valor);
  3. sem ataque, FedAvg (referência de acurácia).
Portão: o LP "pega" se o sucesso final sem defesa for >= 20%. Se o FedAvg sem ataque ficar abaixo de 40% de
acurácia no round 40, o resultado é "modelo não treinou". Só se o LP pegar é que a ResNet18 entra com as defesas
(nossa proposta, LASA, AlignIns, FLAME × α 0,5 e 0,3 × sementes 46, 47, 48). Nada é reajustado olhando o resultado.
Uso: python experiments/fase8_r18.py --job N (N = 0, 1, 2)
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6
from src.federated.simulate import run

OUT = Path("results/fase8")
JOBS = [("lp_full", "lg_shadow", 1.0), ("lp_full", "lg_shadow", 0.5), ("none", "fedavg", None)]


def cfg(i):
    attack, defense, lam = JOBS[i]
    c = fase6.cfg(attack, defense, 0.5, 42)
    c.model = "resnet18gn"
    tag = "_r18"
    if lam is not None:
        c.attack_params["lambda"] = lam
        tag += f"L{lam:g}"
    c.name = f"{attack}__{defense}__a0.5_s42{tag}"
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True)
    c = cfg(ap.parse_args().job)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    f = OUT / "raw" / f"{c.name}.json"
    if f.exists():
        sys.exit(0)
    print(f"=== {c.name}", flush=True)
    json.dump(run(c), open(f, "w"))
