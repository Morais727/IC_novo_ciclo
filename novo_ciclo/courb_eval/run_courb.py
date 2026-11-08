#!/usr/bin/env python
"""Avalia o RAGNAR no framework do CoUrb-2024 (Flower 1.7 + Keras) contra os 8 ataques do
fl-robustness-lab, para comparar com a avaliação feita no laboratório.

Braços por ataque (mesma semente, mesmos cids maliciosos do laboratório):
  fedavg         sem defesa
  ragnar_pre     RAGNAR com o classificador pré-treinado do CoUrb (MODELOS/*_CNN)
  ragnar_5r_nv   RAGNAR treinado com 5 rounds simulados no início, sem a família do ataque
                 avaliado (mesmo protocolo "não visto" do laboratório)
  ragnar_5r_v    idem, com o bootstrap só com o ataque avaliado ("visto")
+ clean (sem ataque, FedAvg).

Uso (a partir de courb_eval/):
    .venv/bin/python run_courb.py                         # tudo, 40 rounds
    .venv/bin/python run_courb.py --rounds 3 --attacks lie --arms fedavg ragnar_pre   # teste rápido
Resultados: results/<ataque>__<braço>_s<seed>.json (formato do fl-robustness-lab).
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import flwr as fl
import numpy as np
import tensorflow as tf

from attacks import ATTACK_PARAMS, NUM_MALICIOUS
from client import CoUrbClient, create_model
from ragnar_courb import load_pretrained, train_like_courb
from strategy import RagnarCoUrb

ATTACKS = ["gaussian_noise", "ipm", "lie", "minmax", "minsum", "label_flip", "backdoor_pixel", "backdoor_pgd"]
FAMILY = {"backdoor_pixel": "backdoor", "backdoor_pgd": "backdoor", "minmax": "minmax", "minsum": "minmax"}
BOOT_POOL = ["gaussian_noise", "sign_flip", "ipm", "lie", "minmax", "minsum", "label_flip",
             "backdoor_pixel", "backdoor_pgd"]
ARMS = ["fedavg", "ragnar_pre", "ragnar_5r_nv", "ragnar_5r_v"]


def fam(a):
    return FAMILY.get(a, a)


def simulate(strategy, args, num_rounds):
    def client_fn(cid: str):
        return CoUrbClient(int(cid), args.alpha, args.dataset_size, args.seed).to_client()

    threads = {"TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1",
               "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""}
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=args.clients, strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"num_cpus": args.workers, "include_dashboard": False, "log_to_driver": False,
                       "runtime_env": {"env_vars": threads, "working_dir": str(Path(__file__).parent),
                                       "excludes": [".venv", "results", "*.log"]}},
    )


def bootstrap_classifier(attack, arm, args, init_w):
    """5 rounds simulados a partir do modelo inicial, um ataque sorteado do pool por round."""
    pool = [attack] if arm == "ragnar_5r_v" else [a for a in BOOT_POOL if fam(a) != fam(attack)]
    rng = np.random.default_rng(args.seed + 999)
    plan = {}
    for r in range(1, args.boot_rounds + 1):
        mal = set(rng.choice(args.clients, size=NUM_MALICIOUS[attack], replace=False).tolist())
        plan[r] = (str(rng.choice(pool)), mal)
    strat = RagnarCoUrb(initial_weights=init_w, schedule=lambda r: plan[r], mode="collect", seed=args.seed)
    simulate(strat, args, args.boot_rounds)
    X, y = np.concatenate(strat.collected_X), np.concatenate(strat.collected_y)
    return train_like_courb(X, y, seed=args.seed), pool, [plan[r][0] for r in sorted(plan)]


def run_one(attack, arm, args, test_data, eval_model):
    tf.keras.utils.set_random_seed(args.seed)
    init_w = create_model().get_weights()
    n_mal = 0 if attack == "none" else NUM_MALICIOUS[attack]
    mal = set(np.random.default_rng(args.seed).choice(args.clients, size=n_mal, replace=False).tolist()) if n_mal else set()
    extra, clf, mode = {}, None, "fedavg"
    if arm == "ragnar_pre":
        clf, mode = load_pretrained(), "ragnar"
    elif arm.startswith("ragnar_5r"):
        clf, pool, drawn = bootstrap_classifier(attack, arm, args, init_w)
        mode = "ragnar"
        extra = {"ragnar_bootstrap_pool": pool, "ragnar_bootstrap_drawn": drawn}

    strat = RagnarCoUrb(initial_weights=init_w, schedule=lambda r: (attack, mal), mode=mode, classifier=clf,
                        seed=args.seed, eval_model=eval_model, test_data=test_data,
                        eval_every=args.eval_every, num_rounds=args.rounds)
    t0 = time.time()
    simulate(strat, args, args.rounds)
    return {"scenario": f"{attack}__{arm}", "framework": "CoUrb-2024 (Flower 1.7 + Keras CNN)",
            "attack": attack, "defense": arm, "attack_params": ATTACK_PARAMS.get(attack, {}),
            "num_clients": args.clients, "num_malicious": n_mal, "malicious_ids": sorted(mal),
            "dirichlet_alpha": args.alpha, "dataset_size_per_client": args.dataset_size,
            "rounds": args.rounds, "seed": args.seed, "wall_time_s": time.time() - t0, **extra,
            "round_log": {str(k): v for k, v in strat.round_log.items()}, "history": strat.history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--boot-rounds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dataset-size", type=int, default=1000, help="amostras por cliente (padrão do CoUrb)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="clientes treinando em paralelo (Ray)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    logging.getLogger("flwr").setLevel(logging.WARNING)

    (_, _), (xt, yt) = tf.keras.datasets.cifar10.load_data()
    test_data = ((xt / 255.0).astype(np.float32), yt.reshape(-1))
    eval_model = create_model()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = ([] if args.no_clean else [("none", "fedavg")]) + [(a, arm) for arm in args.arms for a in args.attacks]
    t0 = time.time()
    for i, (attack, arm) in enumerate(jobs, 1):
        path = out / f"{'clean' if attack == 'none' else attack}__{arm}_s{args.seed}.json"
        if path.exists():
            print(f"[{i}/{len(jobs)}] {path.name} já existe — pulando")
            continue
        print(f"\n[{i}/{len(jobs)}] {attack} / {arm} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
        res = run_one(attack, arm, args, test_data, eval_model)
        if attack == "none":
            res["scenario"] = "clean__fedavg"
        path.write_text(json.dumps(res, indent=2))
    print(f"\n=== Concluído em {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    sys.exit(main())
