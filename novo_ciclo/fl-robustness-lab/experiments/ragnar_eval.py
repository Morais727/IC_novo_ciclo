#!/usr/bin/env python
"""Avaliação do RAGNAR (método proposto) contra todos os ataques implementados.

Para cada ataque roda três braços, com o mesmo protocolo e semente:
  <ataque>__fedavg         sem defesa — mede o dano do ataque
  <ataque>__ragnar_unseen  RAGNAR com a família do ataque EXCLUÍDA do bootstrap
                           (generalização a ataque não visto — número honesto)
  <ataque>__ragnar_seen    RAGNAR com bootstrap SÓ com o ataque avaliado
                           (limite superior do método) — ainda não executado
  <ataque>__ragnar_pool_todos  (opcional) bootstrap com TODOS os ataques, inclusive o avaliado.
                           As execuções de 28/09 antes rotuladas "seen" eram este braço
mais `clean__fedavg` (sem ataque) como teto de acurácia.

Parâmetros de ataque iguais aos de scenarios/*.yaml. `sign_flip` (scale=1) é
omitido por padrão: gera exatamente o mesmo vetor que `ipm` (epsilon=1).

Execuções já concluídas (JSON em --out-dir) são puladas — retomável.

Uso:
    python experiments/ragnar_eval.py                        # semente 42
    python experiments/ragnar_eval.py --seeds 42 43 44
    python experiments/ragnar_eval.py --arms fedavg ragnar_unseen
    python experiments/aggregate_results.py --results-dir results/ragnar_eval
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ScenarioConfig
from src.federated.simulate import run

ATTACKS = {
    "gaussian_noise": dict(num_malicious=4, attack_params={"std": 5.0}),
    "ipm": dict(num_malicious=4, attack_params={"epsilon": 1.0}),
    "lie": dict(num_malicious=4, attack_params={"z": 1.5}),
    "minmax": dict(num_malicious=4, attack_params={"perturbation": "std"}),
    "minsum": dict(num_malicious=4, attack_params={"perturbation": "std"}),
    "label_flip": dict(num_malicious=4, attack_params={"mode": "fixed", "source": 0, "target": 2}),
    "backdoor_pixel": dict(num_malicious=2, attack_params={"target_label": 0, "poison_fraction": 0.5}),
    "backdoor_pgd": dict(num_malicious=2, attack_params={"target_label": 0, "poison_fraction": 0.5,
                                                          "scale": 3.0, "norm_multiplier": 1.0}),
}
# ataques recentes da revisão (2020–2024), src/attacks/recent.py
RECENT_ATTACKS = {
    "mimic": dict(num_malicious=4, attack_params={}),
    "dba": dict(num_malicious=4, attack_params={"target_label": 0, "poison_fraction": 0.5}),
    "lp": dict(num_malicious=2, attack_params={"target_label": 0, "poison_fraction": 0.5, "tau": 0.8}),
    "lf": dict(num_malicious=4, attack_params={"scale": 1.0, "tau": 0.8, "max_layers": 2}),
    "edge_case_pgd": dict(num_malicious=2, attack_params={"target_label": 9, "source": 0, "edge_fraction": 0.05,
                                                           "clean_ratio": 1.0, "scale": 3.0, "norm_multiplier": 1.0}),
}
ATTACKS.update(RECENT_ATTACKS)
OPTIONAL_ATTACKS = {
    "sign_flip": dict(num_malicious=4, attack_params={"scale": 1.0}),
    # LP completo (código dos autores): lambda=cnn_scale=0.5, tau=mode10_tau=0.95, Multi-Krum simulado
    "lp_full": dict(num_malicious=2, attack_params={"target_label": 0, "poison_fraction": 0.5, "tau": 0.95,
                                                     "lambda": 0.5, "simulate_defense": "multi_krum"}),
    # variante evasiva do LP completo: base = delta limpo de cada atacante (não idênticos fora de L*)
    "lp_full_indiv": dict(num_malicious=2, attack_params={"target_label": 0, "poison_fraction": 0.5, "tau": 0.95,
                                                           "lambda": 0.5, "simulate_defense": "multi_krum",
                                                           "individual": True}),
}

OFFLINE_DATASET = "results/signals_offline/raw"

ARMS = {
    "fedavg": dict(defense="fedavg", defense_params={}),
    # Multi-Krum (Blanchard et al., 2017): a defesa que o artigo do LP aponta como a mais forte contra ele
    "multi_krum": dict(defense="multi_krum", defense_params={}),
    # FLAME (Nguyen et al., 2022) como no repositório do LP: barra o LP lá, mas aceita só ~66% dos honestos
    "flame": dict(defense="flame", defense_params={}),
    "ragnar_unseen": dict(defense="ragnar", defense_params={"n_estimators": 60, "max_depth": 3,
                                                            "bootstrap_exclude_eval_attack": True}),
    "ragnar_seen": dict(defense="ragnar", defense_params={"n_estimators": 60, "max_depth": 3,
                                                          "bootstrap_only_eval_attack": True}),
    "ragnar_pool_todos": dict(defense="ragnar", defense_params={"n_estimators": 60, "max_depth": 3,
                                                                "bootstrap_exclude_eval_attack": False}),
    # RAGNAR fiel ao original: classificador treinado uma vez com simulações prévias completas
    # (outra semente), congelado, atuando a partir do round 2
    "ragnar_fiel_nv": dict(defense="ragnar", defense_params={"n_estimators": 60, "max_depth": 3,
                                                             "offline_dataset": OFFLINE_DATASET,
                                                             "bootstrap_exclude_eval_attack": True}),
    "ragnar_fiel_v": dict(defense="ragnar", defense_params={"n_estimators": 60, "max_depth": 3,
                                                            "offline_dataset": OFFLINE_DATASET,
                                                            "bootstrap_exclude_eval_attack": False}),
}
DEFAULT_ARMS = ["fedavg", "ragnar_unseen", "ragnar_seen"]

COMMON = dict(num_clients=20, dirichlet_alpha=0.5, rounds=40, local_epochs=1, batch_size=32,
              lr=0.01, server_holdout_size=100, eval_every=5, backdoor_target_label=0)


def build_jobs(attacks, arms, seeds, overrides):
    jobs = []
    for seed in seeds:
        jobs.append(ScenarioConfig(name="clean__fedavg", attack="none", num_malicious=0,
                                   defense="fedavg", seed=seed, **{**COMMON, **overrides}))
    # braços na ordem pedida, varrendo todos os ataques em cada um — resultados parciais já comparáveis
    for arm in arms:
        for seed in seeds:
            for attack in attacks:
                spec = {**ATTACKS, **OPTIONAL_ATTACKS}[attack]
                jobs.append(ScenarioConfig(
                    name=f"{attack}__{arm}", attack=attack, num_malicious=spec["num_malicious"],
                    attack_params=dict(spec["attack_params"]), seed=seed,
                    defense=ARMS[arm]["defense"], defense_params=dict(ARMS[arm]["defense_params"]),
                    **{**COMMON, **overrides}))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS),
                        choices=list(ATTACKS) + list(OPTIONAL_ATTACKS))
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Sobrescreve dirichlet_alpha (padrão 0.5)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", default="results/ragnar_eval")
    args = parser.parse_args()

    overrides = {}
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.alpha is not None:
        overrides["dirichlet_alpha"] = args.alpha
    if args.quick:
        overrides["quick"] = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.attacks, args.arms, args.seeds, overrides)
    print(f"{len(jobs)} execuções -> {out_dir}/")

    failures = []
    t0 = time.time()
    for i, cfg in enumerate(jobs, 1):
        out = out_dir / f"{cfg.name}_s{cfg.seed}{'_quick' if cfg.quick else ''}.json"
        if out.exists():
            print(f"[{i}/{len(jobs)}] {out.name} já existe — pulando")
            continue
        print(f"\n[{i}/{len(jobs)}] {cfg.name} seed={cfg.seed} (decorrido {(time.time() - t0) / 60:.1f} min)",
              flush=True)
        try:
            result = run(cfg)
        except Exception:
            traceback.print_exc()
            failures.append(out.name)
            continue
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n=== Concluído em {(time.time() - t0) / 60:.1f} min — {len(failures)} falha(s) ===")
    if failures:
        print("Falhas:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
