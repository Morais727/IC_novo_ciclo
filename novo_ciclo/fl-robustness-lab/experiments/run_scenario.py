#!/usr/bin/env python
"""CLI: executa um cenário de avaliação definido em YAML.

Uso:
    python experiments/run_scenario.py --scenario scenarios/boba_vs_byzantine_labelskew.yaml
    python experiments/run_scenario.py --scenario scenarios/... --quick   # override rápido p/ validação
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.config import ScenarioConfig
from src.federated.simulate import run


def load_config(path: str, quick_override: bool, seed: int | None = None) -> ScenarioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ScenarioConfig(**raw)
    if quick_override:
        cfg.quick = True
    if seed is not None:
        cfg.seed = seed
    return cfg


def default_out_path(cfg: ScenarioConfig) -> Path:
    suffix = "_quick" if cfg.quick else ""
    return Path("results") / f"{cfg.name}_s{cfg.seed}{suffix}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Caminho do YAML de cenário")
    parser.add_argument("--quick", action="store_true", help="Força modo rápido (subamostragem)")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescreve a semente do YAML")
    parser.add_argument("--out", default=None,
                        help="Caminho do JSON de saída (default: results/<name>_s<seed>[_quick].json)")
    args = parser.parse_args()

    cfg = load_config(args.scenario, args.quick, args.seed)
    print(f"=== Cenário: {cfg.name} | ataque={cfg.attack} | defesa={cfg.defense} | "
          f"clientes={cfg.num_clients} (mal={cfg.num_malicious}) | alpha={cfg.dirichlet_alpha} | "
          f"rounds={cfg.rounds} | quick={cfg.quick} ===")

    result = run(cfg)

    out_path = Path(args.out) if args.out else default_out_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Resultados salvos em {out_path}")


if __name__ == "__main__":
    main()
