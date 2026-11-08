#!/usr/bin/env python
"""CLI: executa vários cenários x sementes em sequência, pulando execuções já
concluídas (JSON existente em results/). Pode ser interrompido e retomado.

Uso:
    python experiments/run_matrix.py                              # todos os scenarios/*.yaml, sementes 42 43 44
    python experiments/run_matrix.py --seeds 42                   # uma semente só
    python experiments/run_matrix.py --only minmax_krum_labelskew minmax_boba_labelskew
    python experiments/run_matrix.py --quick                      # checagem rápida da matriz inteira
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_scenario import load_config, default_out_path
from src.federated.simulate import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--only", nargs="*", default=None, help="Nomes de cenário (sem .yaml) a executar")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reexecuta mesmo se o JSON já existir")
    args = parser.parse_args()

    paths = sorted(Path(args.scenarios_dir).glob("*.yaml"))
    if args.only:
        paths = [p for p in paths if p.stem in set(args.only)]
    jobs = [(p, s) for p in paths for s in args.seeds]
    print(f"{len(jobs)} execuções ({len(paths)} cenários x {len(args.seeds)} sementes)")

    failures = []
    t0 = time.time()
    for i, (path, seed) in enumerate(jobs, 1):
        cfg = load_config(str(path), args.quick, seed)
        out = default_out_path(cfg)
        if out.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] {out.name} já existe — pulando")
            continue
        print(f"\n[{i}/{len(jobs)}] {cfg.name} seed={seed} (decorrido {(time.time() - t0) / 60:.1f} min)")
        try:
            result = run(cfg)
        except Exception:
            traceback.print_exc()
            failures.append(out.name)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {out} ({result['sec_per_round']:.1f}s/round)")

    print(f"\n=== Matriz concluída em {(time.time() - t0) / 60:.1f} min — {len(failures)} falha(s) ===")
    if failures:
        print("Falhas:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
