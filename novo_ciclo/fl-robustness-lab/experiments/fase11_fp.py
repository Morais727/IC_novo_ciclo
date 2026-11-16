#!/usr/bin/env python
"""Fase 11 (13/11/2026) — FALSOS POSITIVOS SEM ATAQUE (entregável M9; artigo 1).

Pergunta: a família de corroboração purga benignos (precisão 0,82 em T=8 sob ataque). Quanto isso custa de acurácia
quando NÃO há ataque? Desenho FIXADO ANTES DE RODAR:
  - mesmo cenário do 8E (ResNet18-GN, CIFAR-10, 20 nós, 205 rodadas, LR 0,01 -> 0,001), mesmos 6 clientes sorteados
    como "maliciosos", mas a janela de ataque começa DEPOIS da última rodada (start_round=10**6): todos honestos o
    tempo todo, e a partição de dados é idêntica à do 8E;
  - defesas: lg_shadow (= sem defesa, referência), layer_guard_corr, layer_guard_corr_soft, layer_guard_client;
  - α 0,9 (cenário do LaySelFL) e α 0,1 (não-IID forte, onde benignos são mais heterogêneos).
Métrica: acurácia final e taxa de clientes descartados por rodada (do JSON: accepted / kept_fraction).
Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase11_fp.py --defense layer_guard_client --alpha 0.9 --seed 42
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.fase8e as f8e          # fixa o gatilho badnet_cifar
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase11")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    f8e.SEED, f8e.ALPHA = a.seed, a.alpha
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    name = f"noattack__{a.defense}__a{a.alpha}_s{a.seed}"
    f = OUT / "raw" / f"{name}.json"
    if f.exists():
        print(f"[skip] {name} já existe", flush=True); return
    lock = f.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {name}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    c = f8e.cfg(a.defense)
    c.attack_params["start_round"] = 10 ** 6       # nunca ataca: os 6 "maliciosos" ficam honestos
    c.attack_params["end_round"] = 10 ** 6
    c.name = name
    os.environ.setdefault("FL_SAVE_FINAL", "results/models_final")
    print(f"\n=== {name} pid={os.getpid()} (SEM ATAQUE) ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
