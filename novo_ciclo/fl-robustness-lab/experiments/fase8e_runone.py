#!/usr/bin/env python
"""Roda UMA defesa do cenário 8E (LaySelFL BadNet, ResNet18-GN/CIFAR) numa SEMENTE dada, SEM refazer o gate
(o gate lg_shadow é produzido pelos runs de semente em fase8e_seed.py). Serve para completar o head-to-head
com rigor: os concorrentes SOTA + sn nas sementes 43/44/45, para média±desvio na tabela principal.
Lock atômico por job. Uso: CUDA_VISIBLE_DEVICES=0 python -u experiments/fase8e_runone.py --seed 43 --defense lasa
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.fase8e as f8e          # import já fixa o gatilho badnet_cifar
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
# 13/11/2026: salva o modelo final de toda corrida (artigo 2)
os.environ.setdefault("FL_SAVE_FINAL", "results/models_final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--defense", required=True)
    a = ap.parse_args()
    f8e.SEED = a.seed
    (f8e.OUT / "raw").mkdir(parents=True, exist_ok=True)
    name = f"laysel_paper__{a.defense}__a{f8e.ALPHA}_s{a.seed}"
    f = f8e.OUT / "raw" / f"{name}.json"
    if f.exists():
        print(f"[skip] {name} já existe", flush=True); return
    lock = f8e.OUT / "raw" / f"{name}.lock"
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
    print(f"\n=== {name} pid={os.getpid()} device={c.device} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
