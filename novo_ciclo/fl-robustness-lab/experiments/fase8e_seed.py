#!/usr/bin/env python
"""Rigor estatístico — cenário 8E (LaySelFL BadNet, ResNet18-GN/CIFAR, 205 rod.) numa SEMENTE dada.
Roda o gate (lg_shadow) primeiro e, se o ataque pegar (ASR>=20), a nossa família (corr, client, corr-soft),
para média±desvio do resultado-chave em várias sementes (hoje só a 42). Lock atômico por job.

Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase8e_seed.py --seed 43
     (--defenses layer_guard_corr,layer_guard_client,layer_guard_corr_soft é o padrão)
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.fase8e as f8e          # import já fixa o gatilho badnet_cifar (topo do fase8e)
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run


def one(defense):
    name = f"laysel_paper__{defense}__a{f8e.ALPHA}_s{f8e.SEED}"
    f = f8e.OUT / "raw" / f"{name}.json"
    if f.exists():
        return f8e.final(defense)
    lock = f8e.OUT / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {name}", flush=True); return None
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    c = f8e.cfg(defense)
    print(f"\n=== {name} pid={os.getpid()} device={c.device} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)
    return f8e.final(defense)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--defenses", default="layer_guard_corr,layer_guard_client,layer_guard_corr_soft")
    a = ap.parse_args()
    f8e.SEED = a.seed
    (f8e.OUT / "raw").mkdir(parents=True, exist_ok=True)
    g = one("lg_shadow")                   # gate primeiro
    if g and g["asr"] >= 20:
        print(f"[seed {a.seed}] gate PEGOU (ASR {g['asr']:.1f}%); rodando defesas", flush=True)
        for d in a.defenses.split(","):
            if d and d != "lg_shadow":
                one(d)
    else:
        print(f"[seed {a.seed}] gate NÃO pegou (gate={g}) — defesas puladas", flush=True)


if __name__ == "__main__":
    main()
