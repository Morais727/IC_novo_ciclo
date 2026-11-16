#!/usr/bin/env python
"""Varredura do limiar de corroboração T (confirm_tensors) da layer_guard_corr NO CENÁRIO 8E (BadNet,
ResNet18-GN/CIFAR-10, 205 rodadas, ataque 80-200), medindo o efeito no MODELO (ASR/MA), não só na detecção.

A curva precisão/recall/F1 da detecção por T foi feita offline (results/fase8e/tsweep_corr.json, do diag_log do
portão sem-defesa). Aqui cada T re-treina a corr e registra ASR/MA finais, fechando o trade-off detecção<->modelo.
O T=8 (padrão) já existe em laysel_paper__layer_guard_corr. Este driver roda os demais, nomeando a saida
laysel_paper__layer_guard_corr_T{T}__a0.9_s42.json. Lock por job (seguro rodar sequencial ou em paralelo).

Uso: CUDA_VISIBLE_DEVICES=5 python experiments/fase8e_tsweep.py --tvals 4,11,14
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase8e            # importa já fixa o gatilho badnet_cifar (linha de topo do fase8e)
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = fase8e.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tvals", required=True, help="valores de T separados por virgula, ex.: 4,11,14")
    ap.add_argument("--defense", default="layer_guard_corr")
    ap.add_argument("--seed", type=int, default=fase8e.SEED, help="13/11: varredura de T em outras sementes")
    a = ap.parse_args()
    fase8e.SEED = a.seed
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    for T in [int(x) for x in a.tvals.split(",")]:
        name = f"laysel_paper__{a.defense}_T{T}__a{fase8e.ALPHA}_s{fase8e.SEED}"
        f = OUT / "raw" / f"{name}.json"
        if f.exists():
            print(f"[skip] {name} ja existe", flush=True); continue
        lock = OUT / "raw" / f"{name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {name}", flush=True); continue
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        while not mem_ok():
            time.sleep(30)
        c = fase8e.cfg(a.defense)
        c.defense_params = {**(getattr(c, "defense_params", None) or {}), "confirm_tensors": T}
        c.name = name
        print(f"\n=== {name} pid={os.getpid()} T={T} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"), default=str)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
