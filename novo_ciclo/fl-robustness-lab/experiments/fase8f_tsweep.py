#!/usr/bin/env python
"""Varredura do limiar de corroboração T (confirm_tensors) da layer_guard_corr numa combinação da Fase 8F
(modelo/dataset/gatilho), medindo o efeito no MODELO (ASR/MA). Motivação: no VGG19 a corr com T=8 só reduz
(BA 46%) e não contém; a análise offline do diag_log mostra que o atacante é acusado por menos tensores lá
(mediana 16, p10=2, vs 22/14 na ResNet18) → recall cai. Esta varredura testa T menores (mais agressivos).

Nomeia a saída laysel{tag}_{model}_{dataset}__{defense}_T{T}__a0.9_s42.json. Lock atômico por job (sequencial ou
paralelo). O gate (lg_shadow) já tem de existir para o cenário (jobs() da fase8f confirma que o ataque pegou).

Uso: CUDA_VISIBLE_DEVICES=6 python -u experiments/fase8f_tsweep.py --model vgg19gn --dataset cifar10 \
         --trigger badnet_cifar --tvals 2,4,6
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase8f
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
from src.data import datasets

OUT = fase8f.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--defense", default="layer_guard_corr")
    ap.add_argument("--tvals", required=True, help="valores de T separados por virgula, ex.: 2,4,6")
    a = ap.parse_args()
    datasets.set_trigger(a.trigger)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    base = fase8f.jobname(a.model, a.dataset, a.defense, a.trigger)   # ..._{defense}__a0.9_s42
    for T in [int(x) for x in a.tvals.split(",")]:
        name = base.replace(f"__{a.defense}__", f"__{a.defense}_T{T}__")
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
        c = fase8f.cfg(a.model, a.dataset, a.defense, a.trigger)
        c.defense_params = {**(getattr(c, "defense_params", None) or {}), "confirm_tensors": T}
        c.name = name
        print(f"\n=== {name} pid={os.getpid()} T={T} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"), default=str)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
