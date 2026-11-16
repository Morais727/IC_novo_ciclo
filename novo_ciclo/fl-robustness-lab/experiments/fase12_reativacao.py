#!/usr/bin/env python
"""Fase 12 (13/11/2026) — REATIVAÇÃO e DORMÊNCIA do backdoor residual (artigo 2, experimento 2.2).

Pergunta: um modelo final com ASR residual guarda um backdoor "adormecido"? Desenho FIXADO ANTES DE RODAR, cenário 8E
(ResNet18-GN, CIFAR-10, 20 nós, α 0,9, mesma semente do modelo de origem => mesma partição e mesmos 6 atacantes):
  - dormencia : FL SEM ataque e SEM defesa (FedAvg) por 20 rodadas a partir do modelo final, LR 0,001. O ASR cai ao
                piso (backdoor frágil) ou persiste (backdoor gravado)?
  - reativacao: os 6 atacantes voltam (LaySelFL, mesmos parâmetros) SEM defesa por 20 rodadas. Quantas rodadas até
                ASR >= 50%? Comparado com o MESMO procedimento a partir do modelo limpo da rodada 79 (controle).
Avalia a cada 2 rodadas. Saída results/fase12/raw/<origem>__<modo>.json.
Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase12_reativacao.py --model results/fase10/models/X.pt --seed 44 --mode reativacao
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.fase8e as f8e          # fixa o gatilho badnet_cifar
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["dormencia", "reativacao"], required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="só para teste de fumaça")
    a = ap.parse_args()
    f8e.SEED = a.seed
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    name = f"{Path(a.model).stem}__{a.mode}"
    f = OUT / "raw" / f"{name}.json"
    if f.exists() and not a.quick:
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
    c = f8e.cfg(f8e.SH)                       # sem defesa (lg_shadow = FedAvg + diagnóstico)
    c.rounds, c.eval_every = a.rounds, 2
    c.lr_schedule = [[a.rounds, 0.001]]       # fase final do treino original
    on = a.mode == "reativacao"
    c.attack_params["start_round"] = 1 if on else 10 ** 6
    c.attack_params["end_round"] = a.rounds if on else 10 ** 6
    c.name = name
    if a.quick:
        c.quick, c.rounds = True, 2
    os.environ["FL_INIT_VECTOR"] = a.model
    print(f"\n=== {name} pid={os.getpid()} modelo={a.model} ===", flush=True)
    res = run(c)
    if not a.quick:
        json.dump(res, open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
