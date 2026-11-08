#!/usr/bin/env python
"""Reprodução semeada do LP (Zhuang et al., ICLR 2024) no repositório original: linha FLAME da Tab. 2
(CNN, Fashion-MNIST não-IID, hiperparâmetros do Apêndice A-7), uma execução por semente.

O repositório lê --seed mas nunca semeia os geradores; aqui um wrapper semeia random, numpy e torch
antes de executar main_fed.py, sem alterar o código dos autores. Cada execução tem:
  - faulthandler no SIGUSR1 (pilha Python em <save>/pilha.txt sob demanda);
  - vigia: sem novo round por 15 min -> grava a pilha, encerra a execução e segue para a próxima.

Uso (venv do repositório original, que reaproveita o torch do laboratório):
    python experiments/repro_lp_seeded.py --seeds 1 2 3
Saídas: <repo>/save/repro_seed<S>/ e resumo em results/repro_lp_seeded.md
"""
import argparse
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path.home() / "Documentos/IC/Poisoning_Backdoor-critical_Layers_Attack-main"
ARGS = ("--dataset fashion_mnist --model rlr_mnist --lr 0.01 --malicious 0.1 --poison_frac 0.5 --local_ep 2 "
        "--local_bs 64 --attack_begin 0 --defence flame --epochs 200 --attack_label 5 --attack_goal -1 "
        "--trigger square --triggerX 23 --triggerY 23 --gpu 0 --iid 0 --attack lp_attack")
WRAPPER = """
import faulthandler, signal, random, runpy, sys
import numpy as np, torch
faulthandler.register(signal.SIGUSR1, file=open('{save}/pilha.txt', 'w'), all_threads=True)
random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed}); torch.cuda.manual_seed_all({seed})
sys.argv = ['main_fed.py'] + {argv!r}
runpy.run_path('main_fed.py', run_name='__main__')
"""
STALL_S = 15 * 60


def read_acc(save: Path):
    files = list(save.glob("accuracy_file*.txt"))
    if not files:
        return [], [], {}
    s = files[0].read_text()
    get = lambda k: eval(m.group(1)) if (m := re.search(rf"{k}=(\[.*?\])", s)) else []
    prop = dict(re.findall(r"proportion of (\w+) are selected:([\d.]+)", s))
    return get("main_task_accuracy"), get("backdoor_accuracy"), prop


def run_seed(seed: int, python: str) -> dict:
    save = REPO / f"save/repro_seed{seed}"
    save.mkdir(parents=True, exist_ok=True)
    rel = f"save/repro_seed{seed}"
    code = WRAPPER.format(save=rel, seed=seed, argv=(ARGS + f" --seed {seed} --save {rel}").split())
    log = open(REPO / f"save/repro_seed{seed}.log", "w")
    proc = subprocess.Popen([python, "-u", "-c", code], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    last_n, last_t, status = -1, time.time(), "ok"
    while proc.poll() is None:
        time.sleep(30)
        n = len(read_acc(save)[0])
        if n != last_n:
            last_n, last_t = n, time.time()
            print(f"  semente {seed}: {n} rounds", flush=True)
        elif time.time() - last_t > STALL_S:
            proc.send_signal(signal.SIGUSR1); time.sleep(10); proc.kill()
            status = f"travou no round {n} (pilha em {rel}/pilha.txt)"
            break
    ma, ba, prop = read_acc(save)
    k = max(1, len(ba) // 10)
    return {"semente": seed, "status": status, "rounds": len(ba), "ma_final": ma[-1] if ma else float("nan"),
            "bsr_final": float(np.mean(ba[-k:])) if ba else float("nan"), "bsr_melhor": max(ba) if ba else float("nan"),
            "mar": float(prop.get("malicious", "nan")), "bar": float(prop.get("benign", "nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--python", default=str(REPO / ".venv-lp/bin/python"))
    ap.add_argument("--out", default="results/repro_lp_seeded.md")
    args = ap.parse_args()
    rows = []
    for s in args.seeds:
        print(f"=== semente {s}", flush=True)
        rows.append(run_seed(s, args.python))
        lines = ["# Reprodução semeada do LP sob FLAME (CNN, Fashion-MNIST não-IID, 200 rounds)", "",
                 "Artigo, Tab. 2: BSR médio 74,9 ± 2,7%, melhor 84,3 ± 3,1%, acurácia 87,1%.", "",
                 "| semente | status | rounds | BSR final (últimos 10%) | melhor BSR | acurácia final | maliciosos aceitos | honestos aceitos |",
                 "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['semente']} | {r['status']} | {r['rounds']} | {r['bsr_final']:.2f}% | {r['bsr_melhor']:.2f}% | "
                  f"{r['ma_final']:.1f}% | {r['mar']:.1%} | {r['bar']:.1%} |" for r in rows]
        Path(args.out).write_text("\n".join(lines) + "\n")
        print("\n".join(lines[-len(rows):]), flush=True)


if __name__ == "__main__":
    main()
