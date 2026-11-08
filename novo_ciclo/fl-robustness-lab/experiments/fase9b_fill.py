#!/usr/bin/env python
"""Preenche buracos da grade de 5 sementes (46-50) das Fases 7/9B (ver experiments/fase9_sementes.py).

Varre todas as combinações (ataque × defesa × α × semente) da grade e roda, via lock atômico, as cujo arquivo
AINDA NÃO existe (o fase9_sementes.jobs() só olhava as sementes NEW=49,50; este cobre as 5). Hoje falta apenas o
baseline sem defesa (lg_shadow) do LP evasivo em α=0,1, sementes 46/47/48 — sem ele, a coluna evasivo α0,1 fica
sem baseline em 3 das 5 sementes e os denominadores da tabela colapsam.

Cenário (herdado de fase6.cfg): SmallCNN / CIFAR-10 / 40 rodadas / 20 clientes / 2 maliciosos — leve, roda em CPU.
Grava em results/fase9/raw/ (onde fase9_sementes.find_name procura primeiro).
Uso: python experiments/fase9b_fill.py --missing | --one | (sem flag roda todos em sequência)
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6, fase9_sementes as F
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase9/raw")
TEST = (46, 47, 48, 49, 50)


def missing():
    j = []
    for a in F.ALPHAS:
        for x in F.ATTACKS:
            for s in TEST:
                for d in F.DEFENSES:
                    nm = F.dname(x, d)
                    if F.final(x, nm, a, s) is None:
                        j.append((x, nm, a, s))
    # sem duplicatas (dname colapsa SH->fedavg só em none), preservando ordem
    seen, out = set(), []
    for t in j:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    for f in ("--one", "--missing"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jb = missing()
    if a.missing:
        print(len(jb))
        for t in jb:
            print(t)
        return
    for (x, nm, al, s) in jb:
        name = f"{x}__{nm}__a{al}_s{s}"
        f = OUT / f"{name}.json"
        if f.exists():
            continue
        while not mem_ok():
            time.sleep(30)
        lock = OUT / f"{name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0); continue
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True); continue
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        if f.exists():
            lock.unlink(missing_ok=True); continue
        c = fase6.cfg(x, nm, al, s)
        print(f"=== {name} pid={os.getpid()} model={c.model} rounds={c.rounds}", flush=True)
        json.dump(run(c), open(f, "w"))
        lock.unlink(missing_ok=True)
        if a.one:
            return
    if a.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
