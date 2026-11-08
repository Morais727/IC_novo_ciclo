#!/usr/bin/env python
"""Fase 9B-fill (03/11/2026) — completa as sementes 46,47,48 nos alphas intermediarios (0,4 / 0,3 / 0,2) dos 3
ataques LP (LP+ruido, LP evasivo, LP original). As fases anteriores so rodaram os extremos (alpha 0,5 e 0,1)
nessas sementes, e a 9B (fase9_sementes) so gerou as sementes novas 49,50 nos intermediarios — deixando as
celulas "2/2 parcial" nas tabelas. Isto uniformiza a varredura para 5 sementes (46-50) em todos os alphas.

Mesmo cenario, criterio e NOMES da Fase 9B (experiments/fase9_sementes.py); grava em results/fase9/raw/, para o
`fase9_sementes.py --report` somar automaticamente. O baseline "sem ataque" ja existe nas 5 sementes em todos os
alphas, entao aqui so rodam os 3 ataques. Lock atomico por job -> seguro rodar muitos workers em paralelo.
Uso: python experiments/fase9_fill.py --one | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6, fase9_sementes as f9
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = f9.OUT                       # results/fase9
ATTACKS = (f9.LP, f9.EV, f9.PIX)   # os 3 ataques; "none" ja tem as 5 sementes em todos os alphas
ALPHAS = (0.4, 0.3, 0.2)           # os intermediarios que estavam so com 2 sementes
SEEDS = (46, 47, 48)               # as que faltam (49,50 ja existem via 9B)


def jobs():
    j = []
    for a in ALPHAS:
        for x in ATTACKS:
            for s in SEEDS:
                for d in f9.DEFENSES:
                    dn = f9.dname(x, d)
                    if f9.final(x, dn, a, s) is None:
                        j.append(fase6.cfg(x, dn, a, s))
    return j


def main():
    ap = argparse.ArgumentParser()
    for f in ("--one", "--missing"):
        ap.add_argument(f, action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.missing:
        j = jobs(); print(len(j)); [print(c.name) for c in j]; return
    for c in jobs():
        if f9.find_name(c.name):
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        lock = OUT / "raw" / f"{c.name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0); continue
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True); continue
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        if f9.find_name(c.name):
            lock.unlink(missing_ok=True); continue
        print(f"\n=== {c.name} pid={os.getpid()}", flush=True)
        json.dump(run(c), open(OUT / "raw" / f"{c.name}.json", "w"))
        lock.unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
