#!/usr/bin/env python
"""Fase 7 (26/10/2026) — grade completa, pedido do usuário: TODAS as defesas avaliadas nos MESMOS cenários.
Desenho e critério FIXADOS ANTES DE RODAR. Saídas em results/fase7/ (raw/, fila.log, RESULTADO.md).

Grade: 8 formas de agregar (sem defesa = lg_shadow sob ataque e fedavg sem ataque, nossa proposta = layer_guard_sn,
LASA, AlignIns, RFLDefense, FLAME, Multi-Krum, FedSurrogate) × α {0,1; 0,2; 0,3; 0,4; 0,5} × {LP + ruído, LP evasivo,
LP original, sem ataque} × sementes de teste {46, 47, 48} = 480 treinos. Treinos já feitos nas fases anteriores com a
mesma configuração são reaproveitados (fase6.find_name, que procura até a Fase 2); faltavam 302 em 26/10.
Nenhum parâmetro de nenhuma defesa muda em relação às fases anteriores.

Critério por caso = o da Fase 4 (fase6.verdict): ASR <= 10%, acurácia sob ataque >= sem defesa − 3 pp, acurácia
sem ataque >= FedAvg sem ataque − 3 pp; sem defesa com ASR < 20% = inconclusivo.
O relatório dá, por ataque e α, a média das 3 sementes (ASR e acurácia) e quantos casos cada defesa defende.
Não há hipótese nova: a Fase 7 é descritiva (completa a comparação). Ordem da fila: α 0,5, 0,3, 0,1, depois 0,2 e
0,4; dentro de cada α, sem ataque, LP + ruído, LP evasivo, LP original.
Uso: python experiments/fase7.py --one | --report | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase7")
LP, EV, PIX, SH, LG, TEST = fase6.LP, fase6.EV, fase6.PIX, fase6.SH, fase6.LG, fase6.TEST
DEFENSES = (SH, LG, "lasa", "alignins", "rfldefense", "flame", "multi_krum", "fedsurrogate")
ALPHAS = (0.5, 0.3, 0.1, 0.2, 0.4)
ATTACKS = ("none", LP, EV, PIX)
NAMES = {SH: "Sem defesa", LG: "Nossa proposta", "lasa": "LASA", "alignins": "AlignIns", "rfldefense": "RFLDefense",
         "flame": "FLAME", "multi_krum": "Multi-Krum", "fedsurrogate": "FedSurrogate"}
ATK = {LP: "LP + ruído", EV: "LP evasivo", PIX: "LP original", "none": "Sem ataque"}


def dname(attack, d):
    return "fedavg" if (attack == "none" and d == SH) else d


def find_name(name):
    p = OUT / "raw" / f"{name}.json"
    return p if p.exists() else fase6.find_name(name)


def final(attack, defense, alpha, seed):
    f = find_name(f"{attack}__{defense}__a{alpha}_s{seed}")
    if f is None and defense == SH:
        f = find_name(f"{attack}__fedavg__a{alpha}_s{seed}")
    if f is None:
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def verdict(attack, alpha, seed, defense):
    b, o = final(attack, SH, alpha, seed), final(attack, defense, alpha, seed)
    bc, oc = final("none", "fedavg", alpha, seed), final("none", dname("none", defense), alpha, seed)
    if b is None or o is None:
        return None
    if b["asr"] < 20:
        return "inconclusivo"
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return "defende" if ok else "não defende"


def jobs():
    j = []
    for a in ALPHAS:
        for x in ATTACKS:
            for s in TEST:
                for d in DEFENSES:
                    c = fase6.cfg(x, dname(x, d), a, s)
                    if final(x, dname(x, d), a, s) is None:
                        j.append(c)
    return j


def report():
    out = ["# Fase 7 — grade completa (média das sementes de teste 46, 47, 48)", "",
           "Célula: ASR médio (acurácia média) · casos defendidos. Critério: docstring de experiments/fase7.py.", ""]
    for x in (LP, EV, PIX):
        out += [f"## {ATK[x]}", "", "| defesa | " + " | ".join(f"α = {a}" for a in sorted(ALPHAS, reverse=True)) + " |",
                "|---|" + "---|" * len(ALPHAS)]
        for d in DEFENSES:
            row = []
            for a in sorted(ALPHAS, reverse=True):
                r = [final(x, d, a, s) for s in TEST]
                if any(v is None for v in r):
                    row.append(f"{sum(v is not None for v in r)}/3 prontos"); continue
                v = [verdict(x, a, s, d) for s in TEST] if d != SH else []
                asr = sum(t["asr"] for t in r) / 3; acc = sum(t["acc"] for t in r) / 3
                ok = "" if d == SH else f" · {sum(t == 'defende' for t in v)}/{sum(t in ('defende', 'não defende') for t in v)}"
                row.append(f"{asr:.1f} ({acc:.1f}){ok}")
            out.append(f"| {NAMES[d]} | " + " | ".join(row) + " |")
        out.append("")
    out += ["## Sem ataque (acurácia média)", "", "| defesa | " + " | ".join(f"α = {a}" for a in sorted(ALPHAS, reverse=True)) + " |",
            "|---|" + "---|" * len(ALPHAS)]
    for d in DEFENSES:
        r = [[final("none", dname("none", d), a, s) for s in TEST] for a in sorted(ALPHAS, reverse=True)]
        out.append(f"| {NAMES[d]} | " + " | ".join("—" if any(v is None for v in rr) else f"{sum(v['acc'] for v in rr) / 3:.1f}" for rr in r) + " |")
    txt = "\n".join(out) + "\n"
    (OUT / "RESULTADO.md").write_text(txt); print(txt)


def main():
    ap = argparse.ArgumentParser()
    for f in ("--one", "--report", "--missing"):
        ap.add_argument(f, action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    if args.missing:
        j = jobs(); print(len(j)); [print(c.name) for c in j]; return
    for c in jobs():
        if find_name(c.name):
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
        if find_name(c.name):  # outro worker terminou entre montar a lista e pegar o lock
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
