#!/usr/bin/env python
"""Fase 8D (29/10/2026) — a LaySelFL pega num cenário mais perto do artigo? Desenho e critério FIXADOS ANTES DE RODAR.

Motivo: no portão da 8A (2 atacantes de 20, ataque desde a rodada 1) a LaySelFL ficou em 1,5% e parou ali. O artigo
(Li et al., CVPR 2026) usa 6 atacantes de 19 e começa a atacar na rodada 80 de 205, com o modelo já treinado.
Aqui só mudam essas duas coisas; o resto é o cenário da Fase 7 (CIFAR-10, SmallCNN, 20 clientes, 40 rodadas) e a
implementação é a mesma da 8A (src/attacks/recent.py::laysel_attack, β 20, corte 0,05, fração envenenada 0,2).
  - 6 atacantes de 20;
  - start_round 16 (80/205 ≈ 0,39 das rodadas; antes disso os atacantes treinam como honestos).
Etapa 1 — portão: sem defesa (lg_shadow), α 0,5, semente 42. Pega se o sucesso final for >= 20%.
  Se não pegar: registra que a LaySelFL não pega nem perto do cenário do artigo e PARA. Nada é reajustado.
Etapa 2 — só se pegar: as 8 formas de agregar da Fase 7 × α 0,5 × sementes 46, 47, 48 (24 treinos), critério por caso
  igual ao da Fase 7 (fase8a.verdict). Sem ataque reaproveitado da Fase 7.
Uso: python experiments/fase8d.py --one | --report | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6, fase9_sementes as f9
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase8d")
SH = fase6.SH
N_MAL, START, DEV, TEST = 6, 16, 42, (46, 47, 48)


def cfg(defense, alpha, seed):
    c = fase6.cfg("lp_full", defense, alpha, seed)
    c.attack, c.num_malicious = "laysel", N_MAL
    c.attack_params = {"target_label": c.attack_params.get("target_label", 0), "start_round": START}
    c.name = f"laysel6_r{START}__{defense}__a{alpha}_s{seed}"
    return c


def final(defense, alpha, seed, attack="laysel"):
    if attack == "none":
        f = f9.find_name(f"none__{defense}__a{alpha}_s{seed}")
    else:
        f = OUT / "raw" / f"laysel6_r{START}__{defense}__a{alpha}_s{seed}.json"
        f = f if f.exists() else None
    if f is None:
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def gate():
    r = final(SH, 0.5, DEV)
    return None if r is None else r["asr"] >= 20


def verdict(seed, defense):
    b, o = final(SH, 0.5, seed), final(defense, 0.5, seed)
    bc, oc = final("fedavg", 0.5, seed, "none"), final(f9.dname("none", defense), 0.5, seed, "none")
    if b is None or o is None:
        return None
    if b["asr"] < 20:
        return "inconclusivo"
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return "defende" if ok else "não defende"


def jobs():
    j = [cfg(SH, 0.5, DEV)] if final(SH, 0.5, DEV) is None else []
    if gate():
        j += [cfg(d, 0.5, s) for s in TEST for d in f9.DEFENSES if final(d, 0.5, s) is None]
    return j


def report():
    r = final(SH, 0.5, DEV)
    out = ["# Fase 8D — LaySelFL com 6 atacantes de 20, ataque a partir da rodada 16", "",
           "Portão (sem defesa, α 0,5, semente 42; pega se >= 20%): " + ("ainda não rodou" if r is None else
           f"sucesso {r['asr']:.1f}%, acurácia {r['acc']:.1f}% → {'pegou' if r['asr'] >= 20 else 'NÃO pegou (para aqui)'}"), ""]
    if gate():
        out += ["| defesa | sucesso médio (acurácia) | casos defendidos |", "|---|---|---|"]
        for d in f9.DEFENSES:
            rs = [final(d, 0.5, s) for s in TEST]
            if any(v is None for v in rs):
                out.append(f"| {f9.NAMES[d]} | {sum(v is not None for v in rs)}/3 prontos | |"); continue
            v = [verdict(s, d) for s in TEST] if d != SH else []
            out.append(f"| {f9.NAMES[d]} | {sum(t['asr'] for t in rs) / 3:.1f} ({sum(t['acc'] for t in rs) / 3:.1f}) | "
                       + ("" if d == SH else f"{sum(t == 'defende' for t in v)}/{sum(t in ('defende', 'não defende') for t in v)}") + " |")
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
        f = OUT / "raw" / f"{c.name}.json"
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
        if f.exists():
            lock.unlink(missing_ok=True); continue
        print(f"\n=== {c.name} pid={os.getpid()} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"))
        lock.unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
