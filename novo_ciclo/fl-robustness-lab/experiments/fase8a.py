#!/usr/bin/env python
"""Fase 8A (28/10/2026) — os ataques de 2026 contra todas as formas de agregar. Desenho, escolhas de implementação
e critério FIXADOS ANTES DE RODAR (results/fase8/PLANO.md, seção 8A; implementação em src/attacks/recent.py).

Ataques (sem código público; implementados a partir dos textos, escolhas não especificadas no docstring de cada um):
  lsa     LSA, Foroughi et al., IEEE ICC 2026 (arXiv 2602.15161): substituição de camadas (tau 0,8) a cada 5 rodadas,
          ajuste fino só das camadas críticas nos dados envenenados, lambda 1.
  laysel  LaySelFL, Li et al., CVPR 2026: camadas sensíveis por distância L1 (TopK, k = nº de tensores − 2), treino
          alternado em P = 2 grupos com perda + 20·||theta − servidor||², corte de ±0,05 nas camadas normais.
Cenário = o da Fase 7 (fase6.cfg): CIFAR-10, SmallCNN, 20 clientes, 2 atacantes, 40 rodadas, gatilho 4×4, alvo 0.

Etapa 1 — portão (semente 42, de desenvolvimento): cada ataque sem defesa (lg_shadow), α 0,5. O ataque "pega" se o
  sucesso final for >= 20%. Se não pegar, registra e PARA para esse ataque — nada é reajustado olhando a nossa
  defesa. O portão roda no processador (a placa de vídeo estava ocupada pela 8C); a grade roda na placa.
Etapa 2 — grade (só ataques que passaram): 8 formas de agregar (as da Fase 7) × α {0,5; 0,3; 0,1} × sementes de
  teste 46–50 (as 5 da Fase 9B) = 120 treinos por ataque. Sem ataque é reaproveitado da Fase 7 / 9B.
Critério por caso = o da Fase 7 (ASR <= 10%, acurácia >= sem defesa − 3 pp, sem ataque >= FedAvg − 3 pp; sem
  defesa com ASR < 20% = inconclusivo).
Expectativa declarada antes: a nossa proposta procura atacantes com a MESMA alteração nas mesmas camadas. A LSA envia
  o mesmo vetor pelos 2 atacantes (como o LP original) → deve ser pega. A LaySelFL treina cada atacante nos próprios
  dados, com a alteração presa perto do servidor → pode passar; se passar, vai para o artigo como limite.
Uso: python experiments/fase8a.py --one [--cpu] [--gate-only] | --report | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6, fase9_sementes as f9
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase8a")
ATTACKS = ("lsa", "laysel")
ATK = {"lsa": "LSA (ICC 2026)", "laysel": "LaySelFL (CVPR 2026)"}
SH, LG = fase6.SH, fase6.LG
DEFENSES, NAMES = f9.DEFENSES, f9.NAMES
ALPHAS, TEST, DEV = (0.5, 0.4, 0.3, 0.2, 0.1), f9.TEST, 42   # 5 níveis, padronizado com as Fases 7/9B (fase9_sementes)


def cfg(attack, defense, alpha, seed):
    c = fase6.cfg("lp_full", defense, alpha, seed)          # 2 atacantes e o resto do cenário da Fase 7
    c.attack, c.attack_params = attack, {"target_label": c.attack_params.get("target_label", 0)}
    c.name = f"{attack}__{defense}__a{alpha}_s{seed}"
    return c


def final(attack, defense, alpha, seed):
    if attack == "none":
        f = f9.find_name(f"none__{defense}__a{alpha}_s{seed}")
    else:
        f = OUT / "raw" / f"{attack}__{defense}__a{alpha}_s{seed}.json"
        f = f if f.exists() else None
    if f is None:
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def gate(attack):
    r = final(attack, SH, 0.5, DEV)
    return None if r is None else r["asr"] >= 20


def verdict(attack, alpha, seed, defense):
    b, o = final(attack, SH, alpha, seed), final(attack, defense, alpha, seed)
    bc, oc = final("none", "fedavg", alpha, seed), final("none", f9.dname("none", defense), alpha, seed)
    if b is None or o is None:
        return None
    if b["asr"] < 20:
        return "inconclusivo"
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return "defende" if ok else "não defende"


def jobs(gate_only=False):
    j = [cfg(x, SH, 0.5, DEV) for x in ATTACKS if final(x, SH, 0.5, DEV) is None]
    if gate_only:
        return j
    for a in ALPHAS:
        for s in TEST:
            for x in ATTACKS:
                if gate(x):
                    j += [cfg(x, d, a, s) for d in DEFENSES if final(x, d, a, s) is None]
    return j


def report():
    out = ["# Fase 8A — ataques de 2026 (LSA, LaySelFL)", "", "## Portão (semente 42, sem defesa, α 0,5; pega se ASR >= 20%)", ""]
    for x in ATTACKS:
        r = final(x, SH, 0.5, DEV)
        out.append(f"- {ATK[x]}: " + ("ainda não rodou" if r is None else
                   f"ASR {r['asr']:.1f}%, acurácia {r['acc']:.1f}% → {'pegou' if r['asr'] >= 20 else 'NÃO pegou (para aqui)'}"))
    out += ["", f"Célula: ASR médio (acurácia média) · casos defendidos, sementes {', '.join(map(str, TEST))}.", ""]
    for x in ATTACKS:
        if not gate(x):
            continue
        out += [f"## {ATK[x]}", "", "| defesa | " + " | ".join(f"α = {a}" for a in ALPHAS) + " |", "|---|" + "---|" * len(ALPHAS)]
        for d in DEFENSES:
            row = []
            for a in ALPHAS:
                r = [final(x, d, a, s) for s in TEST]
                if any(v is None for v in r):
                    row.append(f"{sum(v is not None for v in r)}/{len(TEST)} prontos"); continue
                v = [verdict(x, a, s, d) for s in TEST] if d != SH else []
                asr, acc = sum(t["asr"] for t in r) / len(r), sum(t["acc"] for t in r) / len(r)
                ok = "" if d == SH else f" · {sum(t == 'defende' for t in v)}/{sum(t in ('defende', 'não defende') for t in v)}"
                row.append(f"{asr:.1f} ({acc:.1f}){ok}")
            out.append(f"| {NAMES[d]} | " + " | ".join(row) + " |")
        out.append("")
    txt = "\n".join(out) + "\n"
    (OUT / "RESULTADO.md").write_text(txt); print(txt)


def main():
    ap = argparse.ArgumentParser()
    for f in ("--one", "--report", "--missing", "--cpu", "--gate-only"):
        ap.add_argument(f, action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    if args.missing:
        j = jobs(args.gate_only); print(len(j)); [print(c.name) for c in j]; return
    for c in jobs(args.gate_only):
        f = OUT / "raw" / f"{c.name}.json"
        if f.exists():
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
        if f.exists():
            lock.unlink(missing_ok=True); continue
        if args.cpu:
            c.device = "cpu"
        print(f"\n=== {c.name} pid={os.getpid()} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"))
        lock.unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
