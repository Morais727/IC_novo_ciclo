#!/usr/bin/env python
"""Fase 8E (03/11/2026) — LaySelFL reproduzido NO CENÁRIO DO ARTIGO (Li et al., CVPR 2026, "Eliminate Distance
Differences Induced by Backdoor Attacks: Layer-Selective Training and Clipping to Mask Backdoor Models";
PDF em IC/artigos/LaySelFL_Li_CVPR2026_Eliminate_Distance_Differences.pdf). Mecanismo do ataque:
src/attacks/recent.py::laysel_attack.

Pergunta: a LaySelFL pega no cenário do próprio artigo (onde no nosso, 2/20 e ataque desde a rodada 1, ficou em
1,4%)? E, se pegar, a nossa proposta (layer_guard_sn) defende?

Cenário do artigo (§5.1), FIXADO ANTES DE RODAR:
  - 20 nós (nó 0 = servidor, 19 participantes); todos participam toda rodada; 6 maliciosos.
  - não-IID: Dirichlet α = 0,9.   dataset: CIFAR-10.   205 rodadas, 1 época local por rodada.
  - ataque nas rodadas 80–200.    SGD, LR 0,01 (rodadas <= 50) → 0,001 (depois).
  - gatilho BadNet: quatro tiras de 3x5 px (datasets.set_trigger("badnet_cifar")).   classe-alvo = 4.
  - taxa de envenenamento 0,2.   k = 60 camadas sensíveis (= nº de tensores − 2 na ResNet18-GN).   β = 20, corte 0,05.
Decisões do lab (registradas antes de rodar):
  - ResNet18-GN (GroupNorm) em vez de ResNet18-BN: BatchNorm é problemática em FL não-IID; o lab padronizou GN.
  - gatilho BadNet (estático), não o A3FL adaptativo — este fica para depois, só se o portão pegar.

Etapa 1 — PORTÃO (semente 42, sem defesa = lg_shadow): pega se o ASR final for >= 20%. Se não pegar, registra
  "a LaySelFL não pega nem no cenário do artigo" e PARA (nada reajustado).
Etapa 2 — só se pegar: a nossa proposta (layer_guard_sn), mesmas condições; depois as demais defesas.
Uso: python experiments/fase8e.py --one [--gate-only] | --report | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
from src.data import datasets

datasets.set_trigger("badnet_cifar")   # gatilho do artigo (CIFAR): quatro tiras 3x5

OUT = Path("results/fase8e")
SH = fase6.SH
ALPHA, SEED = 0.9, 42
DEFENSES = [SH, "layer_guard_sn", "layer_guard_client", "layer_guard_corr", "layer_guard_corr_soft", "lasa", "alignins", "rfldefense", "flame", "multi_krum", "fedsurrogate"]  # portão + nossa proposta + ablação + corroboração + corroboração-soft (substituição por consenso) + SOTA concorrentes


def cfg(defense):
    c = fase6.cfg("lp_full", defense, ALPHA, SEED)   # pega o cenário-base e sobrescreve com o do artigo
    c.attack = "laysel"
    c.model, c.dataset = "resnet18gn", "cifar10"
    c.num_malicious = 6
    c.rounds, c.local_epochs, c.eval_every = 205, 1, 10
    c.lr, c.optimizer = 0.01, "sgd"
    c.lr_schedule = [[50, 0.01], [205, 0.001]]
    c.attack_params = {"target_label": 4, "start_round": 80, "end_round": 200,
                       "beta": 20.0, "clip": 0.05, "poison_fraction": 0.2, "k": 60, "P": 2, "N": 2}
    c.name = f"laysel_paper__{defense}__a{ALPHA}_s{SEED}"
    return c


def final(defense):
    f = OUT / "raw" / f"laysel_paper__{defense}__a{ALPHA}_s{SEED}.json"
    if not f.exists():
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def gate():
    r = final(SH)
    return None if r is None else r["asr"] >= 20


def jobs(gate_only=False):
    j = [SH] if final(SH) is None else []
    if gate_only:
        return j
    if gate():                                        # só roda as defesas se o portão pegou
        j += [d for d in DEFENSES if d != SH and final(d) is None]
    return j


def report():
    out = ["# Fase 8E — LaySelFL no cenário do artigo",
           "ResNet18-GN, CIFAR-10, 20 nós (6 maliciosos), α 0,9, 205 rodadas, ataque 80–200, gatilho BadNet, alvo 4.", ""]
    g = final(SH)
    out.append("Portão (sem defesa): " + ("ainda não rodou" if g is None else
               f"ASR {g['asr']:.1f}%, acurácia {g['acc']:.1f}% → "
               f"{'PEGOU (>= 20%)' if g['asr'] >= 20 else 'NÃO pegou — para aqui'}"))
    if g and g["asr"] >= 20:
        out += ["", "| defesa | ASR | acurácia |", "|---|---|---|"]
        for d in DEFENSES:
            r = final(d)
            nome = fase6.LG if d == "layer_guard_sn" else ("sem defesa" if d == SH else d)
            if r:
                out.append(f"| {nome} | {r['asr']:.1f}% | {r['acc']:.1f}% |")
    txt = "\n".join(out) + "\n"
    (OUT / "RESULTADO.md").write_text(txt)
    print(txt)


def main():
    global SEED
    ap = argparse.ArgumentParser()
    for f in ("--one", "--report", "--missing", "--gate-only"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)   # rigor estatístico: 42 (padrão) + 43/44/45...
    a = ap.parse_args()
    SEED = a.seed
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if a.report:
        return report()
    if a.missing:
        j = jobs(a.gate_only); print(len(j)); [print(x) for x in j]; return
    for d in jobs(a.gate_only):
        f = OUT / "raw" / f"laysel_paper__{d}__a{ALPHA}_s{SEED}.json"
        if f.exists():
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        lock = OUT / "raw" / f"laysel_paper__{d}__a{ALPHA}_s{SEED}.lock"
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
        c = cfg(d)
        print(f"\n=== {c.name} pid={os.getpid()} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"))
        lock.unlink(missing_ok=True)
        if a.one:
            return
    if a.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
