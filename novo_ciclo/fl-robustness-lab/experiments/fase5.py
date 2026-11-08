#!/usr/bin/env python
"""Fase 5 (21/10/2026) — o que falta antes de submeter. Desenho, hipóteses e critérios FIXADOS ANTES DE RODAR.
Saídas em results/fase5/ (raw/, fila.log, RESULTADO.md). A nossa proposta = layer_guard_sn, SEM mudar nenhum
parâmetro (limiar 3). Linha de base sob ataque = lg_shadow (agrega como FedAvg); sem ataque = FedAvg.

E-A  Atacante adaptativo que otimiza contra o nosso escore (src/attacks/recent.py::_evade, ataque "lp_adapt"):
     LP evasivo onisciente que conhece a regra, o limiar e os deltas honestos do round; a cada round varre a força
     s ∈ {3 … 0,05} e o ruído ν ∈ {0; 0,5; 1; 2} da componente maliciosa em L*, simula a nossa proposta e envia a
     combinação de maior s × fração dos (atacante, tensor de L*) aceitos. CIFAR-10, SmallCNN, 2 de 20 atacantes,
     α ∈ {0,5; 0,3}, sementes 42 (referência) e 46, 47, 48 (teste); nossa proposta e lg_shadow. 16 treinos.
     H11: em cada semente de teste e α, a nossa proposta RESISTE se (a) ASR(nossa, adaptativo) <= 10% com acurácia
     >= linha de base − 3 pp, OU (b) ASR(FedAvg, adaptativo) <= 0,5 × ASR(FedAvg, LP evasivo não adaptativo) — fugir
     do filtro custa ao atacante pelo menos metade do backdoor (o mesmo critério B de layer_guard_sn_lab.py).
     Confirmada se resistir nos 6 casos de teste.
E-B  Mais atacantes: LP + ruído ν = 1 com 4 e com 6 atacantes de 20 (20% e 30%), CIFAR-10, SmallCNN, α ∈ {0,5; 0,3},
     sementes 46, 47, 48; nossa proposta e lg_shadow. 24 treinos.
     H12: com 4 atacantes a nossa proposta defende (critério da Fase 4) em todos os casos conclusivos de teste nos
     dois α. Com 6, o mesmo critério é reportado sem hipótese (fora do regime usual de 10%).
E-C  Outro conjunto: Fashion-MNIST (o dos autores do LP), SmallCNN de 1 canal, 2 de 20 atacantes, α ∈ {0,5; 0,3;
     0,1}, sementes 46, 47, 48; LP evasivo e LP + ruído (nossa proposta e lg_shadow) e sem ataque (FedAvg e nossa
     proposta). 54 treinos.
     H13: com α = 0,5 a nossa proposta defende LP evasivo e LP + ruído em todos os casos conclusivos de teste, sem
     custo de acurácia (−3 pp). α = 0,3 e 0,1 são reportados como curva (α* no FMNIST).
E-D  Modelo mais profundo: ResNet-20 com GroupNorm (src/models/cnn.py::ResNet20GN, 65 tensores, 0,27 M parâmetros),
     CIFAR-10, 2 de 20, α ∈ {0,5; 0,3}, sementes 46, 47, 48; LP + ruído (nossa e lg_shadow) e sem ataque (FedAvg e
     nossa). 24 treinos (~1,7 h cada com 3 vagas; medido: 58 s/round sozinho).
     H14: com α = 0,5 a nossa proposta defende o LP + ruído em todos os casos conclusivos de teste, sem custo de
     acurácia. α = 0,3 reportado. Se o FedAvg sem ataque não passar de 40% de acurácia no round 40, o resultado do
     ResNet é declarado "modelo não treinou" e NÃO se reajusta nada nesta fase.
Critério por caso (Fase 4): ASR final <= 10%, acurácia sob ataque >= linha de base − 3 pp, acurácia sem ataque >=
  FedAvg sem ataque − 3 pp; linha de base com ASR < 20% = ataque não pegou (inconclusivo).
ADENDO 21/10 (antes de qualquer resultado da fila; só a 1ª leva de E-A em andamento): a busca bibliográfica achou a
  LASA (Xu, Zhang e Hu, WACV 2025), que já decide por camada e por cliente (MZ-score da norma e da pureza de sinal por
  tensor), e a AlignIns (mesmos autores, CVPR 2025, por cliente). Nenhuma foi avaliada contra o LP. Reimplementadas do
  código oficial com os parâmetros padrão (src/defenses/lasa_alignins.py) e acrescentadas como linhas de base:
E-E  LASA e AlignIns, CIFAR-10, SmallCNN, 2 de 20, α ∈ {0,5; 0,3}, sementes 46, 47, 48: LP + ruído, LP evasivo e sem
     ataque. 36 treinos. Mesmo critério por caso (linha de base sob ataque = lg_shadow; sem ataque = FedAvg).
     H15: a nossa proposta defende em mais casos de teste que a LASA e que a AlignIns (contagem nos 12 casos
     ataque × α × semente). Também se reportam os casos em que só uma delas defende.
Ordem da fila: E-A, E-E, E-B (4), E-C, E-B (6), E-D.
Revisão bibliográfica (FLIP, FreqFed, BackdoorIndicator, defesas por camada 2024–2026): results/fase5/LITERATURA.md.
Uso: python experiments/fase5.py --one (uma vaga; results/fase5/slots.sh roda 3) | --report
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.fase2_lab import C
from experiments.fase4_alpha import find_name as find_old, mem_ok
from src.federated.simulate import run

OUT = Path("results/fase5")
LP, EV, AD = "lp_full_indiv_nu1.0", "lp_full_indiv", "lp_adapt"
TEST = (46, 47, 48)
SH, LG = "lg_shadow", "layer_guard_sn"
# tag do nome -> mudanças na configuração
TAGS = {"": {}, "_m4": dict(num_malicious=4), "_m6": dict(num_malicious=6), "_fm": dict(dataset="fmnist"),
        "_r20": dict(model="resnet20gn")}


def cfg(attack, defense, alpha, seed, tag=""):
    c = C(EV if attack == AD else attack, defense, alpha=alpha, seed=seed)
    if attack == AD:
        c.attack_params["evade"] = LG
    for k, v in TAGS[tag].items():
        setattr(c, k, v)
    if attack == "none":
        c.num_malicious = 0
    c.name = f"{attack}__{defense}__a{alpha}_s{seed}{tag}"
    return c


def jobs():
    j = [cfg(AD, d, a, s) for a in (0.5, 0.3) for s in (42,) + TEST for d in (SH, LG)]                      # E-A
    j += [cfg(x, d, a, s) for a in (0.5, 0.3) for s in TEST for d in ("lasa", "alignins") for x in (LP, EV, "none")]  # E-E
    j += [cfg(LP, d, a, s, "_m4") for a in (0.5, 0.3) for s in TEST for d in (SH, LG)]                      # E-B 4
    for a in (0.5, 0.3, 0.1):                                                                              # E-C
        for s in TEST:
            j += [cfg(x, d, a, s, "_fm") for x in (LP, EV) for d in (SH, LG)]
            j += [cfg("none", d, a, s, "_fm") for d in ("fedavg", LG)]
    j += [cfg(LP, d, a, s, "_m6") for a in (0.5, 0.3) for s in TEST for d in (SH, LG)]                      # E-B 6
    for a in (0.5, 0.3):                                                                                   # E-D
        for s in TEST:
            j += [cfg(LP, d, a, s, "_r20") for d in (SH, LG)] + [cfg("none", d, a, s, "_r20") for d in ("fedavg", LG)]
    return j


def find_name(name):
    p = OUT / "raw" / f"{name}.json"
    return p if p.exists() else (None if name.endswith(tuple(t for t in TAGS if t)) else find_old(name))


def final(attack, defense, alpha, seed, tag=""):
    f = find_name(f"{attack}__{defense}__a{alpha}_s{seed}{tag}")
    if f is None:
        return None
    d = json.load(open(f)); h = d["history"][-1]
    asr, acc = h["backdoor_asr"], h["main_accuracy"]
    return dict(asr=None if asr is None else asr, acc=acc,
                evade=d["attack_params"].get("_evade_log"))


def base(attack, alpha, seed, tag=""):
    return final(attack, SH, alpha, seed, tag) or final(attack, "fedavg", alpha, seed, tag)


def verdict(attack, alpha, seed, tag="", defense=LG):
    """Critério da Fase 4. Sem ataque: as rodadas comuns (sem tag) servem para _m4/_m6 e lp_adapt."""
    b, o = base(attack, alpha, seed, tag), final(attack, defense, alpha, seed, tag)
    ct = tag if tag in ("_fm", "_r20") else ""
    bc, oc = final("none", "fedavg", alpha, seed, ct), final("none", defense, alpha, seed, ct)
    if b is None or o is None:
        return None, b, o
    if b["asr"] < 20:
        return "inconclusivo", b, o
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return ("defende" if ok else "não defende"), b, o


def cell(v, b, o):
    if b is None and o is None:
        return "—"
    f = lambda x: "?" if x is None else f"{x['asr']:.1f}"
    mark = {"defende": " ✓", "não defende": " ✗", "inconclusivo": " (ataque não pegou)"}.get(v, "")
    return f"{f(b)} → {f(o)}{'' if o is None else ' (' + format(o['acc'], '.1f') + ')'}{mark}"


def table(out, title, attack, alphas, seeds, tag=""):
    out += [f"### {title}", "", "| semente | " + " | ".join(f"α = {a}" for a in alphas) + " |", "|---" * (len(alphas) + 1) + "|"]
    tally = {a: [] for a in alphas}
    for s in seeds:
        cells = []
        for a in alphas:
            v, b, o = verdict(attack, a, s, tag); cells.append(cell(v, b, o))
            if s in TEST and v is not None:
                tally[a].append(v)
        out.append(f"| {s}{' (teste)' if s in TEST else ''} | " + " | ".join(cells) + " |")
    out += ["", "Sementes de teste defendidas: " + " · ".join(
        f"α {a}: {sum(x == 'defende' for x in t)}/{sum(x != 'inconclusivo' for x in t)}"
        + (f" (+{sum(x == 'inconclusivo' for x in t)} inconclusivo)" if "inconclusivo" in t else "")
        + ("" if len(t) == 3 else f" [{len(t)}/3 prontos]") for a, t in tally.items()), ""]
    return tally


def clean(out, alphas, tag):
    out += ["Acurácia sem ataque (FedAvg → nossa proposta): " + " · ".join(
        f"α {a} s{s}: " + "/".join("?" if x is None else f"{x['acc']:.1f}" for x in
                                   (final("none", "fedavg", a, s, tag), final("none", LG, a, s, tag)))
        for a in alphas for s in TEST), ""]


def report():
    out = ["# Fase 5 — o que falta antes de submeter (critérios: docstring de experiments/fase5.py)", "",
           "Células: ASR linha de base → ASR nossa proposta (acurácia nossa).", ""]
    # E-A
    out += ["## E-A — atacante adaptativo contra o nosso escore (CIFAR-10, 2 de 20)", ""]
    res = []
    out += ["| semente | α | ASR FedAvg LP evasivo | ASR FedAvg adaptativo | ASR nossa adaptativo (acc) | (a) | (b) | escolhas do atacante (s, ν, aceito) |",
            "|---|---|---|---|---|---|---|---|"]
    for a in (0.5, 0.3):
        for s in (42,) + TEST:
            ev, fb, fo = base(EV, a, s), base(AD, a, s), final(AD, LG, a, s)
            if fb is None and fo is None:
                out.append(f"| {s} | {a} | — | — | — | | | |"); continue
            ca = fo is not None and fb is not None and fo["asr"] <= 10 and fo["acc"] >= fb["acc"] - 3
            cb = ev is not None and fb is not None and fb["asr"] <= 0.5 * ev["asr"]
            if s in TEST and fo is not None and fb is not None:
                res.append(ca or cb)
            log = (fo or {}).get("evade") or []
            ch = ""
            if log:
                import numpy as np
                ch = f"s médio {np.mean([e['s'] for e in log]):.2f}, ν médio {np.mean([e['nu'] for e in log]):.2f}, aceito {np.mean([e['kept'] for e in log]):.2f}"
            g = lambda x: "?" if x is None else f"{x['asr']:.1f}"
            out.append(f"| {s}{' (teste)' if s in TEST else ''} | {a} | {g(ev)} | {g(fb)} | {g(fo)}"
                       f"{'' if fo is None else ' (' + format(fo['acc'], '.1f') + ')'} | {'✓' if ca else '✗'} | {'✓' if cb else '✗'} | {ch} |")
    out += ["", f"H11: resiste em {sum(res)} de {len(res)} casos de teste prontos (de 6)"
            + ("" if len(res) < 6 else (" → CONFIRMADA" if all(res) else " → REFUTADA")), ""]
    # E-E
    out += ["## E-E — comparação com LASA e AlignIns (CIFAR-10, 2 de 20, sementes de teste)", "",
            "| ataque | α | semente | linha de base | nossa proposta | LASA | AlignIns |", "|---|---|---|---|---|---|---|"]
    wins = {LG: 0, "lasa": 0, "alignins": 0}; done = 0
    for x, nome in ((LP, "LP + ruído"), (EV, "LP evasivo")):
        for a in (0.5, 0.3):
            for s in TEST:
                vs = {d: verdict(x, a, s, "", d) for d in wins}
                b = base(x, a, s)
                if all(v[0] is not None for v in vs.values()):
                    done += 1
                    for d, v in vs.items():
                        wins[d] += v[0] == "defende"
                c = lambda d: "—" if vs[d][2] is None else f"{vs[d][2]['asr']:.1f} ({vs[d][2]['acc']:.1f})" + {"defende": " ✓", "não defende": " ✗", "inconclusivo": " (ataque não pegou)"}.get(vs[d][0], "")
                out.append(f"| {nome} | {a} | {s} | {'—' if b is None else format(b['asr'], '.1f') + ' (' + format(b['acc'], '.1f') + ')'} | {c(LG)} | {c('lasa')} | {c('alignins')} |")
    out += ["", "Acurácia sem ataque (FedAvg / nossa / LASA / AlignIns): " + " · ".join(
        f"α {a} s{s}: " + "/".join("?" if v is None else f"{v['acc']:.1f}" for v in
                                   (final("none", d, a, s) for d in ("fedavg", LG, "lasa", "alignins"))) for a in (0.5, 0.3) for s in TEST), ""]
    out += [f"Casos defendidos ({done} de 12 prontos): nossa {wins[LG]} · LASA {wins['lasa']} · AlignIns {wins['alignins']}"
            + ("" if done < 12 else (" → H15 CONFIRMADA" if wins[LG] > max(wins["lasa"], wins["alignins"]) else " → H15 REFUTADA")), ""]
    # E-B
    # correção pós-resultado (24/10): sem nenhum caso conclusivo o veredito é INCONCLUSIVO, não "confirmada"
    hverd = lambda xs: ("INCONCLUSIVA (o ataque não pegou em nenhum caso; nada a testar)" if all(x == "inconclusivo" for x in xs)
                        else "CONFIRMADA" if all(x != "não defende" for x in xs) else "REFUTADA")
    out += ["## E-B — mais atacantes (LP + ruído, CIFAR-10)", ""]
    for m in (4, 6):
        t = table(out, f"{m} atacantes de 20", LP, (0.5, 0.3), TEST, f"_m{m}")
        if m == 4 and all(len(v) == 3 for v in t.values()):
            ok = all(x != "não defende" for v in t.values() for x in v)
            out += [f"H12 {'CONFIRMADA' if ok else 'REFUTADA'}", ""]
    # E-C
    out += ["## E-C — Fashion-MNIST (SmallCNN de 1 canal, 2 de 20)", ""]
    h13 = []
    for x, nome in ((LP, "LP evasivo + ruído"), (EV, "LP evasivo")):
        t = table(out, nome, x, (0.5, 0.3, 0.1), TEST, "_fm"); h13.append(t[0.5])
    clean(out, (0.5, 0.3, 0.1), "_fm")
    if all(len(v) == 3 for v in h13):
        out += [f"H13 {hverd([x for v in h13 for x in v])}", ""]
    # E-D
    out += ["## E-D — ResNet-20 com GroupNorm (CIFAR-10, 2 de 20)", ""]
    t = table(out, "LP evasivo + ruído", LP, (0.5, 0.3), TEST, "_r20")
    clean(out, (0.5, 0.3), "_r20")
    fa = [final("none", "fedavg", 0.5, s, "_r20") for s in TEST]
    if all(fa) and min(x["acc"] for x in fa) < 40:
        out += ["ResNet: FedAvg sem ataque < 40% → modelo não treinou (regra fixada antes); H14 não se aplica.", ""]
    elif len(t[0.5]) == 3:
        out += [f"H14 {hverd(t[0.5])}", ""]
    txt = "\n".join(out) + "\n"
    (OUT / "RESULTADO.md").write_text(txt); print(txt)


def claim(name):
    lock = OUT / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); return False
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); return claim(name)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    for c in jobs():
        if find_name(c.name):
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        if not claim(c.name):
            continue
        print(f"\n=== {c.name} pid={os.getpid()}", flush=True)
        json.dump(run(c), open(OUT / "raw" / f"{c.name}.json", "w"))
        (OUT / "raw" / f"{c.name}.lock").unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
