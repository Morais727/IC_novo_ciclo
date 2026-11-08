#!/usr/bin/env python
"""Fase 6 (25/10/2026) — pendências do esboço do artigo marcadas pelo usuário. Desenho, hipóteses e critérios
FIXADOS ANTES DE RODAR. Saídas em results/fase6/ (raw/, fila.log, RESULTADO.md). A nossa proposta = layer_guard_sn
sem mudar nenhum parâmetro. Critério por caso = o da Fase 4 (experiments/fase5.py::verdict): ASR <= 10%, acurácia
sob ataque >= linha de base − 3 pp, acurácia sem ataque >= FedAvg sem ataque − 3 pp; linha de base < 20% =
inconclusivo. Linha de base sob ataque = lg_shadow quando existe, senão FedAvg.

G1  Tabela 5.1, semente 42, LP + ruído, α = 0,5, sem defesa: já existe (results/fase4/raw, lg_shadow: ASR 35,4%,
    acurácia 72,4%). Nenhum treino.
G2  Multi-Krum e FLAME nas sementes de teste: LP + ruído ν = 1, α = 0,1, sementes 46, 47, 48. 6 treinos.
    H18: nenhuma das duas defende algum caso conclusivo (mantém a frase "nada barra o LP com α = 0,1").
G3  LP original (lp_full, não evasivo) e backdoor PGD em mais sementes: α = 0,1, sementes 44, 45, 46, 47, 48, FedAvg
    e nossa proposta (20 treinos); LP original com α = 0,5, sementes 46, 47, 48, FedAvg e nossa proposta (6 treinos).
    H16: com α = 0,5 a nossa proposta defende o LP original em todos os casos conclusivos de teste. α = 0,1 (LP
    original e PGD) é descritivo; o esperado, pela semente 42, é PGD não defendido.
G4  LASA e AlignIns com α = 0,1: LP + ruído, LP evasivo e sem ataque, sementes 46, 47, 48. 18 treinos.
    H17: nenhuma das duas defende algum caso conclusivo de LP + ruído (mantém a frase de G2).
G5  Fazer o LP pegar no Fashion-MNIST e na ResNet-20-GN (na Fase 5 o modelo de cada atacante tinha backdoor de
    ~100%, mas o ataque não passou de 20% no global: λ = 0,5 dilui a parte envenenada).
    Calibração SÓ na semente 42 (desenvolvimento), α = 0,5, LP + ruído, sem defesa (lg_shadow): candidatos na ordem
    λ = 1 (tag L1), λ = 2 (tag L2). Escolhe-se o PRIMEIRO candidato com ASR >= 40% no round 40. Nenhum outro
    parâmetro muda. Sem candidato aprovado: "ataque não calibrado", G5 para aí. 4 treinos de calibração.
    Com o λ escolhido, repetem-se nas sementes 46, 47, 48 os treinos com ataque de E-C (FMNIST: LP + ruído e LP
    evasivo, α 0,5 / 0,3 / 0,1, lg_shadow e nossa; 36) e de E-D (ResNet: LP + ruído, α 0,5 / 0,3, lg_shadow e
    nossa; 12). Sem ataque: reaproveitam-se os treinos da Fase 5 (λ não entra sem ataque).
    H13' e H14' = H13 e H14 da Fase 5 com o ataque calibrado: com α = 0,5 a nossa proposta defende em todos os
    casos conclusivos de teste, sem custo de acurácia.
G6  (adendo de 25/10, pedido do usuário, fixado antes de rodar) RFLDefense (Shi, Gong e He, Cybersecurity 2026),
    reimplementada em src/defenses/rfldefense.py com os valores fixados lá (β = 0,8 da FPD, top-k 30%, Z = 3,5,
    Isolation Forest padrão). LP + ruído, LP evasivo e sem ataque, α = 0,5 e 0,1, sementes 46, 47, 48. 18 treinos.
    H19: com α = 0,5 a RFLDefense defende MENOS casos conclusivos de LP (+ ruído e evasivo) do que a nossa proposta
    nos mesmos casos (Fase 5). α = 0,1 é descritivo.
Ordem da fila: calibração, G2, G4, G3, G5, G6. Total 54 + 18 treinos + até 48 de G5 (G5 não calibrou: não roda).
Uso: python experiments/fase6.py --one | --report
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase5
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run

OUT = Path("results/fase6")
LP, EV, PIX = fase5.LP, fase5.EV, "lp_full"
PGD = "backdoor_pgd"
TEST, SH, LG = fase5.TEST, fase5.SH, fase5.LG
LAMS = {"L1": 1.0, "L2": 2.0}


def split(tag):
    """'_fmL1' -> ('_fm', 'L1')"""
    for k in LAMS:
        if tag.endswith(k):
            return tag[: -len(k)], k
    return tag, ""


def cfg(attack, defense, alpha, seed, tag=""):
    base, lam = split(tag)
    c = fase5.cfg(attack, defense, alpha, seed, base)
    if lam:
        c.attack_params["lambda"] = LAMS[lam]
    c.name = f"{attack}__{defense}__a{alpha}_s{seed}{tag}"
    return c


def find_name(name):
    p = OUT / "raw" / f"{name}.json"
    if p.exists():
        return p
    stem, seed_tag = name.rsplit("_s", 1)
    num = re.match(r"\d+", seed_tag).group()
    base, lam = split(seed_tag[len(num):])
    if lam:
        if not name.startswith("none__"):
            return None
        name = f"{stem}_s{num}{base}"          # sem ataque: λ não entra
    return fase5.find_name(name)


def final(attack, defense, alpha, seed, tag=""):
    f = find_name(f"{attack}__{defense}__a{alpha}_s{seed}{tag}")
    if f is None:
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def verdict(attack, alpha, seed, tag="", defense=LG):
    b = final(attack, SH, alpha, seed, tag) or final(attack, "fedavg", alpha, seed, tag)
    o = final(attack, defense, alpha, seed, tag)
    ct = split(tag)[0]
    bc, oc = final("none", "fedavg", alpha, seed, ct), final("none", defense, alpha, seed, ct)
    if b is None or o is None:
        return None, b, o
    if b["asr"] < 20:
        return "inconclusivo", b, o
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return ("defende" if ok else "não defende"), b, o


def chosen(base):
    """Regra de calibração de G5: primeiro λ com ASR >= 40% na semente 42. None = ainda rodando; '' = nenhum."""
    for k in LAMS:
        r = final(LP, SH, 0.5, 42, base + k)
        if r is None:
            return None
        if r["asr"] >= 40:
            return k
    return ""


def jobs():
    j = [cfg(LP, SH, 0.5, 42, b + k) for b in ("_fm", "_r20") for k in LAMS]                          # calibração
    j += [cfg(LP, d, 0.1, s) for s in TEST for d in ("multi_krum", "flame")]                            # G2
    j += [cfg(x, d, 0.1, s) for s in TEST for d in ("lasa", "alignins") for x in (LP, EV, "none")]       # G4
    j += [cfg(x, d, 0.1, s) for x in (PIX, PGD) for s in (44, 45) + TEST for d in ("fedavg", LG)]       # G3
    j += [cfg(PIX, d, 0.5, s) for s in TEST for d in ("fedavg", LG)]
    for b, alphas, atks in (("_fm", (0.5, 0.3, 0.1), (LP, EV)), ("_r20", (0.5, 0.3), (LP,))):          # G5
        k = chosen(b)
        if k:
            j += [cfg(x, d, a, s, b + k) for a in alphas for s in TEST for x in atks for d in (SH, LG)]
    j += [cfg(x, "rfldefense", a, s) for a in (0.5, 0.1) for s in TEST for x in (LP, EV, "none")]          # G6
    return j


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
    if (OUT / "raw" / f"{name}.json").exists():  # outro worker terminou entre montar a lista e pegar o lock
        lock.unlink(missing_ok=True); return False
    return True


def cellv(v, b, o):
    if b is None and o is None:
        return "—"
    f = lambda x: "?" if x is None else f"{x['asr']:.1f}"
    mark = {"defende": " ✓", "não defende": " ✗", "inconclusivo": " (ataque não pegou)"}.get(v, "")
    return f"{f(b)} → {f(o)}{'' if o is None else ' (' + format(o['acc'], '.1f') + ')'}{mark}"


def grid(out, title, attack, alphas, seeds, tag="", defenses=(LG,)):
    out += [f"### {title}", "", "| semente | defesa | " + " | ".join(f"α = {a}" for a in alphas) + " |",
            "|---" * (len(alphas) + 2) + "|"]
    tally = {(d, a): [] for d in defenses for a in alphas}
    for s in seeds:
        for d in defenses:
            cells = []
            for a in alphas:
                v, b, o = verdict(attack, a, s, tag, d); cells.append(cellv(v, b, o))
                if s in TEST and v is not None:
                    tally[d, a].append(v)
            out.append(f"| {s}{' (teste)' if s in TEST else ''} | {d} | " + " | ".join(cells) + " |")
    out += ["", "Sementes de teste defendidas: " + " · ".join(
        f"{d} α {a}: {sum(x == 'defende' for x in t)}/{sum(x != 'inconclusivo' for x in t)}"
        + (f" (+{sum(x == 'inconclusivo' for x in t)} inconclusivo)" if "inconclusivo" in t else "")
        + ("" if len(t) == 3 else f" [{len(t)}/3 prontos]") for (d, a), t in tally.items()), ""]
    return tally


def hverd(xs, want_defend):
    if not xs or any(x is None for x in xs):
        return "pendente"
    if all(x == "inconclusivo" for x in xs):
        return "INCONCLUSIVA (o ataque não pegou em nenhum caso)"
    if want_defend:
        return "CONFIRMADA" if all(x != "não defende" for x in xs) else "REFUTADA"
    return "CONFIRMADA" if not any(x == "defende" for x in xs) else "REFUTADA"


def report():
    out = ["# Fase 6 — pendências do esboço (critérios: docstring de experiments/fase6.py)", "",
           "Células: ASR linha de base → ASR defesa (acurácia da defesa).", "",
           "## G1 — tabela 5.1: semente 42, LP + ruído, α = 0,5, sem defesa (Fase 4, lg_shadow)", ""]
    r = fase5.final(LP, SH, 0.5, 42)
    out += [f"ASR {r['asr']:.1f}% · acurácia {r['acc']:.1f}%" if r else "—", ""]
    out += ["## G2 — Multi-Krum e FLAME, LP + ruído, α = 0,1", ""]
    t = grid(out, "LP + ruído", LP, (0.1,), TEST, defenses=("multi_krum", "flame"))
    out += [f"H18 {hverd([x for v in t.values() for x in v] if all(len(v) == 3 for v in t.values()) else [None], False)}", ""]
    out += ["## G4 — LASA e AlignIns com α = 0,1", ""]
    t = grid(out, "LP + ruído", LP, (0.1,), TEST, defenses=("lasa", "alignins"))
    grid(out, "LP evasivo", EV, (0.1,), TEST, defenses=("lasa", "alignins"))
    out += [f"H17 {hverd([x for v in t.values() for x in v] if all(len(v) == 3 for v in t.values()) else [None], False)}", ""]
    out += ["## G3 — LP original e PGD em mais sementes (nossa proposta)", ""]
    t = grid(out, "LP original", PIX, (0.5, 0.1), (42, 44, 45) + TEST)
    out += [f"H16 {hverd(t[LG, 0.5] if len(t[LG, 0.5]) == 3 else [None], True)}", ""]
    grid(out, "Backdoor PGD", PGD, (0.1,), (42, 44, 45) + TEST)
    out += ["## G5 — Fashion-MNIST e ResNet com o ataque calibrado", ""]
    for b, nome in (("_fm", "Fashion-MNIST"), ("_r20", "ResNet-20-GN")):
        cal = " · ".join(f"λ = {LAMS[k]}: " + (lambda r: "rodando" if r is None else f"ASR {r['asr']:.1f}% (acc {r['acc']:.1f})")(
            final(LP, SH, 0.5, 42, b + k)) for k in LAMS)
        k = chosen(b)
        out += [f"### {nome}", "", f"Calibração (semente 42, α = 0,5, sem defesa): {cal} → "
                + ("pendente" if k is None else "nenhum λ aprovado: ataque não calibrado" if k == "" else f"λ = {LAMS[k]}"), ""]
        if k:
            alphas = (0.5, 0.3, 0.1) if b == "_fm" else (0.5, 0.3)
            ts = [grid(out, f"{nome} · {x}", x, alphas, TEST, b + k) for x in ((LP, EV) if b == "_fm" else (LP,))]
            h = [x for tt in ts for x in tt[LG, 0.5]]
            out += [f"{'H13' if b == '_fm' else 'H14'}' {hverd(h if len(h) == 3 * len(ts) else [None], True)}", ""]
    out += ["## G6 — RFLDefense (adendo)", ""]
    ts = [grid(out, nm, x, (0.5, 0.1), TEST, defenses=("rfldefense", LG)) for nm, x in (("LP + ruído", LP), ("LP evasivo", EV))]
    r = [x for t in ts for x in t["rfldefense", 0.5]]; o = [x for t in ts for x in t[LG, 0.5]]
    if len(r) < 6 or len(o) < 6:
        out += ["H19 pendente", ""]
    else:
        nr, no = sum(x == "defende" for x in r), sum(x == "defende" for x in o)
        out += [f"H19 {'CONFIRMADA' if nr < no else 'REFUTADA'} (α = 0,5: RFLDefense {nr} vs nossa proposta {no} casos defendidos)", ""]
    txt = "\n".join(out) + "\n"
    (OUT / "RESULTADO.md").write_text(txt); print(txt)


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
