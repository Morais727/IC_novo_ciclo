#!/usr/bin/env python
"""Fase 4 (18/10/2026) — varredura de α de Dirichlet entre 0,5 e 0,1. Pedido: repetir os testes com α = 0,4, 0,3,
0,2 para achar onde a nossa proposta (layer_guard_sn, SEM nenhuma mudança de parâmetro) deixa de barrar o LP.
Desenho, hipóteses e critérios fixados ANTES de rodar. Saídas em results/fase4/ (raw/, fila.log, RESULTADO.md).

Treinos (α = 0,5 e 0,1 já existem em results/fase2 e results/fase3; aqui só os que faltam):
  P1  α ∈ {0,4; 0,3; 0,2} × sementes 46, 47, 48 (teste): LP evasivo + ruído ν = 1 com lg_shadow (agrega como
      FedAvg e registra os escores por tensor = linha de base + diagnóstico) e com a nossa proposta; sem ataque com
      FedAvg e com a nossa proposta. 36 treinos.
  P2  o mesmo na semente 42 (onde a proposta foi desenvolvida). 12 treinos.
  P3  lacunas para a curva completa: α = 0,5 LP ν1 com lg_shadow nas sementes 42, 46, 47, 48 (linha de base que
      faltava na s42 + diagnóstico em α = 0,5). 4 treinos.
  P4  LP evasivo sem ruído, α ∈ {0,4; 0,3; 0,2} × sementes 42, 46, 47, 48 (lg_shadow e nossa proposta) e
      α = 0,1 × sementes 46, 47, 48 (os dois; só existia a s42). 30 treinos.
  Ordem da fila: P1, P2, P3, P4.

Critério por caso (semente × α × ataque), o mesmo das fases anteriores — a nossa proposta DEFENDE se:
  ASR final <= 10%, acurácia sob ataque >= linha de base sob o mesmo ataque − 3 pp, e acurácia sem ataque >= FedAvg
  sem ataque − 3 pp (mesma semente e α). Caso com ASR da linha de base < 20% = ataque não pegou (inconclusivo).
Hipóteses:
  H9   a fração de sementes defendidas (LP + ruído) não aumenta quando α diminui (0,5 → 0,1).
       α* = menor α em que a nossa proposta defende nas 3 sementes de teste. Reporta-se α* e a curva.
  H10  (diagnóstico, explica H9) quando α diminui: o escore médio dos atacantes nos tensores de L* cai e o número
       médio de tensores em L* por round sobe. Medido nos registros do lg_shadow (como em experiments/fase2_diag.py).
       Confirmada se as duas tendências forem monótonas na média das sementes de teste entre α = 0,5 e 0,1.
Uso: python experiments/fase4_alpha.py --one (uma vaga; results/fase4/slots.sh roda 3) | --report
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from experiments.fase2_lab import C
from experiments.fase2_diag import load, score
from src.federated.simulate import run

OUT = Path("results/fase4")
LP, EV = "lp_full_indiv_nu1.0", "lp_full_indiv"
NEW = (0.4, 0.3, 0.2)
TEST = (46, 47, 48)
ALPHAS = (0.5, 0.4, 0.3, 0.2, 0.1)


def jobs():
    j = []
    for s in TEST:
        for a in NEW:
            j += [C(LP, d, alpha=a, seed=s) for d in ("lg_shadow", "layer_guard_sn")]
            j += [C("none", d, alpha=a, seed=s) for d in ("fedavg", "layer_guard_sn")]
    for a in NEW:
        j += [C(LP, d, alpha=a, seed=42) for d in ("lg_shadow", "layer_guard_sn")]
        j += [C("none", d, alpha=a, seed=42) for d in ("fedavg", "layer_guard_sn")]
    j += [C(LP, "lg_shadow", alpha=0.5, seed=s) for s in (42,) + TEST]
    for a in NEW:
        j += [C(EV, d, alpha=a, seed=s) for s in (42,) + TEST for d in ("lg_shadow", "layer_guard_sn")]
    j += [C(EV, d, alpha=0.1, seed=s) for s in TEST for d in ("lg_shadow", "layer_guard_sn")]
    return j


def find_name(name):
    """Procura o treino pelo nome em todas as pastas de resultados (fases anteriores incluídas)."""
    for d in ("results/fase4/raw", "results/fase3/raw", "results/fase2/raw", "results/fase3c/raw"):
        if (Path(d) / f"{name}.json").exists():
            return Path(d) / f"{name}.json"
    if name.endswith("_s42"):   # rodadas antigas da semente 42 não levam o sufixo _s42
        for d in ("results/final_lab/raw", "results/layer_guard_sn/raw", "results/layer_guard/raw", "results/flame_lab/raw"):
            p = Path(d) / f"{name[:-4]}.json"
            if p.exists():
                return p
    return None


def find(attack, defense, alpha, seed):
    return find_name(f"{attack}__{defense}__a{alpha}_s{seed}")


def final(attack, defense, alpha, seed):
    f = find(attack, defense, alpha, seed)
    if f is None:
        return None
    h = json.load(open(f))["history"][-1]
    asr, acc = h["backdoor_asr"], h["main_accuracy"]
    return dict(asr=None if asr is None else asr, acc=acc, file=f)


def base(attack, alpha, seed):
    return final(attack, "lg_shadow", alpha, seed) or final(attack, "fedavg", alpha, seed)


def verdict(attack, alpha, seed):
    b, o = base(attack, alpha, seed), final(attack, "layer_guard_sn", alpha, seed)
    bc, oc = final("none", "fedavg", alpha, seed), final("none", "layer_guard_sn", alpha, seed)
    if b is None or o is None:
        return None, b, o
    if b["asr"] < 20:
        return "inconclusivo", b, o
    ok = o["asr"] <= 10 and o["acc"] >= b["acc"] - 3 and (bc is None or oc is None or oc["acc"] >= bc["acc"] - 3)
    return ("defende" if ok else "não defende"), b, o


def diag(alpha, seed):
    """H10: escore médio dos atacantes nos tensores de L* e nº médio de tensores em L* por round."""
    f = find(LP, "lg_shadow", alpha, seed)
    if f is None:
        return None
    L = load(f); s = score(L)
    vals, sizes = [], []
    for r in range(L["T"]):
        if r >= len(L["lp"]):
            continue
        gi = [L["names"].index(g) for g in L["lp"][r]]
        sizes.append(len(gi))
        vals += [s[r, g, m] for g in gi for m in L["mal"]]
    return dict(score=float(np.mean(vals)), n_lstar=float(np.mean(sizes)))


def report():
    out = ["# Fase 4 — varredura de α (critérios: docstring de experiments/fase4_alpha.py)\n"]
    curve = {}
    for attack, nome in ((LP, "LP evasivo + ruído ν = 1"), (EV, "LP evasivo sem ruído")):
        out += [f"## {nome}: ASR linha de base → nossa proposta (acurácia nossa)", "",
                "| semente | " + " | ".join(f"α = {a}" for a in ALPHAS) + " |", "|---" * (len(ALPHAS) + 1) + "|"]
        for s in (42,) + TEST:
            cells = []
            for a in ALPHAS:
                v, b, o = verdict(attack, a, s)
                if b is None and o is None:
                    cells.append("—"); continue
                fmt = lambda x: "?" if x is None else f"{x['asr']:.1f}"
                mark = {"defende": " ✓", "não defende": " ✗", "inconclusivo": " (ataque não pegou)"}.get(v, "")
                cells.append(f"{fmt(b)} → {fmt(o)}{'' if o is None else f' ({o[chr(97)+chr(99)+chr(99)]:.1f})'}{mark}")
                if attack == LP and s in TEST:
                    curve.setdefault(a, []).append(v)
            out.append(f"| {s}{' (teste)' if s in TEST else ''} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## H9 — sementes de teste defendidas contra LP + ruído, por α", ""]
    full = [a for a in ALPHAS if curve.get(a) and len(curve[a]) == 3 and None not in curve[a]]
    for a in ALPHAS:
        v = curve.get(a, [])
        out.append(f"- α = {a}: {sum(x == 'defende' for x in v)} de {len(v)} {v}")
    if len(full) == len(ALPHAS):
        fr = [sum(x == "defende" for x in curve[a]) for a in ALPHAS]
        ok = all(fr[i] >= fr[i + 1] for i in range(len(fr) - 1))
        good = [a for a in ALPHAS if all(x == "defende" for x in curve[a])]
        out.append(f"\nH9 {'CONFIRMADA' if ok else 'REFUTADA (curva não monótona)'}; α* = {min(good) if good else 'nenhum'}")
    out += ["", "## H10 — diagnóstico nos registros do lg_shadow (média das sementes de teste)", "",
            "| α | escore médio dos atacantes em L* | tensores em L* por round | sementes |", "|---|---|---|---|"]
    sc, nl = [], []
    for a in ALPHAS:
        ds = [d for d in (diag(a, s) for s in TEST) if d]
        if ds:
            sc.append(np.mean([d["score"] for d in ds])); nl.append(np.mean([d["n_lstar"] for d in ds]))
            out.append(f"| {a} | {sc[-1]:.2f} | {nl[-1]:.2f} | {len(ds)} |")
        else:
            out.append(f"| {a} | — | — | 0 |")
    if len(sc) == len(ALPHAS):
        mono = all(sc[i] >= sc[i + 1] for i in range(len(sc) - 1)) and all(nl[i] <= nl[i + 1] for i in range(len(nl) - 1))
        out.append(f"\nH10 {'CONFIRMADA' if mono else 'REFUTADA (tendência não monótona)'}")
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


def mem_ok(gb=2.5):
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6 >= gb
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    for cfg in jobs():
        if find_name(cfg.name):
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        if not claim(cfg.name):
            continue
        print(f"\n=== {cfg.name} pid={os.getpid()}", flush=True)
        json.dump(run(cfg), open(OUT / "raw" / f"{cfg.name}.json", "w"))
        (OUT / "raw" / f"{cfg.name}.lock").unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
