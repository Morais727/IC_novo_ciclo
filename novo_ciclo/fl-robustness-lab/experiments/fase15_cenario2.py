#!/usr/bin/env python
"""Fase 15 (13/11/2026) — SEGUNDO CENÁRIO do artigo 2 ("ASR baixo não significa modelo limpo"), experimento 2.3.
Desenho FIXADO ANTES DE RODAR.

Cenário = o da Fase 8A com o ataque LSA (Foroughi et al., ICC 2026): SmallCNN, CIFAR-10, 20 clientes, 2 atacantes,
40 rodadas, gatilho 4×4, alvo 0, ataque desde a 1ª rodada (fase8a.cfg). Escolhido porque (i) é barato (~15 min por
corrida) e (ii) na grade da 8A muitas defesas terminam com ASR residual na faixa cinzenta (5–30%): LASA, AlignIns,
FedSurrogate e RFLDefense em α 0,2–0,4 — exatamente o caso que o artigo 2 estuda.
Etapa 1 (--worker): refaz a grade da 8A nas faixas α {0,2; 0,3; 0,4} × sementes 46–50 × 8 agregações (as da 8A),
  agora SALVANDO o modelo final (results/fase15/models), mais o controle limpo (sem ataque, FedAvg) por α×semente.
  Serve também de teste de reprodutibilidade da 8A (mesma configuração e semente).
Etapa 2 (--rea-worker): para cada modelo final com ASR residual entre 3% e 40% (defesas, não o sem-defesa) e para
  os controles limpos: dormência (10 rodadas sem ataque e sem defesa) e reativação (10 rodadas com os atacantes
  de volta e sem defesa), avaliando a cada rodada. Saída results/fase15/rea/<modelo>__<modo>.json.
Vários workers podem rodar em paralelo (lock atômico por corrida). Cada worker sai quando não há mais nada a pegar;
o último a terminar cria results/fase15/GRID.ok / REA.ok.
Uso: CUDA_VISIBLE_DEVICES=5 python -u experiments/fase15_cenario2.py --worker | --rea-worker | --report | --quick
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase8a

OUT = Path("results/fase15")
ALPHAS = (0.3, 0.2, 0.4)                 # 0,3 primeiro: a faixa com mais modelos na zona cinzenta
SEEDS = (46, 47, 48, 49, 50)             # as sementes de teste da 8A
DEFENSES = fase8a.DEFENSES               # (lg_shadow, layer_guard_sn, lasa, alignins, rfldefense, flame, multi_krum, fedsurrogate)
REA_ROUNDS, REA_LO, REA_HI = 10, 3.0, 40.0


def grid():
    for a in ALPHAS:
        for s in SEEDS:
            yield ("none", fase8a.SH, a, s)                 # controle limpo
            for d in DEFENSES:
                yield ("lsa", d, a, s)


def name(atk, d, a, s):
    return f"c2_{'clean' if atk == 'none' else atk}__{d}__a{a}_s{s}"


def make_cfg(atk, d, a, s):
    c = fase8a.cfg("lsa", d, a, s)
    if atk == "none":   # controle limpo: mesmo ataque registrado (mede o ASR, mesmos ids), mas atacantes sempre honestos
        c.attack_params["start_round"] = c.attack_params["end_round"] = 10 ** 6
    c.name = name(atk, d, a, s)
    return c


def take(lock):
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); return False
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True)
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def final_asr(f):
    h = json.load(open(f))["history"]
    return h[-1]["backdoor_asr"]


def rea_jobs():
    """(modelo .pt, semente, α, modo) — só modelos já treinados, com ASR residual na faixa, mais os limpos."""
    out = []
    for atk, d, a, s in grid():
        n = name(atk, d, a, s)
        f, m = OUT / "raw" / f"{n}.json", OUT / "models" / f"{n}.pt"
        if not (f.exists() and m.exists()):
            continue
        if atk == "none":
            out.append((m, a, s, "reativacao"))
        elif d != fase8a.SH and REA_LO <= (final_asr(f) or 0) <= REA_HI:
            out += [(m, a, s, "dormencia"), (m, a, s, "reativacao")]
    return out


def worker(rea=False):
    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    sub = "rea" if rea else "raw"
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    if rea:
        os.environ.pop("FL_SAVE_FINAL", None); os.environ.pop("FL_SAVE_ROUNDS", None)
    else:
        os.environ["FL_SAVE_FINAL"] = str(OUT / "models"); os.environ.pop("FL_SAVE_ROUNDS", None)
    jobs = rea_jobs() if rea else list(grid())
    for j in jobs:
        if rea:
            m, a, s, mode = j
            n = f"{m.stem}__{mode}"
        else:
            n = name(*j)
        f = OUT / sub / f"{n}.json"
        lock = f.with_suffix(".lock")
        if f.exists() or not take(lock):
            continue
        while not mem_ok():
            time.sleep(30)
        if rea:
            c = make_cfg("lsa", fase8a.SH, a, s)       # mesma partição/atacantes da origem; sem defesa
            on = mode == "reativacao"
            if not on:   # mantém o ataque registrado (para medir o ASR), mas fora da janela: atacantes honestos
                c.attack_params["start_round"] = c.attack_params["end_round"] = 10 ** 6
            c.rounds, c.eval_every, c.name = REA_ROUNDS, 1, n
            os.environ["FL_INIT_VECTOR"] = str(m)
        else:
            c = make_cfg(*j)
            os.environ.pop("FL_INIT_VECTOR", None)
        print(f"\n=== {n} pid={os.getpid()} ===", flush=True)
        try:
            json.dump(run(c), open(f, "w"), default=str)
        finally:
            lock.unlink(missing_ok=True)
    todo = rea_jobs() if rea else list(grid())
    names = [f"{j[0].stem}__{j[3]}" if rea else name(*j) for j in todo]
    if all((OUT / sub / f"{n}.json").exists() for n in names) and (rea or len(names) > 0):
        (OUT / ("REA.ok" if rea else "GRID.ok")).write_text(time.strftime("%F %T"))


def report():
    import collections
    agg = collections.defaultdict(list)
    for f in sorted(glob.glob(str(OUT / "raw" / "*.json"))):
        n = Path(f).stem; atk, d, rest = n.split("__"); a = rest.split("_s")[0]
        agg[(atk, d, a)].append(round(final_asr(f) or 0, 1))
    for k in sorted(agg):
        print(k, sorted(agg[k]))
    for f in sorted(glob.glob(str(OUT / "rea" / "*.json"))):
        h = json.load(open(f))["history"]
        print(Path(f).stem, [round(x["backdoor_asr"] or 0, 1) for x in h])


def quick():
    from src.federated.simulate import run
    for atk in ("lsa", "none"):
        c = make_cfg(atk, "lasa", 0.3, 46)
        c.quick, c.rounds, c.eval_every = True, 2, 1
        r = run(c); print("QUICK", atk, r["history"][-1])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--worker", action="store_true")
    g.add_argument("--rea-worker", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.quick:
        quick()
    else:
        worker(rea=a.rea_worker)


if __name__ == "__main__":
    main()
