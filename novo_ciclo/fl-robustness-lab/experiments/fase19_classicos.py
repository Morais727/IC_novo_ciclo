#!/usr/bin/env python
"""Fase 19 (16/11/2026, pedido do usuário): validar a proposta com os ataques que as próprias defesas citadas usam.
Levantamento dos artigos (seções de experimentos): FLAME (constrain-and-scale, DBA, PGD, edge-case, Fang), Krum
(gaussiano, omnisciente), AlignIns (BadNet, DBA, Scaling, PGD, Neurotoxin, ByzMean), FedSurrogate (CBA, DBA,
Neurotoxin, LP), LASA (ruído, sign-flip, Min-Max, Min-Sum, LIE, ByzMean), FLD (constrain-and-scale, DBA, edge-case,
LIE, PGD, Neurotoxin), CosDefense (IPM), RFLDefense (FLIP direcionado, MR/scaling, DBA, EDGE, Neurotoxin). Os que o laboratório já tinha estão na grade M7/M8 (m78_deteccao.py).
Esta fase cobre os que faltavam — Scaling (fator 2, sem limite de norma), Neurotoxin (máscara top-1%), ByzMean (sobre
o LIE), FLIP direcionado (conjunto auxiliar de 500 imagens da classe fonte com o rótulo alvo) — e o ruído gaussiano com as nossas variantes. Desenho fixado antes de rodar: mesmo cenário e sementes do
M4/M7/M8 (SmallCNN/CIFAR-10, 20 clientes, Dirichlet 0,5, 40 rodadas; s42/46/47/48), T = 2 na SmallCNN; FedAvg como
referência sem defesa.
Saída: results/fase19/raw/<ataque>__<defesa>_s<semente>.json. Uso: --attack A --defense D --seed S | --report
"""
import argparse, glob, json, os, sys, time, statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import m4_efeito as m4

OUT = Path("results/fase19")
SEEDS = (42, 46, 47, 48)
NEW = {  # ataque -> (base do m4 para nº de maliciosos e parâmetros, parâmetros extras)
    "backdoor_scaling": ("backdoor_pixel", {"scale": 2.0}),
    "neurotoxin": ("backdoor_pixel", {"neuro_ratio": 0.01}),
    "byzmean": ("lie", {}),
    "gaussian_noise": ("gaussian_noise", {}),
    "flip_targeted": ("edge_case_pgd", {"aux_n": 500}),   # mesmos atacantes, fonte e alvo do edge-case do M4
}
DEFENSES = ["fedavg", "layer_guard_sn", "layer_guard_corr", "layer_guard_corr_soft", "layer_guard_client"]
T_SMALLCNN = 2


def cfg(attack, defense, seed):
    base, extra = NEW[attack]
    c = m4.cfg(base, seed)
    c.attack = attack
    c.attack_params = {**c.attack_params, **extra}
    c.defense = defense
    c.defense_params = {} if defense == "fedavg" else {"confirm_tensors": T_SMALLCNN}
    c.name = f"{attack}__{defense}_s{seed}"
    return c


def report():
    rows = ["| ataque | defesa | n | MA | ASR |", "|---|---|---|---|---|"]
    by = {}
    for f in glob.glob(str(OUT / "raw" / "*.json")):
        n = Path(f).stem; a, rest = n.split("__", 1); d, s = rest.rsplit("_s", 1)
        h = json.load(open(f))["history"][-1]
        by.setdefault((a, d), []).append((h["main_accuracy"], h["backdoor_asr"]))
    for (a, d), v in sorted(by.items()):
        asr = [x[1] for x in v if x[1] is not None]
        rows.append(f"| {a} | {d} | {len(v)} | {st.mean(x[0] for x in v):.1f} | {st.mean(asr):.1f} |" if asr else
                    f"| {a} | {d} | {len(v)} | {st.mean(x[0] for x in v):.1f} | — |")
    (OUT / "RESULTADO.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=list(NEW)); ap.add_argument("--defense"); ap.add_argument("--seed", type=int)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    c = cfg(a.attack, a.defense, a.seed)
    if a.quick:
        c.rounds, c.eval_every = 2, 1
        c.attack_params["start_round"] = 1
        r = run(c); print("QUICK OK", r["history"][-1]); return
    f = OUT / "raw" / f"{c.name}.json"
    if f.exists():
        print(f"[skip] {c.name}", flush=True); return
    lock = f.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {c.name}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    print(f"\n=== {c.name} pid={os.getpid()} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
