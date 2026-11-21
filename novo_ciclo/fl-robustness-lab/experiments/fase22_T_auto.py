#!/usr/bin/env python
"""Fase 22 (19/11/2026) — treino com T AUTOMÁTICO (params T_auto = k; ver _T_round em src/defenses/layer_guard.py):
o limiar de corroboração sai, a cada rodada, do z robusto das contagens de acusação dos clientes — o mesmo k em
todas as arquiteturas, sem escolher T à mão. Responde à crítica "T é escolhido por arquitetura".

Cenários (os mesmos das varreduras de T à mão):
  paper          ResNet18-GN/CIFAR-10, cenário fiel do LaySelFL (fase 8E), semente --seed (T à mão = 8 funciona)
  vgg19gn        VGG19-GN/CIFAR-10, BadNet, semente 42 (fase 8F; T à mão: só T<=4 contém)
  efficientnetgn EfficientNet-GN/CIFAR-10, BadNet, semente 42 (fase 8F; T à mão: só T=2 contém)
Saída: nome do cenário com a defesa trocada por {defense}_Tauto{k} (e _cap{c} se houver teto).
Uso: CUDA_VISIBLE_DEVICES=6 python -u experiments/fase22_T_auto.py --arch vgg19gn --defense layer_guard_corr --k 3"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
from src.data import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["paper", "vgg19gn", "efficientnetgn"])
    ap.add_argument("--defense", required=True)
    ap.add_argument("--k", type=float, default=None, help="T automático pelo z robusto da contagem")
    ap.add_argument("--top", type=float, default=None, help="T automático: confirma os ⌈β·n⌉ mais acusados (β = fração máx. de atacantes)")
    ap.add_argument("--T", type=int, default=None, help="T fixo (o mesmo em todas as arquiteturas)")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    assert sum(x is not None for x in (a.k, a.top, a.T)) == 1, "use exatamente um de --k, --top, --T"
    if a.k is not None:
        suf, dp = f"_Tauto{a.k:g}" + (f"_cap{a.cap}" if a.cap else ""), {"T_auto": a.k, **({"T_cap": a.cap} if a.cap else {})}
    elif a.top is not None:
        suf, dp = f"_Ttop{a.top:g}", {"T_top": a.top}
    else:
        suf, dp = f"_T{a.T}", {"confirm_tensors": a.T}
    if a.arch == "paper":
        import experiments.fase8e as f8e          # o import fixa o gatilho badnet_cifar
        f8e.SEED = a.seed
        out = f8e.OUT
        name = f"laysel_paper__{a.defense}{suf}__a{f8e.ALPHA}_s{a.seed}"
        mk = lambda: f8e.cfg(a.defense)
    else:
        from experiments import fase8f
        datasets.set_trigger("badnet_cifar")
        out = fase8f.OUT
        name = fase8f.jobname(a.arch, "cifar10", a.defense, "badnet_cifar").replace(f"__{a.defense}__", f"__{a.defense}{suf}__")
        mk = lambda: fase8f.cfg(a.arch, "cifar10", a.defense, "badnet_cifar")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    f = out / "raw" / f"{name}.json"
    if f.exists():
        print(f"[skip] {name} já existe", flush=True); return
    lock = out / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {name}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    c = mk()
    c.defense_params = {**(getattr(c, "defense_params", None) or {}), **dp}
    c.name = name
    print(f"\n=== {name} pid={os.getpid()} params={dp} device={c.device} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
