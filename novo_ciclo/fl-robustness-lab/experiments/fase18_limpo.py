#!/usr/bin/env python
"""Fase 18 (16/11/2026, pedido do usuário): LINHA DE BASE LIMPA — um treino por modelo/dataset avaliado, sem ataque e
sem defesa (FedAvg puro, 0 clientes maliciosos), no MESMO cenário dos experimentos com ataque. Desenho fixado antes de
rodar:
  resnet18gn_cifar10     cenário fiel do LaySelFL (fase8e.cfg)
  resnet18gn_cifar100    idem, CIFAR-100 (como a fase13)
  resnet18gn_fmnist      fase8f.cfg (gatilho badnet_fmnist só para medir o piso do ASR)
  vgg19gn_cifar10        fase8f.cfg (warmup + grad_clip, igual aos treinos com ataque)
  efficientnetgn_cifar10 fase8f.cfg (idem)
O gatilho continua definido apenas para medir o ASR de um modelo que nunca viu o backdoor (piso). Semente 42 (a do
portão de cada cenário). O SmallCNN/CIFAR-10 já tem linha de base limpa (m4_efeito: asr_baseline__trigger, s42/46/47/48).
Saída: results/fase18/raw/limpo__<kind>__s<seed>.json; modelo final em results/fase18/models/.
Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase18_limpo.py --kind vgg19gn_cifar10 [--seed 42] | --report
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("results/fase18")
KINDS = {
    "resnet18gn_cifar10": ("resnet18gn", "cifar10", "badnet_cifar", "8e"),
    "resnet18gn_cifar100": ("resnet18gn", "cifar100", "badnet_cifar", "8e"),
    "resnet18gn_fmnist": ("resnet18gn", "fmnist", "badnet_fmnist", "8f"),
    "vgg19gn_cifar10": ("vgg19gn", "cifar10", "badnet_cifar", "8f"),
    "efficientnetgn_cifar10": ("efficientnetgn", "cifar10", "badnet_cifar", "8f"),
}


def cfg(kind, seed):
    model, dataset, trigger, base = KINDS[kind]
    if base == "8e":
        import experiments.fase8e as f8e
        f8e.SEED = seed
        c = f8e.cfg("fedavg")
        c.dataset = dataset
    else:
        import experiments.fase8f as f8f
        from src.data import datasets
        datasets.set_trigger(trigger)
        f8f.SEED = seed
        c = f8f.cfg(model, dataset, "fedavg", trigger)
    c.defense = "fedavg"
    c.num_malicious = 0                                  # ninguém ataca
    c.attack_params["start_round"] = 10 ** 6
    c.attack_params["end_round"] = 10 ** 6
    c.seed = seed
    c.name = f"limpo__{kind}__s{seed}"
    return c


def report():
    rows = ["| modelo · dataset | MA final | ASR do gatilho (piso) | s/rodada |", "|---|---|---|---|"]
    for f in sorted(glob.glob(str(OUT / "raw" / "*.json"))):
        r = json.load(open(f)); h = r["history"][-1]
        rows.append(f"| {Path(f).stem} | {h['main_accuracy']:.1f} | {h['backdoor_asr'] or 0:.1f} | {r.get('sec_per_round', 0):.0f} |")
    (OUT / "RESULTADO.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(KINDS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    c = cfg(a.kind, a.seed)
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
    os.environ["FL_SAVE_FINAL"] = str(OUT / "models")
    while not mem_ok():
        time.sleep(30)
    print(f"\n=== {c.name} pid={os.getpid()} model={c.model} dataset={c.dataset} mal={c.num_malicious} defesa={c.defense} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
