#!/usr/bin/env python
"""Fase 8F (03/11/2026) — generalidade do LaySelFL e da nossa defesa no cenário do artigo, variando MODELO,
DATASET e GATILHO (mesmo desenho da Fase 8E: 20 nós / 6 maliciosos, α 0,9, 205 rodadas, ataque 80–200,
SGD 0,01→0,001, alvo 4, poison 0,2, k = nº de tensores − 2). Gate (sem defesa) + nossa proposta por combinação.

Gatilhos: badnet_cifar / badnet_fmnist (BadNet estático do artigo) ou a3fl (gatilho adversarial, src/attacks/a3fl.py).
Uso: python experiments/fase8f.py --model vgg19gn   --dataset cifar10 --trigger badnet_cifar  --one
     python experiments/fase8f.py --model resnet18gn --dataset cifar10 --trigger a3fl          --one
     (--missing | --report | --gate-only também). Grava em results/fase8f/raw/ (nome inclui modelo/dataset/gatilho).
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
from src.data import datasets

OUT = Path("results/fase8f")
SH = fase6.SH
ALPHA, SEED = 0.9, 42
DEFENSES = [SH, "layer_guard_sn"]


def _json_default(o):
    """Fallback do json.dump: converte Tensor/np.ndarray (p.ex. o estado do gatilho A3FL em
    attack_params['_a3fl']) em lista, evitando o crash 'Object of type Tensor is not JSON serializable'
    que truncava os resultados dos portões A3FL ao final do treino."""
    if hasattr(o, "detach"):   # torch.Tensor
        return o.detach().cpu().tolist()
    if hasattr(o, "tolist"):   # np.ndarray e afins
        return o.tolist()
    return str(o)


def tag(trigger):
    return "" if trigger in ("badnet_cifar", "badnet_fmnist") else f"_{trigger}"


def jobname(model, dataset, defense, trigger):
    return f"laysel{tag(trigger)}_{model}_{dataset}__{defense}__a{ALPHA}_s{SEED}"


def cfg(model, dataset, defense, trigger):
    c = fase6.cfg("lp_full", defense, ALPHA, SEED)
    c.attack = "laysel"
    c.model, c.dataset = model, dataset
    c.num_malicious = 6
    c.rounds, c.local_epochs, c.eval_every = 205, 1, 10
    c.lr, c.optimizer = 0.01, "sgd"
    c.lr_schedule = [[50, 0.01], [205, 0.001]]
    if model == "vgg19gn":
        # VGG19 é profundo e sem conexões residuais: lr 0,01 diverge (MA travou em 10% = acaso).
        # Correção (04/11): warmup até 0,004 + clipping de gradiente para estabilizar o início.
        c.lr = 0.004
        c.lr_schedule = [[5, 0.0005], [15, 0.002], [150, 0.004], [205, 0.001]]
        c.grad_clip = 5.0
    elif model == "efficientnetgn":
        # EfficientNet-B0 (MBConv+SE, profundo): mesmo risco de divergência que o VGG19 sob lr 0,01.
        # Correção (07/11): warmup + grad_clip, igual ao vgg19gn, para estabilizar o início.
        c.lr = 0.004
        c.lr_schedule = [[5, 0.0005], [15, 0.002], [150, 0.004], [205, 0.001]]
        c.grad_clip = 5.0
    c.attack_params = {"target_label": 4, "start_round": 80, "end_round": 200, "trigger": trigger,
                       "beta": 20.0, "clip": 0.05, "poison_fraction": 0.2, "P": 2, "N": 2}  # k = nº tensores − 2 (default)
    c.name = jobname(model, dataset, defense, trigger)
    return c


def final(model, dataset, defense, trigger):
    f = OUT / "raw" / f"{jobname(model, dataset, defense, trigger)}.json"
    if not f.exists():
        return None
    try:
        h = json.load(open(f))["history"][-1]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None   # json truncado/corrompido (p.ex. crash antigo de serialização) → trata como não-rodado
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def gate(model, dataset, trigger):
    r = final(model, dataset, SH, trigger)
    return None if r is None else r["asr"] >= 20


def jobs(model, dataset, trigger, gate_only=False, only=None):
    j = [SH] if final(model, dataset, SH, trigger) is None else []
    if gate_only:
        return j
    if gate(model, dataset, trigger):
        cand = [only] if only else [d for d in DEFENSES if d != SH]
        j += [d for d in cand if final(model, dataset, d, trigger) is None]
    return j


def report(model, dataset, trigger):
    out = [f"# Fase 8F — LaySelFL ({trigger}) em {model} / {dataset}", ""]
    g = final(model, dataset, SH, trigger)
    out.append("Gate (sem defesa): " + ("ainda não rodou" if g is None else
               f"ASR {g['asr']:.1f}%, acc {g['acc']:.1f}% → {'PEGOU' if g['asr'] >= 20 else 'NÃO pegou'}"))
    if g and g["asr"] >= 20:
        for d in DEFENSES:
            r = final(model, dataset, d, trigger)
            if r:
                nome = "sem defesa" if d == SH else fase6.LG
                out.append(f"- {nome}: ASR {r['asr']:.1f}%, acc {r['acc']:.1f}%")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--defense", default=None, help="roda exatamente esta defesa (ex.: layer_guard_corr), ignorando a lista DEFENSES")
    for f in ("--one", "--report", "--missing", "--gate-only"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    datasets.set_trigger(a.trigger)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(a.model, a.dataset, a.trigger)
    if a.missing:
        j = jobs(a.model, a.dataset, a.trigger, a.gate_only, a.defense); print(len(j)); [print(d) for d in j]; return
    for d in jobs(a.model, a.dataset, a.trigger, a.gate_only, a.defense):
        name = jobname(a.model, a.dataset, d, a.trigger)
        f = OUT / "raw" / f"{name}.json"
        if f.exists():
            continue
        if (OUT / "STOP").exists():
            print("STOP", flush=True); return
        while not mem_ok():
            time.sleep(30)
        lock = OUT / "raw" / f"{name}.lock"
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
        c = cfg(a.model, a.dataset, d, a.trigger)
        print(f"\n=== {c.name} pid={os.getpid()}", flush=True)
        json.dump(run(c), open(f, "w"), default=_json_default)
        lock.unlink(missing_ok=True)
        if a.one:
            return
    if a.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
