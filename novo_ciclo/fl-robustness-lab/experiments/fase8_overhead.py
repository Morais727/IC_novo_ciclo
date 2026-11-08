#!/usr/bin/env python
"""Fase 8B (27/10/2026) — overhead de processamento das formas de agregar. Descritiva, sem hipótese; desenho
FIXADO ANTES DE RODAR (results/fase8/PLANO.md). Saídas: results/fase8/overhead.jsonl e OVERHEAD.md.

Para cada modelo × conjunto: monta os clientes como num treino real (fase6.cfg, sem ataque, α 0,5, semente 46),
treina UMA rodada dos 20 clientes honestos (cronometrada: custo do treino local) e, sobre esses mesmos 20 deltas,
cronometra só a função de agregação de cada defesa (DEFENSE_REGISTRY, parâmetros iguais aos da Fase 7): 1 chamada
de aquecimento + 5 cronometradas (mediana), com o estado da defesa mantido entre chamadas (regime de rodada
avançada, round_idx = 10), no dispositivo do pipeline (GPU) e forçando CPU. Também registra o pico de memória
residente do processo durante as chamadas. Deltas honestos: o tempo depende das dimensões, não do ataque.
Rodar com a máquina sem outros treinos (os números da Fase 7 misturam treino e 7 processos dividindo a GPU).
Uso: python experiments/fase8_overhead.py --model resnet18gn --dataset cifar10 | --report
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6
from src.attacks.registry import ATTACK_REGISTRY
from src.defenses.registry import DEFENSE_REGISTRY
from src.federated.simulate import _build_data, _resolve_device, _train_round, NUM_CLASSES
from src.federated.utils import flatten
from src.models.cnn import build_model

OUT = Path("results/fase8")
DEFENSES = ("fedavg", "layer_guard_sn", "lasa", "alignins", "rfldefense", "flame", "multi_krum", "fedsurrogate")
NAMES = {"fedavg": "FedAvg (sem defesa)", "layer_guard_sn": "Nossa proposta", "lasa": "LASA", "alignins": "AlignIns",
         "rfldefense": "RFLDefense", "flame": "FLAME", "multi_krum": "Multi-Krum", "fedsurrogate": "FedSurrogate"}
REPS = 5


def vmhwm_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024


def reset_hwm():
    try:
        open("/proc/self/clear_refs", "w").write("5")
    except OSError:
        pass


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(model, dataset):
    cfg = fase6.cfg("none", "fedavg", 0.5, 46)
    cfg.model, cfg.dataset = model, dataset
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_ctor = lambda: build_model(cfg.model, dataset=cfg.dataset)
    client_data, server_holdout, _, _ = _build_data(cfg, rng)
    global_vector = flatten(model_ctor().to(device)).cpu()
    sync(); t0 = time.perf_counter()
    deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg, set(),
                                   ATTACK_REGISTRY["none"], cfg.attack_params)
    sync(); train_s = time.perf_counter() - t0
    out = dict(model=model, dataset=dataset, params=int(deltas.shape[1]),
               tensors=len(list(model_ctor().parameters())), clients=int(deltas.shape[0]),
               train_round_s=train_s, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               when=time.strftime("%F %T"), aggregation={})
    print(f"[{model}/{dataset}] {out['params']} parâmetros; treino da rodada {train_s:.1f} s", flush=True)
    for dev in (device, torch.device("cpu")) if device.type == "cuda" else (device,):
        for d in DEFENSES:
            params = fase6.cfg("lp_full_indiv_nu1.0", d, 0.5, 46).defense_params if d != "fedavg" else {}
            ctx = {"model_ctor": model_ctor, "device": dev, "num_classes": NUM_CLASSES,
                   "server_holdout": server_holdout, "global_vector": global_vector, "local_epochs": cfg.local_epochs,
                   "batch_size": cfg.batch_size, "lr": cfg.lr, "ref_steps": 1, "state": {}, "round_idx": 10,
                   "_malicious_ids_diag_only": []}
            fn, times = DEFENSE_REGISTRY[d], []
            try:
                fn(deltas.clone(), weights.clone(), ctx, dict(params))           # aquecimento
                reset_hwm(); rss0 = vmhwm_mb()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(REPS):
                    x, w = deltas.clone(), weights.clone()
                    sync(); t0 = time.perf_counter()
                    fn(x, w, ctx, dict(params))
                    sync(); times.append(time.perf_counter() - t0)
                r = dict(median_s=statistics.median(times), times_s=times, rss_peak_mb=vmhwm_mb(), rss_base_mb=rss0,
                         gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)
            except Exception as e:                                                   # ex.: falta de memória na GPU
                r = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out["aggregation"][f"{d}@{dev.type}"] = r
            print(f"  {d:15s} {dev.type}: " + (f"{r['median_s']:.3f} s" if "median_s" in r else r["error"]), flush=True)
    with open(OUT / "overhead.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


def report():
    rows = [json.loads(l) for l in open(OUT / "overhead.jsonl")]
    out = ["# Fase 8B — overhead de agregação (mediana de 5 chamadas, 20 clientes, deltas de uma rodada real)", "",
           "Célula: segundos de agregação por rodada (% do tempo de treino da rodada). Treino = os 20 clientes de uma",
           "rodada, em sequência, na GPU.", ""]
    for dev in ("cuda", "cpu"):
        out += [f"## Agregação em {'GPU' if dev == 'cuda' else 'CPU'}", "",
                "| modelo / dados | parâmetros | treino da rodada (s) | " + " | ".join(NAMES[d] for d in DEFENSES) + " |",
                "|---|---|---|" + "---|" * len(DEFENSES)]
        for r in rows:
            cells = []
            for d in DEFENSES:
                a = r["aggregation"].get(f"{d}@{dev}")
                if a is None:
                    cells.append("—")
                elif "error" in a:
                    cells.append("erro: " + a["error"].split(":")[0])
                else:
                    cells.append(f"{a['median_s']:.2f} ({100 * a['median_s'] / r['train_round_s']:.1f}%)")
            out.append(f"| {r['model']} / {r['dataset']} | {r['params'] / 1e6:.2f} M | {r['train_round_s']:.0f} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Nossa proposta: fração do tempo da rodada (agregação em GPU / (treino + agregação))", ""]
    for r in rows:
        a = r["aggregation"].get("layer_guard_sn@cuda", {})
        if "median_s" in a:
            out.append(f"- {r['model']} / {r['dataset']}: {100 * a['median_s'] / (r['train_round_s'] + a['median_s']):.1f}%"
                       f" ({a['median_s']:.2f} s de agregação, {r['train_round_s']:.0f} s de treino; pico de memória"
                       f" {a['rss_peak_mb'] / 1024:.1f} GB)")
    txt = "\n".join(out) + "\n"
    (OUT / "OVERHEAD.md").write_text(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report() if a.report else measure(a.model, a.dataset)
