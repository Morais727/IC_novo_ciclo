#!/usr/bin/env python
"""Fase C (adendo em experiments/fase2_lab.py): aplica os reparos FP, FT e FP+FT (src/defenses/repair.py) aos
modelos finais salvos em results/fase2/models e mede acurácia limpa e ASR no conjunto de teste inteiro.
O server_holdout é reconstruído exatamente como em simulate.run (mesma semente, mesma ordem de sorteio).
Saída: results/fase2/repair.json (cache por modelo) e results/fase2/repair.md.
Uso: python experiments/fase2_repair.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from experiments.fase2_lab import C, OUT
from src.data.datasets import BackdoorEvalDataset
from src.defenses.repair import fine_prune, fine_tune
from src.federated.simulate import _build_data
from src.federated.utils import load_vector
from src.metrics import accuracy, backdoor_success_rate
from src.models.cnn import build_model

CACHE = OUT / "repair.json"


def holdout_for(cfg):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    _, holdout, test_full, _ = _build_data(cfg, rng)
    return holdout, test_full


def cfg_from_name(name):
    stem, seed = name.rsplit("_s", 1)
    attack, defense, a = stem.split("__")
    return C(attack, defense, alpha=float(a[1:]), seed=int(seed))


def evaluate(name, device):
    cfg = cfg_from_name(name)
    holdout, test = holdout_for(cfg)
    bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
    m0 = build_model(cfg.model).to(device)
    load_vector(m0, torch.load(OUT / "models" / f"{name}.pt").to(device))
    res = {}

    def score(m):
        return {"acc": accuracy(m, test, device), "asr": backdoor_success_rate(m, bd, 0, device) if bd else None}

    res["sem reparo"] = score(m0)
    fp, frac = fine_prune(copy.deepcopy(m0), holdout, device)
    res["FP"] = {**score(fp), "pruned": frac}
    res["FT"] = score(fine_tune(copy.deepcopy(m0), holdout, device, seed=cfg.seed))
    res["FP+FT"] = {**score(fine_tune(fp, holdout, device, keep_pruned=fp.pruned_idx, seed=cfg.seed)), "pruned": frac}
    return res


def evaluate2(name, device):
    """Variantes do adendo 2."""
    cfg = cfg_from_name(name)
    holdout, test = holdout_for(cfg)
    bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
    m0 = build_model(cfg.model).to(device)
    load_vector(m0, torch.load(OUT / "models" / f"{name}.pt").to(device))
    score = lambda m: {"acc": accuracy(m, test, device), "asr": backdoor_success_rate(m, bd, 0, device) if bd else None}
    res = {}
    for drop in (2.0, 3.0):
        fp, frac = fine_prune(copy.deepcopy(m0), holdout, device, max_drop=drop)
        res[f"FP{drop:.0f}"] = {**score(fp), "pruned": frac}
        if drop == 2.0:
            res["FP2+FTb"] = {**score(fine_tune(fp, holdout, device, lr=0.001, keep_pruned=fp.pruned_idx, seed=cfg.seed)), "pruned": frac}
    res["FTb"] = score(fine_tune(copy.deepcopy(m0), holdout, device, lr=0.001, seed=cfg.seed))
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for f in sorted((OUT / "models").glob("*.pt")):
        if f.stem not in cache:
            cache[f.stem] = evaluate(f.stem, device)
        if "FP2" not in cache[f.stem]:
            cache[f.stem].update(evaluate2(f.stem, device))
            json.dump(cache, open(CACHE, "w"), indent=1)
        if True:
            print(f.stem, {k: {kk: round(vv, 1) if vv is not None else None for kk, vv in v.items()} for k, v in cache[f.stem].items()}, flush=True)
    variants = ["sem reparo", "FP", "FP3", "FP2", "FT", "FTb", "FP+FT", "FP2+FTb"]
    md = ["# Reparo do modelo final no servidor (Fase C)", "", "ASR (acurácia) no teste inteiro, modelo do round 40.", "",
          "| treino | " + " | ".join(variants) + " | fração podada |", "|---" * (len(variants) + 2) + "|"]
    for k, r in sorted(cache.items()):
        cell = lambda v: f"{v['acc']:.1f}%" if v["asr"] is None else f"{v['asr']:.1f}% ({v['acc']:.0f}%)"
        md.append(f"| {k} | " + " | ".join(cell(r[v]) for v in variants) + f" | {r['FP']['pruned']:.0%} |")
    (OUT / "repair.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
