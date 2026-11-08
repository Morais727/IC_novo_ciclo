#!/usr/bin/env python
"""Adendo 4 (fase2_lab.py): Neural Cleanse enxuto no servidor (src/defenses/nc.py) aplicado aos modelos finais
salvos. Saída: results/fase2/nc.json (cache) e nc.md. Uso: python experiments/fase2_nc.py [nomes...]"""
import copy, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for, OUT
from src.data.datasets import BackdoorEvalDataset
from src.defenses.nc import detect, nc_prune, nc_unlearn
from src.federated.utils import load_vector
from src.metrics import accuracy, backdoor_success_rate
from src.models.cnn import build_model

CACHE = OUT / "nc.json"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    names = sys.argv[1:] or [f.stem for f in sorted((OUT / "models").glob("*.pt"))]
    for name in names:
        if name in cache and "NC-U" not in cache[name]:
            del cache[name]
        if name in cache:
            continue
        t0 = time.time()
        cfg = cfg_from_name(name); hold, test = holdout_for(cfg)
        bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
        m0 = build_model(cfg.model).to(dev)
        load_vector(m0, torch.load(OUT / "models" / f"{name}.pt", weights_only=True).to(dev))
        sc = lambda m: {"acc": accuracy(m, test, dev), "asr": backdoor_success_rate(m, bd, 0, dev) if bd else None}
        det = detect(m0, hold, dev, seed=cfg.seed)
        r = {"sem reparo": sc(m0), "det": {k: det[k] for k in ("target", "ai", "l1", "succ", "cand")}}
        for drop in (2.0, 5.0):
            mp, info = nc_prune(copy.deepcopy(m0), hold, dev, det=det, max_drop=drop)
            r[f"NC{drop:.0f}"] = {**sc(mp), "pruned": info["pruned"]}
        r["NC-U"] = sc(nc_unlearn(copy.deepcopy(m0), hold, dev, det, lr=0.001, epochs=10, seed=cfg.seed))
        cache[name] = r
        json.dump(cache, open(CACHE, "w"), indent=1)
        d = r["det"]
        print(f"{name}: alvo {d['target']} (cand {d['cand']}, AI {max(d['ai']):.1f}) | sem {r['sem reparo']} | NC2 {r['NC2']} | NC5 {r['NC5']} ({time.time()-t0:.0f}s)", flush=True)
    md = ["# Neural Cleanse enxuto no servidor (adendo 4)", "", "| treino | alvo detectado (AI) | sem reparo | NC2 | NC5 | NC-U | podado (NC5) |", "|---|---|---|---|---|---|---|"]
    cell = lambda v: f"{v['acc']:.1f}%" if v["asr"] is None else f"{v['asr']:.1f}% ({v['acc']:.1f}%)"
    for k, r in sorted(cache.items()):
        d = r["det"]
        md.append(f"| {k} | {d['target']} ({d['ai'][d['cand']]:.1f}) | {cell(r['sem reparo'])} | {cell(r['NC2'])} | {cell(r['NC5'])} | {cell(r['NC-U'])} | {r['NC5']['pruned']:.0%} |")
    (OUT / "nc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
