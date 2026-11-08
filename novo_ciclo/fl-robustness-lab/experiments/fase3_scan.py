#!/usr/bin/env python
"""Adendo A6 (fase3_repair.py) — varredura de quadrados de cor sólida por cliente (src/defenses/dnc.py::patch_scan),
dumps s44 e s45, round 20 (depois 10 e 30 se passar). Fixado antes: tamanhos 3/4/5, 8 cores válidas, passo 2;
escore do cliente = max_c max_patch G. Critério para ciclo fechado: os 2 atacantes nos 2 maiores escores em
>= 4 de 6 (semente, round). Ressalva declarada: supõe gatilho em forma de quadrado.
Saída: results/fase3/scan.jsonl (uma linha por cliente). Uso: python experiments/fase3_scan.py [rounds...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from src.defenses.nc import _tensors
from src.defenses.dnc import patch_scan
from src.federated.utils import load_vector
from src.models.cnn import build_model

OUT = Path("results/fase3/scan.jsonl")
done = {(j["seed"], j["round"], j["client"]) for j in map(json.loads, open(OUT))} if OUT.exists() else set()
dev = torch.device("cuda")
for r in [int(a) for a in sys.argv[1:]] or [20]:
    for seed in (44, 45):
        hold, _ = holdout_for(cfg_from_name(f"lp_full_indiv_nu1.0__fedavg__a0.1_s{seed}")); X, Y = _tensors(hold)
        d = torch.load(f"results/fase3/dumps/lp_full_indiv_nu1.0_s{seed}_r{r}.pt", weights_only=False)
        fg = build_model("smallcnn").to(dev); load_vector(fg, d["global"].to(dev)); fg.eval()
        fi = build_model("smallcnn").to(dev); fi.eval()
        for i in range(d["deltas"].shape[0]):
            if (seed, r, i) in done:
                continue
            t0 = time.time(); load_vector(fi, (d["global"] + d["deltas"][i]).to(dev))
            g, arg = patch_scan(fi, fg, X, Y, dev, stride=2)
            with open(OUT, "a") as f:
                f.write(json.dumps({"seed": seed, "round": r, "client": i, "mal": i in d["malicious"], "G": g, "arg": arg, "s": time.time() - t0}) + "\n")
