#!/usr/bin/env python
"""Adendo A11 (fixado antes de rodar): a varredura validada (tamanhos 3/4/5, passo 2, 100 amostras) custa ~24 min
por round com a GPU dividida — inviável no treino. Configuração enxuta da defesa trigger_scan (tamanhos 3/5, passo
3 com deslocamento 1 — não alinhado ao canto do gatilho —, 50 amostras). Aceita se o 1º colocado por A7 for atacante
em >= 5 dos 6 dumps (s44/s45 × r10/20/30). Limiar thr_top do ciclo fechado = ponto médio entre o menor escore de
atacante 1º colocado e o maior escore do 2º colocado nos 6 dumps (calibrado só em desenvolvimento).
Saída: results/fase3/scan3.json. Uso: python experiments/fase3_scan3.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from src.defenses.nc import _tensors
from src.defenses.dnc import patch_effects
from src.federated.utils import load_vector
from src.models.cnn import build_model

OUT = Path("results/fase3/scan3.json"); res = json.load(open(OUT)) if OUT.exists() else {}
dev = torch.device("cuda")
for r in (10, 20, 30):
    for seed in (44, 45):
        key = f"s{seed}_r{r}"
        if key in res:
            continue
        t0 = time.time()
        hold, _ = holdout_for(cfg_from_name(f"lp_full_indiv_nu1.0__fedavg__a0.1_s{seed}")); X, Y = _tensors(hold)
        d = torch.load(f"results/fase3/dumps/lp_full_indiv_nu1.0_s{seed}_r{r}.pt", weights_only=False)
        fg = build_model("smallcnn").to(dev); load_vector(fg, d["global"].to(dev)); fg.eval(); fis = []
        for i in range(d["deltas"].shape[0]):
            f = build_model("smallcnn").to(dev); load_vector(f, (d["global"] + d["deltas"][i]).to(dev)); f.eval(); fis.append(f)
        G, P = patch_effects(fis, fg, X[:50], Y[:50], dev, sizes=(3, 5), stride=3, offset=1)
        med = G.median(0).values; mad = 1.4826 * (G - med).abs().median(0).values + 0.05; Z = (G - med) / mad
        a7 = Z.flatten(1).max(1).values.numpy(); o = np.argsort(-a7)
        res[key] = {"A7": a7.tolist(), "malicious": d["malicious"], "top1": int(o[0]), "top1_mal": bool(o[0] in d["malicious"]),
                    "v1": float(a7[o[0]]), "v2": float(a7[o[1]]), "s": time.time() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, "1º", o[0], "ATAC" if o[0] in d["malicious"] else "honesto", round(a7[o[0]], 1), "| 2º", round(a7[o[1]], 1),
              "| atac", [round(float(a7[m]), 1) for m in d["malicious"]], f"| {time.time()-t0:.0f}s", flush=True)
ok = sum(v["top1_mal"] for v in res.values())
if len(res) == 6:
    thr = (min(v["v1"] for v in res.values() if v["top1_mal"]) + max(v["v2"] for v in res.values())) / 2
    print(f"A11: 1º colocado atacante em {ok} de 6 → {'ACEITA' if ok >= 5 else 'REJEITADA'}; thr_top = {thr:.2f}")
