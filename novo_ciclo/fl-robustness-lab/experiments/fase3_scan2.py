#!/usr/bin/env python
"""Adendo A7 (fase3_repair.py) — fixado ANTES de rodar, depois de ver na s44 r20 (varredura A6, 19 de 20 clientes):
atacantes em 1º e 4º pelo G máximo, ambos com o MESMO quadrado (4x4 claro no canto → classe 0, o gatilho real),
enquanto os honestos chegam a quadrados escuros 5x5 variados (atalhos naturais que mexem com vários modelos).
Escore A7 do cliente i = max sobre (quadrado, classe) de [G_i − mediana_j G_j] / MAD_j (comparação ENTRE clientes no
mesmo quadrado e classe). Critério para ciclo fechado: 2 atacantes nos 2 maiores escores em s44 e s45 (round 20),
depois confirmado nos rounds 10 e 30 (>= 4 de 6 no total). Também reportado o G máximo bruto (A6).
Adendo A8 (fixado às 13h de 17/10, depois de ver só a s44 r20, ANTES de qualquer número da s45): no quadrado do
gatilho real os dois atacantes têm z alto e os honestos não; os honestos que passam à frente reagem, cada um
SOZINHO, a quadrados diferentes. Regra A8: para cada (quadrado, classe), escore = o 2º maior z entre os clientes
(exige >= 2 clientes reagindo juntos); candidato = (quadrado, classe) de maior escore; clientes marcados = os de
z > 3 nesse candidato. Critério: em s45 r20 (validação) os marcados são exatamente os 2 atacantes; depois rounds
10 e 30 nas duas sementes (>= 4 de 6 no total). Limitação declarada: precisa de >= 2 atacantes com o mesmo gatilho.
RESULTADO A8 (17/10, 13h10): s44 r20 OK (marca exatamente os 2 atacantes, candidato = gatilho real); s45 r20 FALHA
(só 1 atacante reage ao gatilho, z 8,3; o outro não; a regra marca [3, 4, 14] num quadrado qualquer) → A8 REFUTADA.
Adendo A10 (fixado ANTES de ver os rounds 10 e 30, que estavam começando): nos dois r20 o 1º colocado pelo escore A7
foi um atacante. Critério A10: o 1º colocado por A7 é atacante em >= 5 dos 6 (semente, round ∈ {10, 20, 30}).
Se passar, a defesa em ciclo fechado excluiria o 1º colocado de cada varredura quando A7 > 8 (limiar declarado
agora; nos r20 os atacantes 1º colocados tinham 13,2 e o valor da s45 está em scan2.json).
Saída: results/fase3/scan2.json. Uso: python experiments/fase3_scan2.py [rounds...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from src.defenses.nc import _tensors
from src.defenses.dnc import patch_effects
from src.federated.utils import load_vector
from src.models.cnn import build_model

OUT = Path("results/fase3/scan2.json")
res = json.load(open(OUT)) if OUT.exists() else {}
dev = torch.device("cuda")
for r in [int(a) for a in sys.argv[1:]] or [20]:
    for seed in (44, 45):
        key = f"s{seed}_r{r}"
        if key in res:
            continue
        t0 = time.time()
        hold, _ = holdout_for(cfg_from_name(f"lp_full_indiv_nu1.0__fedavg__a0.1_s{seed}")); X, Y = _tensors(hold)
        d = torch.load(f"results/fase3/dumps/lp_full_indiv_nu1.0_s{seed}_r{r}.pt", weights_only=False)
        fg = build_model("smallcnn").to(dev); load_vector(fg, d["global"].to(dev)); fg.eval()
        fis = []
        for i in range(d["deltas"].shape[0]):
            f = build_model("smallcnn").to(dev); load_vector(f, (d["global"] + d["deltas"][i]).to(dev)); f.eval(); fis.append(f)
        G, patches = patch_effects(fis, fg, X, Y, dev)
        torch.save({"G": G.half(), "patches": patches, "malicious": d["malicious"]}, f"results/fase3/scan2_{key}.pt")
        med = G.median(0).values; mad = 1.4826 * (G - med).abs().median(0).values + 0.05
        Z = (G - med) / mad
        a7, a6 = Z.flatten(1).max(1).values, G.flatten(1).max(1).values
        arg = Z.flatten(1).argmax(1)
        mal = d["malicious"]
        rank = lambda s: sorted((torch.argsort(-s).tolist().index(a) + 1) for a in mal)
        res[key] = {"malicious": mal, "A7": a7.tolist(), "A6": a6.tolist(),
                    "A7_arg": [(patches[int(k) // G.shape[2]], int(k) % G.shape[2]) for k in arg],
                    "rank_A7": rank(a7), "rank_A6": rank(a6), "s": time.time() - t0}
        json.dump(res, open(OUT, "w"), indent=1)
        print(key, "postos A7", res[key]["rank_A7"], "| A6", res[key]["rank_A6"], "|", round(time.time() - t0), "s", flush=True)
