#!/usr/bin/env python
"""Confirmação em semente nova (43) do resultado central do LayerGuard v2, fixada antes de rodar (16/10, 03h40).
Pergunta: o resultado da semente 42 (ASR <= 10% contra o LP com ruído, inclusive com norma preservada, e acurácia
limpa igual à do FedAvg) se repete com outra partição dos dados, outra escolha de atacantes e outra inicialização?
Critério: ASR(v2) <= 10% em lp_full_indiv_nu1.0 e lp_full_indiv_nu1.0k com a semente 43, e ASR(FedAvg) >= 40% no
primeiro (o ataque funciona nessa semente). α = 0,1, 40 rounds.
Resultado semente 43 (16/10, 04h08): INCONCLUSIVO — o LP não funcionou nem no FedAvg (ASR 2,9%); v2 2,8% e 2,1%.
Declarado antes de rodar: repete-se com a semente 44, mesmo critério (python experiments/confirm_seed.py 44).
Resultado semente 44 (04h26): FALHA — FedAvg 58,3%, v2 55,0% (ν=1) e 48,3% (norma preservada); a v2 pegou só 1
dos 2 atacantes (o outro teve 95% dos parâmetros aceitos). Réplica extra com a semente 45 (ν=1, FedAvg e v2);
todas as sementes são reportadas.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.final_lab import claim, cfg_for
from src.federated.simulate import run

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 43

OUT = Path("results/final_lab/raw")
RUNS = [("lp_full_indiv_nu1.0", "layer_guard_sn"), ("lp_full_indiv_nu1.0", "fedavg"), ("lp_full_indiv_nu1.0k", "layer_guard_sn")]
for a, d in (RUNS[:2] if SEED >= 45 else RUNS):
    cfg = cfg_for(a, d, 0.1, seed=SEED)
    cfg.name += f"_s{SEED}"
    if (OUT / f"{cfg.name}.json").exists() or not claim(cfg.name):
        continue
    json.dump(run(cfg), open(OUT / f"{cfg.name}.json", "w"), indent=2)
    (OUT / f"{cfg.name}.lock").unlink(missing_ok=True)
