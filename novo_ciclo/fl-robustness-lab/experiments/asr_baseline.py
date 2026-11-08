#!/usr/bin/env python
"""Taxa de sucesso de base (sem atacantes) dos ataques direcionados: FedAvg limpo, mesma semente e
protocolo do ragnar_eval, medindo o que um modelo limpo já prevê como alvo.
  trigger: gatilho 4x4 -> classe 0 (backdoor_pixel/pgd, dba, lp)
  edge:    aviões atípicos do teste -> caminhão (edge_case_pgd; dados edge retirados de todos)
Saída: results/ragnar_eval/asr_baseline__{trigger,edge}_s42.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.ragnar_eval import ATTACKS, COMMON
from src.config import ScenarioConfig
from src.federated.simulate import run

for kind, attack in (("trigger", "backdoor_pixel"), ("edge", "edge_case_pgd")):
    out = Path(f"results/ragnar_eval/asr_baseline__{kind}_s42.json")
    if out.exists():
        continue
    cfg = ScenarioConfig(name=f"asr_baseline__{kind}", attack=attack, num_malicious=0,
                         attack_params=dict(ATTACKS[attack]["attack_params"]), defense="fedavg", seed=42, **COMMON)
    out.write_text(json.dumps(run(cfg), indent=2))
