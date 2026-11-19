#!/usr/bin/env python
"""17/11 — diagnóstico da fase 20: por que o peso gradual perdeu a detecção no ataque padrão (s42, rodada > 130)?
Refaz o cenário fiel (fase8e) com a defesa dada e FL_LOG_COUNTS=1: registra, por rodada, a contagem de acusações de
cada cliente e a norma de cada atualização. Roda a corr-soft (controle) e a corr-soft com peso gradual.
Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase20_diag.py --defense layer_guard_corr_soft_grad"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["FL_LOG_COUNTS"] = "1"
import experiments.fase8e as f8e
from src.federated.simulate import run

ap = argparse.ArgumentParser(); ap.add_argument("--defense", required=True); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
f8e.SEED = a.seed
c = f8e.cfg(a.defense); c.name = f"diag20__{a.defense}__s{a.seed}"
out = Path("results/fase20") / f"{c.name}.json"
r = run(c)
r["malicious_ids"] = r.get("malicious_ids")
json.dump(r, open(out, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("salvo", out)
