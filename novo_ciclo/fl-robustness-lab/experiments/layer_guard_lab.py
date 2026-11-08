#!/usr/bin/env python
"""Item 4 do plano de 13/10 e teste da H4 — critérios FIXADOS ANTES DE RODAR (14/10/2026).

ITEM 4 — resposta cirúrgica por tensor em ciclo fechado (src/defenses/layer_guard.py, sinal do item 2).
  Grade: layer_guard x {sem ataque, lp_full, lp_full_indiv} x α ∈ {0,1; 0,5}, mais FedAvg, Multi-Krum e
  FLAME contra lp_full_indiv (os braços contra lp_full e sem ataque vêm de results/flame_lab/raw).
  CIFAR-10, 20 clientes, 2 atacantes, 40 rounds, semente 42 (COMMON de ragnar_eval).
  Critério (α = 0,1, caso decisivo): ganho real se, contra lp_full E lp_full_indiv,
      ASR <= 10%  E  acurácia limpa do layer_guard sem ataque >= FedAvg sem ataque - 2 pp
      E  fração de parâmetros honestos aceitos >= 95%.
  α = 0,5 é secundário (mesmo critério, reportado à parte).

H4 — atacante que conhece a defesa e soma ruído em L* (attack_params l_noise = ν; norma do ruído =
  ν x norma da componente maliciosa de cada atacante, ruído independente entre atacantes).
  (a) detecção, agregação oráculo, 15 rounds, α = 0,1, ν ∈ {0,5; 1; 2; 4}: AUC de layer_sim (e, secundário,
      layer_norm = z robusto da norma por tensor), como no item 2;
  (b) ciclo fechado, 40 rounds, α = 0,1, mesmos ν, com FedAvg e com layer_guard: ASR.
  Critério: H4 SUSTENTADA (fugir custa backdoor) se, para todo ν com AUC(layer_sim) < 0,80,
      ASR(FedAvg, ν) <= 0,5 x ASR(FedAvg, ν = 0)   [ν = 0 é lp_full_indiv x FedAvg do item 4].
  Senão REFUTADA: o ruído foge do sinal por tensor sem perder o backdoor.

Uso:
    nohup .venv/bin/python -u experiments/layer_guard_lab.py >> results/layer_guard.log 2>&1 &
    python experiments/layer_guard_lab.py --analyze-only
O script pula o que já está em raw/. Saídas: results/layer_guard/{raw/*, summary.md}
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from experiments.layer_similarity import collect, evaluate
from experiments.ragnar_eval import COMMON, OPTIONAL_ATTACKS
from src.config import ScenarioConfig
from src.federated.simulate import run

NUS = [0.5, 1.0, 2.0, 4.0]
FLAME_RAW = Path("results/flame_lab/raw")
NAMES = {"fedavg": "FedAvg", "multi_krum": "Multi-Krum", "flame": "FLAME", "layer_guard": "por tensor (nosso)"}


def cfg_for(attack, defense, alpha, seed, nu=0.0):
    spec = OPTIONAL_ATTACKS.get(attack, dict(num_malicious=0, attack_params={}))
    params = dict(spec["attack_params"])
    name = f"{attack}__{defense}__a{alpha}"
    if nu:
        params["l_noise"] = nu
        name = f"{attack}_nu{nu}__{defense}__a{alpha}"
    return ScenarioConfig(name=name, attack=attack, num_malicious=spec["num_malicious"], attack_params=params,
                          defense=defense, defense_params={}, seed=seed, **{**COMMON, "dirichlet_alpha": alpha})


def jobs(seed):
    j = [cfg_for(a, "layer_guard", 0.1, seed) for a in ("lp_full_indiv", "lp_full", "none")]
    j += [cfg_for("lp_full_indiv", d, 0.1, seed) for d in ("fedavg", "multi_krum", "flame")]
    j += [cfg_for("lp_full_indiv", d, 0.1, seed, nu) for nu in NUS for d in ("fedavg", "layer_guard")]
    j += [cfg_for(a, "layer_guard", 0.5, seed) for a in ("lp_full_indiv", "lp_full", "none")]
    j += [cfg_for("lp_full_indiv", d, 0.5, seed) for d in ("fedavg", "multi_krum", "flame")]
    return j


def load(raw, name):
    for d in (raw, FLAME_RAW):
        if (d / f"{name}.json").exists():
            return json.load(open(d / f"{name}.json"))
    return None


def metrics(d):
    h = d["history"][-2:]
    acc = d["acceptance"]
    return {"acc": np.mean([e["main_accuracy"] for e in h]),
            "asr": np.mean([e["backdoor_asr"] for e in h]) if h[-1]["backdoor_asr"] is not None else None,
            "hon": acc["benign"], "mal": acc["malicious"],
            "hon_par": acc.get("benign_params"), "mal_par": acc.get("malicious_params")}


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.0%}" if pct else f"{v:.1f}%"


def summarize(out):
    raw = out / "raw"
    md = ["# Item 4 (resposta por tensor) e H4 (ruído em L*)", "", "Critérios fixados antes (docstring de "
          "experiments/layer_guard_lab.py). ASR e acurácia: média dos rounds 35 e 40. Aceitos: clientes "
          "inteiros; entre parênteses, fração de parâmetros (só a defesa por tensor aceita parte do cliente).", ""]
    verdict = {}
    for alpha in (0.1, 0.5):
        md += [f"## α = {alpha}", "", "| ataque | defesa | acurácia | ASR | honestos aceitos | maliciosos aceitos |",
               "|---|---|---|---|---|---|"]
        for attack in ("none", "lp_full", "lp_full_indiv"):
            for defense in ("fedavg", "multi_krum", "flame", "layer_guard"):
                d = load(raw, f"{attack}__{defense}__a{alpha}")
                if d is None:
                    continue
                m = metrics(d)
                hon = fmt(m["hon"], True) + (f" ({m['hon_par']:.1%})" if m["hon_par"] is not None else "")
                mal = fmt(m["mal"], True) + (f" ({m['mal_par']:.1%})" if m["mal_par"] is not None else "")
                md.append(f"| {attack} | {NAMES[defense]} | {m['acc']:.1f}% | {fmt(m['asr'])} | {hon} | {mal} |")
        md.append("")
        lg = {a: load(raw, f"{a}__layer_guard__a{alpha}") for a in ("none", "lp_full", "lp_full_indiv")}
        base = load(raw, f"none__fedavg__a{alpha}")
        if all(lg.values()) and base:
            m = {a: metrics(v) for a, v in lg.items()}
            ok_asr = all(m[a]["asr"] <= 10 for a in ("lp_full", "lp_full_indiv"))
            ok_acc = m["none"]["acc"] >= metrics(base)["acc"] - 2
            ok_hon = all(m[a]["hon_par"] >= 0.95 for a in m)
            verdict[alpha] = ok_asr and ok_acc and ok_hon
            md += [f"Critério do item 4 (α = {alpha}): ASR <= 10% {'ok' if ok_asr else 'FALHA'}; acurácia limpa "
                   f"{m['none']['acc']:.1f}% vs FedAvg {metrics(base)['acc']:.1f}% {'ok' if ok_acc else 'FALHA'}; "
                   f"parâmetros honestos >= 95% {'ok' if ok_hon else 'FALHA'} → "
                   f"**{'GANHO REAL' if verdict[alpha] else 'NÃO SUSTENTADO'}**", ""]

    md += ["## H4: ruído em L* (α = 0,1)", "", "| ν | AUC layer_sim | AUC layer_norm | ASR FedAvg | ASR por tensor | acurácia por tensor |",
           "|---|---|---|---|---|---|"]
    det = raw / "h4_detection.csv"
    ev = evaluate(pd.read_csv(det)) if det.exists() else None
    item2 = Path("results/layer_similarity/summary.csv")
    ref = pd.read_csv(item2) if item2.exists() else None
    asr0 = load(raw, "lp_full_indiv__fedavg__a0.1")
    asr0 = metrics(asr0)["asr"] if asr0 else None
    h4_ok, h4_any = True, False
    for nu in [0.0] + NUS:
        if nu == 0:
            a_sim = ref[(ref.ataque == "lp_full_indiv") & (ref.alpha == 0.1) & (ref.escore == "layer_sim")].auc.iloc[0] if ref is not None else None
            a_norm = None
            fa, lgd = load(raw, "lp_full_indiv__fedavg__a0.1"), load(raw, "lp_full_indiv__layer_guard__a0.1")
        else:
            sel = lambda s: ev[(ev.ataque == f"nu{nu}") & (ev.escore == s)].auc
            a_sim = sel("layer_sim").iloc[0] if ev is not None and len(sel("layer_sim")) else None
            a_norm = sel("layer_norm").iloc[0] if ev is not None and len(sel("layer_norm")) else None
            fa = load(raw, f"lp_full_indiv_nu{nu}__fedavg__a0.1")
            lgd = load(raw, f"lp_full_indiv_nu{nu}__layer_guard__a0.1")
        fa_m, lg_m = (metrics(fa) if fa else None), (metrics(lgd) if lgd else None)
        if nu and a_sim is not None and a_sim < 0.8 and fa_m and asr0 is not None:
            h4_any = True
            h4_ok &= fa_m["asr"] <= 0.5 * asr0
        f2 = lambda v: "—" if v is None else f"{v:.2f}"
        md.append(f"| {nu} | {f2(a_sim)} | {f2(a_norm)} | {fmt(fa_m['asr']) if fa_m else '—'} | "
                  f"{fmt(lg_m['asr']) if lg_m else '—'} | {fmt(lg_m['acc']) if lg_m else '—'} |")
    md += ["", f"Critério da H4: {'nenhum ν fugiu do sinal (AUC >= 0,80 em todos)' if not h4_any else ('**SUSTENTADA**' if h4_ok else '**REFUTADA**')} "
           "(para todo ν com AUC < 0,80, ASR FedAvg <= metade da ASR com ν = 0)."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/layer_guard")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        t0 = time.time()
        all_jobs = jobs(args.seed)
        for k, cfg in enumerate(all_jobs):
            if k == 6:   # depois do bloco decisivo do item 4: coleta de detecção da H4 (oráculo, 15 rounds)
                det = raw / "h4_detection.csv"
                if not det.exists():
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dfs = []
                    for nu in NUS:
                        df = collect("lp_full_indiv", 0.1, 15, args.seed, dev, {"l_noise": nu})
                        df["attack"] = f"nu{nu}"
                        dfs.append(df)
                    pd.concat(dfs).to_csv(det, index=False)
                    summarize(out)
            if (raw / f"{cfg.name}.json").exists() or load(raw, cfg.name) is not None:
                continue
            print(f"\n=== [{k + 1}/{len(all_jobs)}] {cfg.name} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
            json.dump(run(cfg), open(raw / f"{cfg.name}.json", "w"), indent=2)
            summarize(out)
    print(summarize(out))


if __name__ == "__main__":
    main()
