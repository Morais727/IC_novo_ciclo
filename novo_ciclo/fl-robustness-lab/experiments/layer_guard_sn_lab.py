#!/usr/bin/env python
"""Similaridade + norma por tensor (layer_guard_sn) contra o atacante com ruído em L* — critérios FIXADOS
ANTES DE RODAR (15/10/2026), depois da H4 refutada em experiments/layer_guard_lab.py.

Defesa: src/defenses/layer_guard.py::layer_guard_sn (marca o tensor g do cliente i se
max(z_sim, 0) + max(z_norm, 0) > 3; só esse tensor sai da média).
Ataques, α = 0,1 (decisivo): lp_full, lp_full_indiv, lp_full_indiv + ruído ν ∈ {0,5; 1; 2; 4} e o novo
contra-ataque "ruído com norma preservada" (l_noise_keepnorm: soma o ruído e reescala a componente
maliciosa de volta à norma original), ν ∈ {1; 2}, também com FedAvg para medir quanto backdoor ele perde.
α = 0,5 (secundário, por último): sem ataque, lp_full_indiv, ν = 1.

Critério A (defesa combinada, α = 0,1): ASR <= 10% contra lp_full, lp_full_indiv e TODOS os ν do ruído
simples; acurácia sem ataque >= FedAvg sem ataque - 2 pp; >= 95% dos parâmetros honestos aceitos.
Critério B (contra-ataque com norma preservada): a defesa resiste se, para cada ν com ASR(layer_guard_sn) > 10%,
ASR(FedAvg, mesmo ν) <= 0,5 x ASR(FedAvg, ν = 0) = 35% — ou seja, fugir dos dois sinais custa o backdoor.

CIFAR-10, 20 clientes, 2 atacantes, 40 rounds, semente 42.
Uso: nohup .venv/bin/python -u experiments/layer_guard_sn_lab.py >> results/layer_guard_sn.log 2>&1 &
     python experiments/layer_guard_sn_lab.py --analyze-only
Saídas: results/layer_guard_sn/{raw/*.json, summary.md}. Linhas de base (FedAvg etc.) vêm de
results/layer_guard/raw e results/flame_lab/raw.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.layer_guard_lab import fmt, metrics
from experiments.ragnar_eval import COMMON, OPTIONAL_ATTACKS
from src.config import ScenarioConfig
from src.federated.simulate import run

DIRS = [Path("results/layer_guard_sn/raw"), Path("results/layer_guard/raw"), Path("results/flame_lab/raw")]


def cfg_for(attack, defense, alpha, seed=42, nu=0.0, keep=False):
    spec = OPTIONAL_ATTACKS.get(attack, dict(num_malicious=0, attack_params={}))
    params = dict(spec["attack_params"])
    name = attack + (f"_nu{nu}" if nu else "") + ("k" if keep else "")
    if nu:
        params["l_noise"] = nu
    if keep:
        params["l_noise_keepnorm"] = True
    return ScenarioConfig(name=f"{name}__{defense}__a{alpha}", attack=attack, num_malicious=spec["num_malicious"],
                          attack_params=params, defense=defense, defense_params={}, seed=seed,
                          **{**COMMON, "dirichlet_alpha": alpha})


def jobs():
    j = [cfg_for("lp_full_indiv", "layer_guard_sn", 0.1, nu=1.0), cfg_for("none", "layer_guard_sn", 0.1)]
    j += [cfg_for("lp_full_indiv", "layer_guard_sn", 0.1, nu=nu) for nu in (2.0, 0.5, 4.0)]
    j += [cfg_for("lp_full_indiv", "layer_guard_sn", 0.1), cfg_for("lp_full", "layer_guard_sn", 0.1)]
    for nu in (1.0, 2.0):
        j += [cfg_for("lp_full_indiv", d, 0.1, nu=nu, keep=True) for d in ("layer_guard_sn", "fedavg")]
    j += [cfg_for("none", "layer_guard_sn", 0.5), cfg_for("lp_full_indiv", "layer_guard_sn", 0.5),
          cfg_for("lp_full_indiv", "layer_guard_sn", 0.5, nu=1.0)]
    return j


def load(name):
    for d in DIRS:
        if (d / f"{name}.json").exists():
            return metrics(json.load(open(d / f"{name}.json")))
    return None


def summarize():
    rows = [("sem ataque", "none"), ("LP dos autores", "lp_full"), ("LP evasivo", "lp_full_indiv")]
    rows += [(f"ruído ν = {nu}", f"lp_full_indiv_nu{nu}") for nu in (0.5, 1.0, 2.0, 4.0)]
    rows += [(f"ruído ν = {nu}, norma preservada", f"lp_full_indiv_nu{nu}k") for nu in (1.0, 2.0)]
    md = ["# Similaridade + norma por tensor (layer_guard_sn)", "",
          "Critérios fixados antes (docstring de experiments/layer_guard_sn_lab.py). ASR e acurácia: rounds 35 e 40.", ""]
    for alpha in (0.1, 0.5):
        md += [f"## α = {alpha}", "", "| ataque | ASR FedAvg | ASR só similaridade | ASR sim + norma | acurácia sim + norma | "
               "parâmetros honestos aceitos (sim + norma) |", "|---|---|---|---|---|---|"]
        for label, a in rows:
            r = {d: load(f"{a}__{d}__a{alpha}") for d in ("fedavg", "layer_guard", "layer_guard_sn")}
            if r["layer_guard_sn"] is None:
                continue
            g = lambda d, k: "—" if r[d] is None or r[d][k] is None else fmt(r[d][k])
            hp = r["layer_guard_sn"]["hon_par"]
            md.append(f"| {label} | {g('fedavg', 'asr')} | {g('layer_guard', 'asr')} | {g('layer_guard_sn', 'asr')} | "
                      f"{g('layer_guard_sn', 'acc')} | {'—' if hp is None else f'{hp:.1%}'} |")
        md.append("")
    sn = lambda a: load(f"{a}__layer_guard_sn__a0.1")
    need = ["lp_full", "lp_full_indiv"] + [f"lp_full_indiv_nu{nu}" for nu in (0.5, 1.0, 2.0, 4.0)]
    if all(sn(a) for a in need + ["none"]):
        base = load("none__fedavg__a0.1")
        ok1 = all(sn(a)["asr"] <= 10 for a in need)
        ok2 = sn("none")["acc"] >= base["acc"] - 2
        ok3 = all(sn(a)["hon_par"] >= 0.95 for a in need + ["none"])
        md.append(f"Critério A (α = 0,1): ASR <= 10% em todos {'ok' if ok1 else 'FALHA'}; acurácia limpa "
                  f"{sn('none')['acc']:.1f}% vs FedAvg {base['acc']:.1f}% {'ok' if ok2 else 'FALHA'}; parâmetros honestos "
                  f">= 95% {'ok' if ok3 else 'FALHA'} → **{'SUSTENTADO' if ok1 and ok2 and ok3 else 'NÃO SUSTENTADO'}**")
    keep = [(sn(f"lp_full_indiv_nu{nu}k"), load(f"lp_full_indiv_nu{nu}k__fedavg__a0.1")) for nu in (1.0, 2.0)]
    if all(a and b for a, b in keep):
        ref = load("lp_full_indiv__fedavg__a0.1")["asr"]
        ok = all(b["asr"] <= 0.5 * ref for a, b in keep if a["asr"] > 10)
        md.append(f"Critério B (norma preservada): {'**A DEFESA RESISTE**' if ok else '**A DEFESA É FURADA**'} "
                  f"(ASR FedAvg ν = 0: {ref:.1f}%; limite 50% disso quando a defesa deixa passar > 10%)")
    Path("results/layer_guard_sn/summary.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    DIRS[0].mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        t0, js = time.time(), jobs()
        for k, cfg in enumerate(js):
            if any((d / f"{cfg.name}.json").exists() for d in DIRS):
                continue
            print(f"\n=== [{k + 1}/{len(js)}] {cfg.name} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
            json.dump(run(cfg), open(DIRS[0] / f"{cfg.name}.json", "w"), indent=2)
            summarize()
    print(summarize())


if __name__ == "__main__":
    main()
