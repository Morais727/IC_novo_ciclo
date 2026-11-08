#!/usr/bin/env python
"""Avaliação final do LayerGuard (src/defenses/layer_guard.py::layerguard) — desenho e critérios FIXADOS
ANTES DE RODAR (16/10/2026, 00h), depois de layer_guard_lab (item 4, H4) e layer_guard_sn_lab.

Defesa: por tensor, marca o par (cliente, tensor) se max(z_sim,0) + max(z_norm,0) > 3 e o tira da média;
depois corta a norma de cada contribuição aceita na mediana daquele tensor; média ponderada.

Cenário decisivo: CIFAR-10, 20 clientes, α = 0,1 (não-IID severo), 40 rounds, semente 42.
Ataques (todos os do laboratório que funcionam como ataque nesse cenário, agrupados):
  backdoor  lp_full, lp_full_indiv, ruído ν=1 (lp_full_indiv_nu1.0), ruído ν=1 com norma preservada
            (lp_full_indiv_nu1.0k), backdoor_pixel, backdoor_pgd (gatilho + projeção de norma), dba
  modelo    gaussian_noise, ipm, lie (não direcionados)
Defesas: FedAvg, Multi-Krum, FLAME, LayerGuard. Ablação: layer_guard_sn (sem corte) e layerguard_cliponly
(só o corte) em parte dos ataques.

Critérios (α = 0,1):
  C1 (backdoor, "a maioria dos casos"): ASR do LayerGuard <= 10% em pelo menos 5 dos 7 ataques de backdoor,
     e ASR média do LayerGuard menor que a de FedAvg, Multi-Krum e FLAME.
  C2 (ataques de modelo): acurácia do LayerGuard >= acurácia limpa do FedAvg - 5 pp em gaussian_noise, ipm e lie.
  C3 (custo sem ataque): acurácia limpa do LayerGuard >= FedAvg limpo - 2 pp.
  A fração de parâmetros honestos aceitos é reportada, mas deixa de ser critério: o limite de 95% era um
  substituto do custo em acurácia, e C3 mede esse custo diretamente (mudança declarada aqui, antes de rodar).
  Ablação, sem critério: quanto cada componente contribui.
Se sobrar tempo: α = 0,5 com LayerGuard nos principais ataques (secundário).

ADENDO 16/10, 01h55 — v1 REFUTADA e v2 declarada ANTES de rodar a v2 nos ataques restantes:
  v1 (com corte por tensor) falhou C3 (65,7% vs 68,7%) e C1 (pixel 94,9%, PGD 75,6%, DBA 47,1%).
  experiments/diag_separate_backdoors.py: nenhum sinal de atualização (Krum, norma, por tensor, acumulado no
  tempo) separa pixel/PGD/DBA dos honestos com α = 0,1 — os mais anômalos são honestos de uma classe só.
  v2 = LayerGuard sem o corte = layer_guard_sn (já avaliado contra a família LP em layer_guard_sn_lab).
  C1–C3 inalterados e aplicados à v2. Fila extra: v2 em pixel, PGD, DBA, gaussian_noise, ipm, lie.

Uso (dois processos dividem a fila; cada treino é reservado por um arquivo .lock):
    nohup .venv/bin/python -u experiments/final_lab.py >> results/final_lab.log 2>&1 &
    python experiments/final_lab.py --analyze-only
Saídas: results/final_lab/{raw/*.json, summary.md}. Reaproveita treinos de layer_guard*, flame_lab.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.layer_guard_lab import metrics
from experiments.layer_guard_sn_lab import cfg_for as sn_cfg
from experiments.ragnar_eval import ATTACKS, COMMON, OPTIONAL_ATTACKS
from src.config import ScenarioConfig
from src.federated.simulate import run

OUT = Path("results/final_lab")
DIRS = [OUT / "raw", Path("results/layer_guard_sn/raw"), Path("results/layer_guard/raw"), Path("results/flame_lab/raw")]
BACKDOOR = ["lp_full", "lp_full_indiv", "lp_full_indiv_nu1.0", "lp_full_indiv_nu1.0k", "backdoor_pixel", "backdoor_pgd", "dba"]
MODEL = ["gaussian_noise", "ipm", "lie"]
BASE = ["fedavg", "multi_krum", "flame"]
NAMES = {"fedavg": "FedAvg", "multi_krum": "Multi-Krum", "flame": "FLAME", "layerguard": "LayerGuard v1 (com corte)",
         "layer_guard_sn": "LayerGuard v2", "layerguard_cliponly": "só corte"}
LG = "layer_guard_sn"   # defesa avaliada pelos critérios (v2, ver adendo)


def cfg_for(attack, defense, alpha=0.1, seed=42):
    if attack.startswith("lp_full_indiv_nu"):
        nu = float(attack.split("nu")[1].rstrip("k"))
        return sn_cfg("lp_full_indiv", defense, alpha, seed, nu=nu, keep=attack.endswith("k"))
    spec = {**ATTACKS, **OPTIONAL_ATTACKS}.get(attack, dict(num_malicious=0, attack_params={}))
    return ScenarioConfig(name=f"{attack}__{defense}__a{alpha}", attack=attack, num_malicious=spec["num_malicious"],
                          attack_params=dict(spec["attack_params"]), defense=defense, defense_params={}, seed=seed,
                          **{**COMMON, "dirichlet_alpha": alpha})


def jobs():
    lg = lambda a, al=0.1: cfg_for(a, "layerguard", al)
    # ordem: o que decide primeiro, linhas de base intercaladas, ablação e α = 0,5 no fim
    j = [lg("none"), lg("lp_full_indiv_nu1.0k"), lg("backdoor_pixel")]
    j += [cfg_for("backdoor_pixel", d) for d in BASE]
    j += [lg("dba"), lg("backdoor_pgd")] + [cfg_for(a, d) for a in ("dba", "backdoor_pgd") for d in BASE]
    j += [lg(a) for a in ("lp_full_indiv", "lp_full_indiv_nu1.0", "lp_full")]
    j += [lg(a) for a in MODEL] + [cfg_for(a, d) for a in MODEL for d in BASE]
    j += [cfg_for(a, "layer_guard_sn") for a in ("backdoor_pixel", "dba", "backdoor_pgd", "gaussian_noise")]
    j += [cfg_for(a, "layerguard_cliponly") for a in ("backdoor_pixel", "lp_full_indiv", "lp_full_indiv_nu1.0k", "none")]
    j += [lg(a, 0.5) for a in ("none", "lp_full_indiv", "lp_full_indiv_nu1.0", "backdoor_pixel", "dba")]
    return j


def jobs_v2():
    j = [cfg_for(a, "layer_guard_sn") for a in ("backdoor_pixel", "dba", "lie", "ipm", "gaussian_noise")]
    j += [cfg_for(a, d) for a in ("lp_full_indiv_nu1.0", "lp_full_indiv_nu1.0k") for d in ("multi_krum", "flame")]
    return j + [cfg_for("backdoor_pgd", "layer_guard_sn")]


def load(name):
    for d in DIRS:
        if (d / f"{name}.json").exists():
            return metrics(json.load(open(d / f"{name}.json")))
    return None


def claim(name):
    lock = OUT / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = int(lock.read_text() or 0)
        try:
            os.kill(pid, 0)
            return False          # outro processo vivo está rodando este treino
        except (ProcessLookupError, ValueError):
            lock.unlink(); return claim(name)   # trava velha (processo morreu)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def table(alpha, attacks, defenses):
    md = ["| ataque | " + " | ".join(NAMES[d] for d in defenses) + " |", "|---" * (len(defenses) + 1) + "|"]
    for a in attacks:
        cells = []
        for d in defenses:
            m = load(f"{a}__{d}__a{alpha}")
            if m is None:
                cells.append("—")
            elif m["asr"] is not None:
                cells.append(f"{m['asr']:.1f}% ({m['acc']:.0f}%)")
            else:
                cells.append(f"{m['acc']:.1f}%")
        md.append(f"| {a} | " + " | ".join(cells) + " |")
    return md


def summarize():
    D = BASE + [LG, "layerguard"]
    md = ["# Avaliação final do LayerGuard", "", "Critérios fixados antes (docstring de experiments/final_lab.py). "
          "Backdoor: ASR (acurácia entre parênteses); modelo e sem ataque: acurácia. Rounds 35 e 40.", "",
          "## α = 0,1", ""]
    md += table(0.1, ["none"] + BACKDOOR + MODEL, D) + [""]
    hp = [load(f"{a}__{LG}__a0.1") for a in ["none"] + BACKDOOR + MODEL]
    hp = [m["hon_par"] for m in hp if m and m["hon_par"] is not None]
    if hp:
        md.append(f"Parâmetros honestos aceitos pela v2: média {np.mean(hp):.1%}, mínimo {min(hp):.1%}.")
    get = lambda a, d: load(f"{a}__{d}__a0.1")
    if all(get(a, d) for a in BACKDOOR for d in BASE + [LG]):
        asr = {d: [get(a, d)["asr"] for a in BACKDOOR] for d in BASE + [LG]}
        n_ok = sum(v <= 10 for v in asr[LG])
        better = all(np.mean(asr[LG]) < np.mean(asr[d]) for d in BASE)
        md.append(f"C1 (v2): <= 10% em {n_ok} de 7; ASR média " + ", ".join(f"{NAMES[d]} {np.mean(asr[d]):.1f}%" for d in BASE + [LG])
                  + f" → **{'OK' if n_ok >= 5 and better else 'FALHA'}**")
    clean = get("none", "fedavg")
    if clean and all(get(a, LG) for a in MODEL):
        ok = all(get(a, LG)["acc"] >= clean["acc"] - 5 for a in MODEL)
        md.append(f"C2 (v2): acurácia sob ataques de modelo " + ", ".join(f"{a} {get(a, LG)['acc']:.1f}%" for a in MODEL)
                  + f" vs FedAvg limpo {clean['acc']:.1f}% - 5 → **{'OK' if ok else 'FALHA'}**")
    if clean and get("none", LG):
        a = get("none", LG)["acc"]
        md.append(f"C3 (v2): acurácia limpa {a:.1f}% vs FedAvg {clean['acc']:.1f}% → **{'OK' if a >= clean['acc'] - 2 else 'FALHA'}**")
    md += ["", "## Ablação (α = 0,1)", ""]
    md += table(0.1, ["none", "backdoor_pixel", "dba", "backdoor_pgd", "gaussian_noise", "lp_full_indiv", "lp_full_indiv_nu1.0k"],
                ["layer_guard_sn", "layerguard_cliponly", "layerguard"]) + [""]
    md += ["## α = 0,5 (secundário)", ""]
    md += table(0.5, ["none", "lp_full_indiv", "lp_full_indiv_nu1.0", "backdoor_pixel", "dba"], ["fedavg", LG, "layerguard"])
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--v2", action="store_true", help="só a fila extra da v2")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        t0, js = time.time(), (jobs_v2() if args.v2 else jobs())
        for k, cfg in enumerate(js):
            if any((d / f"{cfg.name}.json").exists() for d in DIRS) or not claim(cfg.name):
                continue
            print(f"\n=== [{k + 1}/{len(js)}] {cfg.name} pid={os.getpid()} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
            res = run(cfg)
            json.dump(res, open(OUT / "raw" / f"{cfg.name}.json", "w"), indent=2)
            (OUT / "raw" / f"{cfg.name}.lock").unlink(missing_ok=True)
            summarize()
    print(summarize())


if __name__ == "__main__":
    main()
