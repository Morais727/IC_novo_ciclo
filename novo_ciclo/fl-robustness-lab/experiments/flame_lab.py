#!/usr/bin/env python
"""Item 1 do plano de 13/10: a linha de base FLAME no laboratório.

No repositório dos autores, o FLAME barrou o LP em 4 de 5 execuções, mas aceitou só ~66% dos
honestos. A pergunta aqui: com o LP completo, quanto backdoor passa e quantos honestos cada defesa
descarta, com não-IID moderado (α = 0,5) e severo (α = 0,1)?

Grade: {sem ataque, lp_full} x {FedAvg, Multi-Krum, FLAME} x α ∈ {0,5; 0,1}; CIFAR-10, 20 clientes,
40 rounds, semente 42 (COMMON de ragnar_eval). Aceitação medida em todos os rounds.

Uso:
    python experiments/flame_lab.py              # coleta o que falta e resume
    python experiments/flame_lab.py --analyze-only
Saídas: results/flame_lab/{raw/*.json, summary.md}
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.ragnar_eval import COMMON, OPTIONAL_ATTACKS
from src.config import ScenarioConfig
from src.federated.simulate import run

DEFENSES = ["fedavg", "multi_krum", "flame"]
ATTACKS = ["none", "lp_full"]
ALPHAS = [0.5, 0.1]
NAMES = {"fedavg": "FedAvg", "multi_krum": "Multi-Krum", "flame": "FLAME", "none": "sem ataque", "lp_full": "LP completo"}


def cfg_for(attack, defense, alpha, seed):
    spec = OPTIONAL_ATTACKS.get(attack, dict(num_malicious=0, attack_params={}))
    return ScenarioConfig(name=f"{attack}__{defense}__a{alpha}", attack=attack, num_malicious=spec["num_malicious"],
                          attack_params=dict(spec["attack_params"]), defense=defense, defense_params={},
                          seed=seed, **{**COMMON, "dirichlet_alpha": alpha})


def summarize(raw: Path) -> str:
    md = ["# FLAME no laboratório (item 1)", "",
          "CIFAR-10, 20 clientes (2 atacantes no LP), 40 rounds, semente 42. Aceitação = fração dos "
          "clientes aceitos pela defesa, somada em todos os rounds. ASR e acurácia: média das duas últimas "
          "avaliações (rounds 35 e 40).", "",
          "| α | defesa | ataque | acurácia | ASR | honestos aceitos | maliciosos aceitos |",
          "|---|---|---|---|---|---|---|"]
    for alpha in ALPHAS:
        for defense in DEFENSES:
            for attack in ATTACKS:
                p = raw / f"{attack}__{defense}__a{alpha}.json"
                if not p.exists():
                    continue
                d = json.load(open(p))
                h = d["history"][-2:]
                ma = sum(e["main_accuracy"] for e in h) / len(h)
                asr = sum(e["backdoor_asr"] for e in h) / len(h) if h[-1]["backdoor_asr"] is not None else None
                acc = d["acceptance"]
                mal = "—" if acc["malicious"] is None else f"{acc['malicious']:.0%}"
                asr_s = "—" if asr is None else f"{asr:.1f}%"
                md.append(f"| {alpha} | {NAMES[defense]} | {NAMES[attack]} | {ma:.1f}% | "
                          f"{asr_s} | {acc['benign']:.0%} | {mal} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/flame_lab")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    raw = Path(args.out_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        t0 = time.time()
        for alpha in ALPHAS:
            for defense in DEFENSES:
                for attack in ATTACKS:
                    cfg = cfg_for(attack, defense, alpha, args.seed)
                    out = raw / f"{cfg.name}.json"
                    if out.exists():
                        continue
                    print(f"\n=== {cfg.name} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
                    json.dump(run(cfg), open(out, "w"), indent=2)
                    (Path(args.out_dir) / "summary.md").write_text(summarize(raw))
    s = summarize(raw)
    (Path(args.out_dir) / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
