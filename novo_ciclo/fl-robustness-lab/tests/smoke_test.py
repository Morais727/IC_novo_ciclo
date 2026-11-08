#!/usr/bin/env python
"""Validação do ambiente: executa cada combinação relevante de ataque x defesa
em modo "quick" (poucos clientes, poucas amostras/round) apenas para provar
que o pipeline roda ponta-a-ponta sem exceções e produz métricas no intervalo
esperado ([0,100] para MA%/BA%). NÃO mede robustez real (rounds/dados
insuficientes para isso) — ver scenarios/*.yaml para as execuções completas.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ScenarioConfig
from src.federated.simulate import run

QUICK_KWARGS = dict(
    num_clients=6,
    dirichlet_alpha=0.5,
    rounds=2,
    local_epochs=1,
    batch_size=16,
    lr=0.05,
    server_holdout_size=30,
    eval_every=1,
    quick=True,
    quick_train_per_client=40,
    quick_test_size=200,
    seed=1,
)

CASES = [
    dict(name="smoke_none_fedavg", num_malicious=0, attack="none", attack_params={}, defense="fedavg", defense_params={}),
    dict(name="smoke_gaussian_krum", num_malicious=2, attack="gaussian_noise", attack_params={"std": 5.0}, defense="krum", defense_params={"num_malicious_assumed": 2}),
    dict(name="smoke_signflip_multikrum", num_malicious=2, attack="sign_flip", attack_params={}, defense="multi_krum", defense_params={"num_malicious_assumed": 2}),
    dict(name="smoke_ipm_median", num_malicious=2, attack="ipm", attack_params={}, defense="median", defense_params={}),
    dict(name="smoke_lie_trimmedmean", num_malicious=2, attack="lie", attack_params={}, defense="trimmed_mean", defense_params={"num_malicious_assumed": 2}),
    dict(name="smoke_minmax_boba", num_malicious=2, attack="minmax", attack_params={}, defense="boba", defense_params={"trim_ratio": 0.2, "tau": 0.15}),
    dict(name="smoke_minsum_boba", num_malicious=2, attack="minsum", attack_params={}, defense="boba", defense_params={"trim_ratio": 0.2, "tau": 0.15}),
    dict(name="smoke_labelflip_foolsgold", num_malicious=2, attack="label_flip", attack_params={"mode": "fixed", "source": 0, "target": 2}, defense="foolsgold", defense_params={}),
    dict(name="smoke_labelflip_ragnar", num_malicious=2, attack="label_flip", attack_params={"mode": "fixed", "source": 0, "target": 2}, defense="ragnar", defense_params={}),
    dict(name="smoke_backdoor_fedavg", num_malicious=1, attack="backdoor_pixel", attack_params={"target_label": 0, "poison_fraction": 0.8}, backdoor_target_label=0, defense="fedavg", defense_params={}),
    dict(name="smoke_backdoor_whitening", num_malicious=1, attack="backdoor_pgd", attack_params={"target_label": 0, "poison_fraction": 0.8, "scale": 2.0}, backdoor_target_label=0, defense="multi_metric_whitening", defense_params={"suspect_fraction": 0.2}),
    dict(name="smoke_backdoor_ragnar", num_malicious=1, attack="backdoor_pixel", attack_params={"target_label": 0, "poison_fraction": 0.8}, backdoor_target_label=0, defense="ragnar", defense_params={}),
]


def main():
    failures = []
    t0 = time.time()
    for case in CASES:
        cfg = ScenarioConfig(**{**QUICK_KWARGS, **case})
        print(f"\n>>> validando: {cfg.name}")
        try:
            result = run(cfg)
            for entry in result["history"]:
                ma = entry["main_accuracy"]
                assert 0.0 <= ma <= 100.0, f"MA fora do intervalo: {ma}"
                if entry["backdoor_asr"] is not None:
                    assert 0.0 <= entry["backdoor_asr"] <= 100.0, f"BA fora do intervalo: {entry['backdoor_asr']}"
            print(f"    OK — última MA={result['history'][-1]['main_accuracy']:.2f}%")
        except Exception:
            print(f"    FALHOU: {cfg.name}")
            traceback.print_exc()
            failures.append(cfg.name)

    dt = time.time() - t0
    print(f"\n=== Validação concluída em {dt:.1f}s — {len(CASES) - len(failures)}/{len(CASES)} casos OK ===")
    if failures:
        print("Casos com falha:", failures)
        sys.exit(1)
    print("Ambiente VALIDADO: todas as combinações ataque x defesa executam ponta-a-ponta sem erro.")


if __name__ == "__main__":
    main()
