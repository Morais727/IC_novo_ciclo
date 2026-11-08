#!/usr/bin/env python
"""Fase 3 (17/10/2026) — pedido: (item 1) confirmar com mais sementes o que a nossa proposta (layer_guard_sn, sem
NENHUMA mudança de parâmetro) só mostrou na semente 42; (item 2) uma defesa eficiente contra o LP com α = 0,1.
Desenho e critérios fixados ANTES de rodar. Saídas em results/fase3/ (raw/, models/, fase3.log).

P1 — confirmação da nossa proposta, sementes 46, 47, 48 (nunca usadas com α = 0,5 nem com ataques de modelo):
  P1a  α = 0,5; ataques none, lp_full_indiv (LP evasivo), lp_full_indiv_nu1.0 (LP + ruído); FedAvg e nossa proposta.
       Critério: ASR <= 10% em todo caso com ASR do FedAvg >= 20% (caso com FedAvg < 20% = inconclusivo, reportado);
       acurácia sem ataque >= FedAvg - 2 pp nas 3 sementes. Na semente 42: 1,4% / 1,8% / 72,2% (FedAvg 72,6%).
  P1b  α = 0,1; ataques de modelo gaussian_noise, ipm, lie (4 de 20 atacantes, como em final_lab); FedAvg e nossa
       proposta. Critério: acurácia final sob ataque >= acurácia da nossa proposta sem ataque (mesma semente,
       results/fase2) - 3 pp nos 9 casos. Na semente 42: 67,2 / 66,1 / 66,9% contra 68,8% sem ataque.

P2 — triagem das defesas que já existem no laboratório contra LP + ruído ν = 1, α = 0,1, nas sementes difíceis 44 e 45
  (desenvolvimento): median, trimmed_mean, foolsgold, boba, multi_metric_whitening e rlr (Robust Learning Rate, novo
  no laboratório, θ = 20% dos clientes como no artigo). Critério para promover uma defesa ao teste completo: ASR <= 10%
  nas duas sementes com acurácia >= FedAvg sob o mesmo ataque - 3 pp. Referências já medidas (ASR/acc): FedAvg
  58,5/63 e 72,4/66; Multi-Krum 72,2/50 e 8,3/49; FLAME 59,4/41 e 0,2/36; FedSurrogate 71,3/51 e 8,5/46.

Defesas novas (reparo do modelo, I-BAU e variantes) ficam em experiments/fase3_repair.py, com critérios próprios.
Ordem da fila: P2 primeiro (informa o desenho), depois P1a, depois P1b.
Uso: python experiments/fase3_lab.py --one   (uma vaga; results/fase3/slots.sh roda 3)
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.fase2_lab import C
from src.federated.simulate import run

OUT = Path("results/fase3")
SEEDS = (46, 47, 48)
LP = "lp_full_indiv_nu1.0"


def jobs():
    # diagnóstico D1 (adendo 1, fase3_repair.py): deltas por cliente nos rounds 10/20/30, sementes 44 e 45
    j = [C(LP, "fedavg_dump", seed=s, dir="results/fase3/dumps", tag=f"{LP}_s{s}") for s in (44, 45)]
    j += [C(LP, d, seed=s) for d in ("median", "trimmed_mean", "rlr", "foolsgold", "boba", "multi_metric_whitening") for s in (44, 45)]
    for s in SEEDS:
        j += [C(a, d, alpha=0.5, seed=s) for a in ("none", "lp_full_indiv", LP) for d in ("fedavg", "layer_guard_sn")]
    for s in SEEDS:
        j += [C(a, d, seed=s) for a in ("gaussian_noise", "ipm", "lie") for d in ("fedavg", "layer_guard_sn")]
    return j


def claim(name):
    lock = OUT / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); return False
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); return claim(name)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def mem_ok(gb=2.5):
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6 >= gb
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FL_SAVE_FINAL", str(OUT / "models"))
    for cfg in jobs():
        if (OUT / "raw" / f"{cfg.name}.json").exists():
            continue
        if (OUT / "STOP3").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        if not claim(cfg.name):
            continue
        print(f"\n=== {cfg.name} pid={os.getpid()}", flush=True)
        json.dump(run(cfg), open(OUT / "raw" / f"{cfg.name}.json", "w"))
        (OUT / "raw" / f"{cfg.name}.lock").unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
