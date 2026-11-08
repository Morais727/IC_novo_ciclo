#!/usr/bin/env python
"""Fase 3, item 2 — reparo do modelo final por perturbação universal densa (src/defenses/uap.py), fixado ANTES de rodar.

Hipótese H6: o LP escapa do NC porque o NC procura máscara pequena; uma perturbação universal densa, limitada só em
norma L2, acha a direção do backdoor do LP, e desaprendê-la (I-BAU simplificado) remove o backdoor sem saber gatilho
nem alvo, com as mesmas 100 amostras limpas do servidor.

Desenvolvimento (modelos FedAvg salvos em results/fase2/models, sementes 42/44/45): grade eps ∈ {4, 8, 16} ×
lr ∈ {0,001; 0,005}, rounds = 10. Escolha: a configuração que cumpre D em mais casos; empate → menor eps, depois
menor lr. D (por caso): LP + ruído ν=1 com ASR <= 10% e acurácia >= sem reparo - 2 pp (s42, s44, s45), e modelo
limpo s42 com acurácia >= sem reparo - 2 pp. Também reportados (sem critério): pixel, DBA, PGD, lp_full, lp_full_indiv s42.
Teste (só a configuração escolhida, sementes 46/47/48): T1 LP + ruído (FedAvg e nossa proposta) ASR <= 10% em
todo caso com ASR sem reparo >= 40%; T2 modelo limpo (FedAvg e nossa proposta) acurácia >= sem reparo - 2 pp;
reportados pixel e DBA. Se nenhuma configuração cumprir D em >= 3 dos 4 casos de desenvolvimento, H6 é refutada e
NÃO vai ao teste.
Adendo A1 (17/10, depois de ver só o 1º modelo de dev do I-BAU: eps 4/8 AUMENTA o ASR, eps 16 derruba a acurácia
para 35–47%): mesma avaliação para o ANP enxuto (src/defenses/anp.py), grade eps ∈ {0,2; 0,4}, thr = 0,2, α = 0,2,
500 iterações; mesmo critério D e mesma regra de escolha e de refutação. Modo: anp_dev / anp_test.
Adendo A2 — diagnóstico D1 (experiments/fase3_clientnc.py): NC enxuto no modelo de CADA cliente (global + Δ_i),
deltas salvos nos rounds 10/20/30 (fedavg_dump, sementes 44 e 45). Critério para virar defesa em ciclo fechado:
AUC atacante × honesto >= 0,9 em cada semente (rounds juntos) e os 2 atacantes entre os 2 maiores escores em
>= 4 dos 6 pares (semente, round).
Adendo A3 (17/10, depois de ver: NC por cliente no round 10 não destaca os atacantes; com o gatilho verdadeiro,
honestos com quase só a classe 0 têm ASR alto porque chutam 0 para tudo, e o efeito descontado do gatilho separa
os atacantes nos rounds 10–20): busca DIFERENCIAL por cliente, experiments/fase3_dnc.py (critério lá).
Uso: python experiments/fase3_repair.py dev | test | anp_dev | anp_test      Saída: results/fase3/repair_{dev,test}.json/.md
"""
import copy, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for
from src.data.datasets import BackdoorEvalDataset
from src.defenses.uap import ibau_lite
from src.defenses.anp import anp
from src.federated.utils import load_vector
from src.metrics import accuracy, backdoor_success_rate
from src.models.cnn import build_model

M2, OUT = Path("results/fase2/models"), Path("results/fase3")
GRID = [(e, lr) for e in (4.0, 8.0, 16.0) for lr in (0.001, 0.005)]
LP = "lp_full_indiv_nu1.0"
DEV_D = [f"{LP}__fedavg__a0.1_s{s}" for s in (42, 44, 45)] + ["none__fedavg__a0.1_s42"]
DEV_X = [f"{a}__fedavg__a0.1_s42" for a in ("backdoor_pixel", "dba", "backdoor_pgd", "lp_full", "lp_full_indiv")]
TEST = [f"{a}__{d}__a0.1_s{s}" for s in (46, 47, 48) for a in (LP, "none") for d in ("fedavg", "layer_guard_sn")] + \
       [f"{a}__fedavg__a0.1_s{s}" for s in (46, 47, 48) for a in ("backdoor_pixel", "dba")]


def evaluate(name, configs, dev, models=M2):
    cfg = cfg_from_name(name); hold, test = holdout_for(cfg)
    bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
    m0 = build_model(cfg.model).to(dev)
    load_vector(m0, torch.load(models / f"{name}.pt", weights_only=True).to(dev))
    sc = lambda m: {"acc": accuracy(m, test, dev), "asr": backdoor_success_rate(m, bd, 0, dev) if bd else None}
    r = {"sem reparo": sc(m0)}
    for e, lr in configs:
        if lr is None:   # ANP: (eps, None)
            m = anp(copy.deepcopy(m0), hold, dev, eps=e, seed=cfg.seed)
            r[f"anp{e:g}"] = {**sc(m), "pruned": m.anp_pruned}
        else:
            r[f"e{e:g}_lr{lr:g}"] = sc(ibau_lite(copy.deepcopy(m0), hold, dev, eps=e, lr=lr, seed=cfg.seed))
    return r


def ok(r, k):
    a, b = r[k], r["sem reparo"]
    return a["acc"] >= b["acc"] - 2 and (a["asr"] is None or a["asr"] <= 10)


def table(res, keys):
    cell = lambda v: f"{v['acc']:.1f}%" if v["asr"] is None else f"{v['asr']:.1f}% ({v['acc']:.1f}%)"
    md = ["| modelo | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    md += [f"| {n} | " + " | ".join(cell(r[k]) for k in keys) + " |" for n, r in res.items()]
    return md


def main():
    mode = sys.argv[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = OUT / f"repair_{mode}.json"
    res = json.load(open(f)) if f.exists() else {}
    anp_mode = mode.startswith("anp")
    grid = [(0.2, None), (0.4, None)] if anp_mode else GRID
    kname = lambda c: f"anp{c[0]:g}" if c[1] is None else f"e{c[0]:g}_lr{c[1]:g}"
    chf = OUT / ("anp_choice.json" if anp_mode else "repair_choice.json")
    if mode.endswith("dev"):
        names, configs = DEV_D + DEV_X, grid
    else:
        names, configs = TEST, [tuple(json.load(open(chf))["config"])]
    for n in names:
        if n not in res:
            t0 = time.time(); res[n] = evaluate(n, configs, dev); json.dump(res, open(f, "w"), indent=1)
            print(n, {k: (round(v["asr"], 1) if v["asr"] is not None else None, round(v["acc"], 1)) for k, v in res[n].items()},
                  f"{time.time()-t0:.0f}s", flush=True)
    keys = ["sem reparo"] + [kname(c) for c in configs]
    md = [f"# Reparo por perturbação universal (fase 3, {mode})", ""] + table(res, keys)
    if mode.endswith("dev"):
        score = {k: sum(ok(res[n], k) for n in DEV_D) for k in keys[1:]}
        best = max(grid, key=lambda c: (score[kname(c)], -c[0], -(c[1] or 0)))
        md += ["", f"Casos D cumpridos (de 4): {score}", f"Escolha: {best} ({score[kname(best)]} de 4)" +
               ("" if score[kname(best)] >= 3 else " → REFUTADA, não vai ao teste")]
        json.dump({"config": best, "score": score}, open(chf, "w"))
    (OUT / f"repair_{mode}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
