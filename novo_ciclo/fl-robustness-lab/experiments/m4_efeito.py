#!/usr/bin/env python
"""Entregável do mês 4, seção 4.1 (28/10/2026): efeito de cada ataque sem defesa, com mais sementes e uma medida
de efeito para TODO ataque. Desenho FIXADO ANTES DE RODAR.

Mesmo protocolo de experiments/attack_effect.py (ragnar_eval: CIFAR-10, SmallCNN, 20 clientes, Dirichlet 0,5,
40 rounds, FedAvg, mesmos parâmetros de ataque), agora nas sementes 42 (já rodada), 46, 47 e 48:
  - os 13 ataques do relatório 3 + inversão de sinal (sign_flip, que estava no catálogo sem validação);
  - referências sem atacantes: asr_baseline__trigger (= FedAvg limpo; na semente 42 a acurácia é idêntica à de
    clean__fedavg, 72,85%) e asr_baseline__edge (o Edge-case retira as amostras raras de todos).
Na semente 42 só rodam sign_flip e, de novo, label_flip e asr_baseline__trigger, para guardar o modelo final.
Medida de efeito por ataque (a do tipo de ataque):
  - backdoors: ASR (gatilho ou edge-case → classe-alvo), contra a ASR do modelo limpo;
  - troca de rótulos (0 → 2): taxa de troca = fração das imagens de teste da classe 0 classificadas como 2, contra
    a do modelo limpo (calculada do modelo final salvo); também a acurácia da classe 0;
  - não direcionados: queda de acurácia em relação ao modelo limpo da mesma semente.
Relatório: média das 4 sementes (mín–máx). 51 treinos (~6–13 min cada sozinhos).
ADENDO (28/10, 18h, pedido do usuário: resolver as pendências do mês 3; antes de qualquer resultado):
  - Ruído gaussiano calibrado (`norm_mult`): norma do ruído = m × mediana das normas honestas do round, m ∈ {1, 3,
    10, 30}, 4 atacantes, sementes 42, 46, 47, 48 (16 treinos). O desvio 5 atual derruba o modelo a 10% (inútil
    para medir detecção). Padrão injetável = o MENOR m com queda média de acurácia >= 2 p.p. (a ordem de LIE/IPM)
    e acurácia média >= 20% (sem colapso). Se nenhum m cumprir, registra-se a curva e o menor m que derruba >= 2 p.p.
  - Edge-case com as imagens EXTERNAS do artigo (aviões da Southwest, 784 treino / 196 teste, alvo caminhão);
    todos os outros parâmetros iguais ao edge_case_pgd atual (2 atacantes, PGD, escala 3, mistura 1:1). Referência:
    modelo limpo nas mesmas 196 imagens. 8 treinos. Funcional se ASR média >= 20%; nada é reajustado depois.
Uso: python experiments/m4_efeito.py --one | --report | --missing
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.fase4_alpha import mem_ok
from experiments.ragnar_eval import ATTACKS, COMMON, OPTIONAL_ATTACKS
from src.attacks.registry import ATTACK_REGISTRY
from src.config import ScenarioConfig

OUT, OLD = Path("results/m4_efeito"), Path("results/ragnar_eval")
SEEDS = (42, 46, 47, 48)
ORDER = ["gaussian_noise", "sign_flip", "ipm", "lie", "minmax", "minsum", "mimic", "lf", "label_flip",
         "backdoor_pixel", "backdoor_pgd", "dba", "lp", "edge_case_pgd"]
SPEC = {**ATTACKS, **OPTIONAL_ATTACKS}
BASE = {"trigger": "backdoor_pixel", "edge": "edge_case_pgd"}
LABEL = {"gaussian_noise": "Ruído gaussiano", "sign_flip": "Inversão de sinal", "ipm": "IPM", "lie": "LIE",
         "minmax": "MinMax", "minsum": "MinSum", "mimic": "Mimic *", "lf": "LF *", "label_flip": "Troca de rótulos",
         "backdoor_pixel": "Backdoor (gatilho)", "backdoor_pgd": "Backdoor (PGD)", "dba": "DBA *", "lp": "LP *",
         "edge_case_pgd": "Edge-case PGD *"}


EDGE_SW = {**ATTACKS["edge_case_pgd"]["attack_params"], "edge_source": "southwest"}
GAUSS_M = (1, 3, 10, 30)
EXTRA = {f"gaussian_noise_m{m}": ("gaussian_noise", 4, {"norm_mult": float(m)}) for m in GAUSS_M}
EXTRA["edge_case_sw"] = ("edge_case_pgd", 2, EDGE_SW)
EXTRA["asr_baseline__edgesw"] = ("edge_case_pgd", 0, EDGE_SW)
LABEL.update({f"gaussian_noise_m{m}": f"Ruído gaussiano (m = {m})" for m in GAUSS_M})
LABEL["edge_case_sw"] = "Edge-case PGD, imagens externas *"


def cfg(name, seed):
    if name in EXTRA:
        a, n, ap = EXTRA[name]
        tag = name if name.startswith("asr_baseline__") else f"{name}__fedavg"
        return ScenarioConfig(name=f"{tag}_s{seed}", attack=a, num_malicious=n, attack_params=dict(ap),
                              defense="fedavg", seed=seed, **COMMON)
    if name.startswith("asr_baseline__"):
        a = BASE[name.split("__")[1]]
        return ScenarioConfig(name=f"{name}_s{seed}", attack=a, num_malicious=0,
                              attack_params=dict(SPEC[a]["attack_params"]), defense="fedavg", seed=seed, **COMMON)
    s = SPEC[name]
    return ScenarioConfig(name=f"{name}__fedavg_s{seed}", attack=name, num_malicious=s["num_malicious"],
                          attack_params=dict(s["attack_params"]), defense="fedavg", seed=seed, **COMMON)


def jobs():
    # pendências do mês 3 primeiro (pedido do usuário, 18h35): Edge-case externo, ruído calibrado, troca de rótulos
    j = [cfg(n, s) for s in SEEDS for n in ("edge_case_sw", "asr_baseline__edgesw") + tuple(f"gaussian_noise_m{m}" for m in GAUSS_M)]
    j += [cfg(n, s) for s in SEEDS for n in ("label_flip", "asr_baseline__trigger")]
    j += [cfg("sign_flip", 42)]
    for s in SEEDS[1:]:
        j += [cfg(n, s) for n in ("asr_baseline__edge",) + tuple(ORDER)]
    seen = set()
    return [c for c in j if not (OUT / "raw" / f"{c.name}.json").exists() and not (c.name in seen or seen.add(c.name))]


def last(name):
    for d in (OUT / "raw", OLD):
        f = d / f"{name}.json"
        if f.exists():
            return json.load(open(f))["history"][-1]
    return None


_TEST = None


def flip_rates(name):
    """(taxa de troca 0→2, acurácia da classe 0) do modelo final salvo."""
    global _TEST
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.data.datasets import load_dataset
    from src.federated.utils import load_vector
    from src.models.cnn import build_model
    f = OUT / "final" / f"{name}.pt"
    if not f.exists():
        return None
    if _TEST is None:
        test = load_dataset("cifar10")[1]
        _TEST = Subset(test, [i for i, y in enumerate(test.targets) if y == 0])
    m = build_model("smallcnn", dataset="cifar10")
    load_vector(m, torch.load(f))
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(x).argmax(1) for x, _ in DataLoader(_TEST, batch_size=500)])
    return 100 * (pred == 2).float().mean().item(), 100 * (pred == 0).float().mean().item()


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return "–", None
    m = sum(v) / len(v)
    return (f"{m:.1f} ({min(v):.1f}–{max(v):.1f})" if len(v) > 1 else f"{m:.1f}"), len(v)


def report():
    import csv
    rows, md = [], ["# Mês 4, seção 4.1 — efeito de cada ataque sem defesa (média das sementes; mín–máx)", "",
                    "| Ataque | Maliciosos | Acurácia (%) | Δ acurácia (p.p.) | Medida de efeito | Ataque (%) | Modelo limpo (%) | sementes |",
                    "|---|---|---|---|---|---|---|---|"]
    clean = {s: last(f"asr_baseline__trigger_s{s}") for s in SEEDS}
    edge0 = {s: last(f"asr_baseline__edge_s{s}") for s in SEEDS}
    edgesw0 = {s: last(f"asr_baseline__edgesw_s{s}") for s in SEEDS}
    order = ORDER[:1] + [f"gaussian_noise_m{m}" for m in GAUSS_M] + ORDER[1:] + ["edge_case_sw"]
    for a in order:
        h = {s: last(f"{a}__fedavg_s{s}") for s in SEEDS}
        ok = [s for s in SEEDS if h[s] and clean[s]]
        acc, n = stat([h[s]["main_accuracy"] for s in ok])
        dacc, _ = stat([h[s]["main_accuracy"] - clean[s]["main_accuracy"] for s in ok])
        ev = ATTACK_REGISTRY[EXTRA.get(a, (a,))[0]].get("eval")
        if ev:
            ref = clean if ev == "trigger" else (edgesw0 if a == "edge_case_sw" else edge0)
            what = ("ASR (gatilho → classe 0)" if ev == "trigger" else "ASR (aviões Southwest → caminhão)"
                    if a == "edge_case_sw" else "ASR (aviões raros → caminhão)")
            eff, _ = stat([h[s]["backdoor_asr"] for s in ok])
            base, _ = stat([ref[s]["backdoor_asr"] for s in ok if ref[s]])
        elif a == "label_flip":
            what = "taxa de troca 0 → 2"
            fr = {s: flip_rates(f"label_flip__fedavg_s{s}") for s in SEEDS}
            f0 = {s: flip_rates(f"asr_baseline__trigger_s{s}") for s in SEEDS}
            eff, _ = stat([fr[s][0] for s in SEEDS if fr[s]])
            base, _ = stat([f0[s][0] for s in SEEDS if f0[s]])
        else:
            what, eff, base = "queda de acurácia", dacc, "0,0"
        r = dict(ataque=LABEL[a], maliciosos=EXTRA[a][1] if a in EXTRA else SPEC[a]["num_malicious"], acc=acc, dacc=dacc, medida=what, efeito=eff,
                 base=base, sementes=n or 0)
        rows.append(r)
        md.append("| " + " | ".join(str(v) for v in r.values()) + " |")
    if any(clean.values()):
        c, _ = stat([v["main_accuracy"] for v in clean.values() if v])
        md += ["", f"Modelo limpo (sem atacantes): acurácia {c}%."]
    lf = [flip_rates(f"label_flip__fedavg_s{s}") for s in SEEDS]
    l0 = [flip_rates(f"asr_baseline__trigger_s{s}") for s in SEEDS]
    if any(lf):
        md.append(f"Troca de rótulos, acurácia da classe 0: {stat([x[1] for x in lf if x])[0]}% contra "
                  f"{stat([x[1] for x in l0 if x])[0]}% no modelo limpo.")
    (OUT / "EFEITO.md").write_text("\n".join(md) + "\n")
    with open(OUT / "efeito.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    for f in ("--one", "--report", "--missing"):
        ap.add_argument(f, action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    os.environ["FL_SAVE_FINAL"] = str(OUT / "final")
    if args.report:
        return report()
    if args.missing:
        j = jobs(); print(len(j)); [print(c.name) for c in j]; return
    from src.federated.simulate import run
    for c in jobs():
        f = OUT / "raw" / f"{c.name}.json"
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        lock = OUT / "raw" / f"{c.name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0); continue
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True); continue
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        if f.exists():
            lock.unlink(missing_ok=True); continue
        print(f"\n=== {c.name} pid={os.getpid()}", flush=True)
        json.dump(run(c), open(f, "w"))
        lock.unlink(missing_ok=True)
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
