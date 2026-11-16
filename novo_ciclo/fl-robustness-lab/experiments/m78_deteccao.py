#!/usr/bin/env python
"""Entregáveis M7 e M8 (15/11/2026): DETECÇÃO do agente sob envenenamento de dados (M7) e de modelo (M8).
Desenho FIXADO ANTES DE RODAR (pedido do usuário: o plano chama M7/M8 de "Relatório de detecção"; os rascunhos só
mediam o efeito dos ataques sem defesa).

Cenário = o do Quadro 4.1 / experiments/m4_efeito.py (CIFAR-10, SmallCNN, 20 clientes, Dirichlet 0,5, 40 rodadas,
mesmos parâmetros de ataque e nº de maliciosos), sementes 42, 46, 47, 48 (as do m4_efeito: o FedAvg já existe lá).
Ataques: dados (M7) = label_flip, backdoor_pixel, backdoor_pgd, dba, edge_case_pgd;
         modelo (M8) = ipm, sign_flip, lf, minsum, minmax, lie, lp;
         sem ataque (falsos positivos) = asr_baseline__trigger (0 maliciosos).
Agente: layer_guard_sn (perfil por tensor, 1º estágio), layer_guard_corr, layer_guard_corr_soft, layer_guard_client.
Limiar de corroboração na SmallCNN (10 tensores): mesma FRAÇÃO da ResNet18-GN (T = 8 de 62 ≈ 13%), arredondada para
cima e com mínimo 2 -> T = 2. Fixado aqui, antes de qualquer corrida; nada é reajustado olhando o resultado.
Medidas: detecção por rodada (precisão, recall, F1 — `history[*].detection`), efeito do ataque (ASR / queda de
acurácia, como no m4_efeito) e, sem ataque, falsos positivos e acurácia.
Saída: results/m78/raw/<ataque>__<defesa>_s<semente>.json.  Uso: --attack A --defense D --seed S | --report
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import m4_efeito as m4

OUT = Path("results/m78")
SEEDS = (42, 46, 47, 48)
DATA = ["label_flip", "backdoor_pixel", "backdoor_pgd", "dba", "edge_case_pgd"]
MODEL = ["ipm", "sign_flip", "lf", "minsum", "minmax", "lie", "lp"]
CLEAN = "asr_baseline__trigger"
DEFENSES = ["layer_guard_sn", "layer_guard_corr", "layer_guard_corr_soft", "layer_guard_client"]
T_SMALLCNN = 2


def cfg(attack, defense, seed):
    c = m4.cfg(attack, seed)
    c.defense = defense
    c.defense_params = {"confirm_tensors": T_SMALLCNN}
    c.name = f"{attack}__{defense}_s{seed}"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack"); ap.add_argument("--defense"); ap.add_argument("--seed", type=int)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    c = cfg(a.attack, a.defense, a.seed)
    f = OUT / "raw" / f"{c.name}.json"
    if f.exists():
        print(f"[skip] {c.name}", flush=True); return
    lock = f.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {c.name}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    print(f"\n=== {c.name} pid={os.getpid()} T={T_SMALLCNN} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


def report():
    import glob, statistics as st
    rows = {}
    for p in glob.glob(str(OUT / "raw" / "*.json")):
        r = json.load(open(p)); h = r["history"]; n = Path(p).stem
        atk, rest = n.split("__", 1) if not n.startswith("asr_baseline") else (CLEAN, n[len(CLEAN) + 2:])
        d, s = rest.rsplit("_s", 1)
        det = [x["detection"] for x in h if x.get("detection")]
        P = st.mean(x["precision"] for x in det) if det else None
        R = st.mean(x["recall"] for x in det) if det else None
        fp = st.mean(x.get("fp", 0) for x in det) if det else None
        rows.setdefault((atk, d), []).append((h[-1]["main_accuracy"], h[-1]["backdoor_asr"], P, R, fp))
    out = ["# M7/M8 — detecção do agente (média das sementes)", "",
           "| ataque | defesa | n | MA | ASR | precisão | recall | FP/aval. |", "|---|---|---|---|---|---|---|---|"]
    for (atk, d), v in sorted(rows.items()):
        m = lambda i: (f"{st.mean(x[i] for x in v if x[i] is not None):.2f}" if any(x[i] is not None for x in v) else "—")
        out.append(f"| {atk} | {d} | {len(v)} | {m(0)} | {m(1)} | {m(2)} | {m(3)} | {m(4)} |")
    (OUT / "RESULTADO.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
