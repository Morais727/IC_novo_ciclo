#!/usr/bin/env python
"""Fase 13 (13/11/2026) — generalização do cenário 8E (artigo 1, experimentos 1.7 e 1.8). Desenho FIXADO ANTES DE RODAR.

Base = cenário 8E fiel ao LaySelFL (fase8e.cfg: ResNet18-GN, α 0,9, 205 rodadas, ataque 80–200, BadNet de 4 tiras,
alvo 4, 30% de atacantes). Muda UMA coisa por variante:
  escala   : 100 clientes, 30 atacantes (os mesmos 30%), 20 clientes sorteados por rodada (participação parcial,
             src/federated/simulate.py, gerador próprio). Por rodada: mesmo nº de participantes do 8E, ~6 atacantes
             em média, mas cada cliente tem 1/5 dos dados.
  cifar100 : CIFAR-100 no lugar do CIFAR-10 (100 classes; holdout do servidor = 1 imagem por classe).
  escala50 : (15/11/2026, REDESENHO antes de qualquer defesa rodar) 100 clientes, 30 atacantes, 50 sorteados por rodada.
             Motivo: no portão da "escala" (K=20) cada rodada treina com 1/5 dos dados do 8E -> MA 41% na rodada 80
             (8E: 73%) e 50,6% no fim; ASR oscila 0–22% e só "pegou" por um ponto isolado (22,7%). Com K=50 a rodada
             vê metade dos dados do 8E e ~15 atacantes. Portão mais rígido, fixado antes de rodar: ASR MÉDIO nas
             rodadas 150–200 >= 20% E MA final >= 60%. A variante "escala" fica registrada como descartada.
  escala100: (15/11/2026 ~15h, pedido do usuário; desenho fixado ANTES de rodar) 100 clientes, 30 atacantes, TODOS
             participam em toda rodada (participação total). Motivo: o escala50 não passou no portão rígido (s42: ASR
             médio 63,3 mas MA 57,0; s43: 20,1 / 55,1). Com participação total cada rodada vê os mesmos dados do 8E
             (100 x 1/5). Portão = o mesmo do escala50 (ASR médio r150–200 >= 20% E MA final >= 60%). Prioridade
             mínima na fila (só ocupa vaga que nada mais quer) e só nas GPUs 3/4/5.
Nada é reajustado olhando a defesa: T = 8 e todos os demais parâmetros iguais aos do 8E.
Portão: lg_shadow (sem defesa + diagnóstico). O ataque "pega" se o PICO do ASR na janela 80–200 for >= 20%. As defesas
de uma variante×semente só rodam depois que o portão dela pegou (a fila consulta --gate-ok).
Saídas: results/fase13/raw/<variante>__<defesa>__s<semente>.json; modelo final em results/fase13/models/ (fora de
results/models_final, para não misturar com a análise da fase 10).
Uso: CUDA_VISIBLE_DEVICES=3 python -u experiments/fase13_generaliza.py --kind escala --seed 42 --defense lg_shadow
     python experiments/fase13_generaliza.py --gate-ok --kind cifar100 --seed 42   (código de saída 0 = pegou)
     python experiments/fase13_generaliza.py --report
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.fase8e as f8e          # import já fixa o gatilho badnet_cifar

OUT = Path("results/fase13")
KINDS = ("escala", "escala50", "escala100", "cifar100")
GATE_MIN = 20.0


def cfg(kind, defense, seed):
    f8e.SEED = seed
    c = f8e.cfg(defense)
    if kind == "escala":
        c.num_clients, c.num_malicious, c.client_samples_per_round = 100, 30, 20
    elif kind == "escala50":
        c.num_clients, c.num_malicious, c.client_samples_per_round = 100, 30, 50
    elif kind == "escala100":
        c.num_clients, c.num_malicious, c.client_samples_per_round = 100, 30, None
    elif kind == "cifar100":
        c.dataset = "cifar100"
    else:
        raise ValueError(kind)
    c.name = f"{kind}__{defense}__s{seed}"
    return c


def peak(res, lo=80, hi=200):
    v = [h["backdoor_asr"] for h in res["history"] if lo <= h["round"] <= hi and h["backdoor_asr"] is not None]
    return max(v) if v else None


def gate_ok(kind, seed):
    f = OUT / "raw" / f"{kind}__{f8e.SH}__s{seed}.json"
    if not f.exists():
        return False
    r = json.load(open(f))
    if kind in ("escala50", "escala100"):   # portão rígido (ver docstring)
        v = [h["backdoor_asr"] for h in r["history"] if 150 <= h["round"] <= 200 and h["backdoor_asr"] is not None]
        return bool(v) and sum(v) / len(v) >= GATE_MIN and r["history"][-1]["main_accuracy"] >= 60.0
    p = peak(r)
    return p is not None and p >= GATE_MIN


def report():
    rows = []
    for f in sorted(glob.glob(str(OUT / "raw" / "*.json"))):
        r = json.load(open(f)); h = r["history"]
        rows.append(f"{Path(f).stem:45s} pico={peak(r) or 0:5.1f}  final={h[-1]['backdoor_asr'] or 0:5.1f}  "
                    f"MA={h[-1]['main_accuracy']:5.1f}  {r['sec_per_round']:.0f}s/rod")
    print("\n".join(rows) or "nada ainda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--defense")
    ap.add_argument("--gate-ok", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quick", action="store_true", help="teste de fumaça (2 rodadas, poucos dados, nada é salvo)")
    a = ap.parse_args()
    if a.report:
        return report()
    if a.gate_ok:
        sys.exit(0 if gate_ok(a.kind, a.seed) else 1)

    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    c = cfg(a.kind, a.defense, a.seed)
    f = OUT / "raw" / f"{c.name}.json"
    if a.quick:
        c.quick, c.rounds, c.eval_every = True, 2, 1
        c.attack_params["start_round"] = 1
        r = run(c); print("QUICK OK", r["history"][-1], r["sec_per_round"]); return
    if f.exists():
        print(f"[skip] {c.name} já existe", flush=True); return
    if a.defense != f8e.SH and not gate_ok(a.kind, a.seed):
        print(f"[espera] portão de {a.kind} s{a.seed} ainda não pegou (ou não existe)", flush=True); sys.exit(3)
    lock = f.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {c.name}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    os.environ["FL_SAVE_FINAL"] = str(OUT / "models")
    os.environ["FL_SAVE_ROUNDS"] = "79,205"     # limpo (antes do ataque) + final: para especificidade/reativação depois
    while not mem_ok():
        time.sleep(30)
    print(f"\n=== {c.name} pid={os.getpid()} clientes={c.num_clients} atac={c.num_malicious} "
          f"K={c.client_samples_per_round} dataset={c.dataset} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
