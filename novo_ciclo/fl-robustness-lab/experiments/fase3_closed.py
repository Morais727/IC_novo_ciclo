#!/usr/bin/env python
"""Fase 3, teste em ciclo fechado da defesa trigger_scan (src/defenses/trigger_scan.py) — 17/10/2026.
Desenho e critérios fixados ANTES de rodar qualquer treino desta fila. Saídas em results/fase3c/.

Defesa: a cada 4 rounds a partir do round 8, varre quadrados de cor sólida (tamanhos 3/5, passo 3, deslocamento 1,
50 amostras do servidor) no modelo de cada cliente ainda não excluído; exclui para sempre o(s) cliente(s) suspeito(s)
e desaprende no modelo agregado os gatilhos achados. Duas variantes, sem nenhum ajuste depois de ver os treinos:
  V1 top1   — exclui o 1º colocado por max z (escore A7) se passar de thr_top; thr_top = calibrado no adendo A11
              (experiments/fase3_scan3.py, só com dumps de desenvolvimento s44/s45), lido de results/fase3/scan3.json.
  V2 shared — exclui os clientes com z > 3 no candidato cujo 2º maior z passa de 3 (gatilho compartilhado, A8).
Ataque: LP + ruído ν = 1 (lp_full_indiv_nu1.0), α = 0,1, 2 atacantes de 20, 40 rounds — o caso que nada parou.

DESENVOLVIMENTO (sementes 42, 44, 45): as duas variantes × {LP ν1, sem ataque}; mais FedAvg sem ataque em 44 e 45
  (referência que faltava). Uma variante PASSA no desenvolvimento se, nas 3 sementes:
    (a) ASR final <= 10% sob LP ν1   (FedAvg: 75,9 / 59,7 / 75,9%);
    (b) acurácia final sob LP ν1 >= acurácia do FedAvg sob o mesmo ataque − 3 pp (68,8 / 63,7 / 66,7%);
    (c) acurácia final sem ataque >= FedAvg sem ataque − 3 pp (s42: 69,0%; s44/s45: da referência desta fila).
  Se as duas passarem, escolhe a de menor ASR média (empate: maior acurácia média). Se nenhuma passar: a defesa é
  refutada e o teste NÃO roda (registra-se o resultado; nada de reajustar limiar e rodar de novo).
TESTE (sementes 46, 47, 48, nunca usadas com esta defesa): só a variante escolhida, LP ν1 e sem ataque. Mesmos
  critérios (a)–(c); referências FedAvg (results/fase2): sob LP ν1 65,3 / 67,1 / 66,7%; sem ataque 64,7 / 68,0 / 66,7%.
  A defesa é ACEITA se cumprir nas 3 sementes. Também se reportam: em que round cada atacante foi excluído e
  quantos honestos foram excluídos (falsos positivos).
ADENDO B1 (17/10, depois do 1º treino concluído, shared s42: excluiu os 2 atacantes no round 8 mas seguiu excluindo
  honestos a cada varredura até sobrarem 2 clientes; acc 35,5%, ASR 33%): uma variante que já falhou em alguma semente
  de desenvolvimento não roda as sementes restantes (o veredito NÃO passa já está dado). Só economia de máquina;
  nenhum critério, limiar ou parâmetro mudou.
Uso: python experiments/fase3_closed.py --one (uma vaga; results/fase3c/slots.sh roda 3) | --report
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.fase2_lab import C
from src.federated.simulate import run

OUT = Path("results/fase3c")
LP = "lp_full_indiv_nu1.0"
DEV, TEST = (42, 44, 45), (46, 47, 48)
REF = {  # (acc FedAvg sob LP ν1, acc FedAvg sem ataque); None = medido nesta fila
    42: (68.82, 68.98), 44: (63.74, None), 45: (66.67, None), 46: (65.34, 64.69), 47: (67.07, 67.99), 48: (66.69, 66.73)}


def thr_top():
    return json.load(open(OUT / "thr_top.json"))["thr_top"]


def ts(attack, seed, mode):
    p = dict(mode=mode, start=8, every=4, sizes=(3, 5), stride=3, offset=1, n_scan=50)
    if mode == "top1":
        p["thr_top"] = thr_top()
    cfg = C(attack, "trigger_scan", seed=seed, **p)
    cfg.name = f"{attack}__trigger_scan_{mode}__a0.1_s{seed}"
    return cfg


def failed(mode):
    """Adendo B1: já falhou em alguma semente de desenvolvimento concluída?"""
    ref_none = {s: (REF[s][1] if REF[s][1] is not None else (final(f"none__fedavg__a0.1_s{s}") or {}).get("acc")) for s in DEV}
    for s in DEV:
        a = final(f"{LP}__trigger_scan_{mode}__a0.1_s{s}"); n = final(f"none__trigger_scan_{mode}__a0.1_s{s}")
        if a and (a["asr"] > 10 or a["acc"] < REF[s][0] - 3):
            return True
        if n and ref_none[s] is not None and n["acc"] < ref_none[s] - 3:
            return True
    return False


def jobs():
    modes = [m for m in ("top1", "shared") if not failed(m)]
    j = [ts(LP, s, m) for s in DEV for m in modes]
    j += [C("none", "fedavg", seed=s) for s in (44, 45)]
    j += [ts("none", s, m) for s in DEV for m in modes]
    choice = OUT / "choice.json"
    if choice.exists():
        m = json.load(open(choice))["mode"]
        if m:
            j += [ts(a, s, m) for s in TEST for a in (LP, "none")]
    return j


def final(name):
    f = OUT / "raw" / f"{name}.json"
    if not f.exists():
        return None
    d = json.load(open(f)); h = d["history"][-1]
    asr = h["backdoor_asr"]; asr = None if asr is None else asr
    acc = h["main_accuracy"]; acc = 100 * acc if acc <= 1 else acc
    log = d.get("scan_log") or []
    mal = set(d["malicious_ids"])
    when = {}
    for e in log:
        for c in e.get("flagged", []):
            when.setdefault(c, e["round"])
    return dict(asr=asr, acc=acc, mal_flag={m: when.get(m) for m in mal}, fp=sorted(c for c in when if c not in mal))


def check(mode, seeds, ref_none):
    rows, ok = [], True
    for s in seeds:
        a = final(f"{LP}__trigger_scan_{mode}__a0.1_s{s}"); n = final(f"none__trigger_scan_{mode}__a0.1_s{s}")
        if a is None:
            ok = None if ok is not False else ok; continue
        ca, cb = a["asr"] <= 10, a["acc"] >= REF[s][0] - 3
        cc = None if n is None else n["acc"] >= ref_none[s] - 3
        if ok is not False:
            ok = False if not (ca and cb and cc is not False) else (ok if cc else None)
        mk = lambda v: "✓" if v else ("—" if v is None else "✗")
        rows.append(f"| {s} | {a['asr']:.1f} {mk(ca)} | {a['acc']:.1f} {mk(cb)} (ref {REF[s][0]:.1f}) "
                    f"| {'não rodou' if n is None else f'{n[chr(97)+chr(99)+chr(99)]:.1f}'} {mk(cc)} (ref {ref_none[s]:.1f}) "
                    f"| {a['mal_flag']} | {a['fp']} / {'—' if n is None else n['fp']} |")
    return ok, rows


def report():
    ref_none = {s: (REF[s][1] if REF[s][1] is not None else (final(f"none__fedavg__a0.1_s{s}") or {}).get("acc")) for s in REF}
    if any(ref_none[s] is None for s in DEV):
        print("referência FedAvg sem ataque incompleta"); return
    head = "| semente | ASR LP ν1 | acc LP ν1 | acc sem ataque | round de exclusão dos atacantes | honestos excluídos (LP / sem) |\n|---|---|---|---|---|---|"
    out, res = ["# Fase 3c — ciclo fechado (critérios: docstring de experiments/fase3_closed.py)\n", f"thr_top (A11) = {thr_top():.2f}\n"], {}
    for m in ("top1", "shared"):
        ok, rows = check(m, DEV, ref_none); res[m] = False if failed(m) else ok
        out += [f"## Desenvolvimento — {m}: {'PASSA' if ok else ('incompleto' if ok is None else 'NÃO passa')}", head] + rows + [""]
    choice = OUT / "choice.json"
    if all(v is not None for v in res.values()) and not choice.exists():
        cand = [m for m, v in res.items() if v]
        def key(m):
            r = [final(f"{LP}__trigger_scan_{m}__a0.1_s{s}") for s in DEV]
            return (sum(x["asr"] for x in r), -sum(x["acc"] for x in r))
        json.dump({"mode": min(cand, key=key) if cand else None}, open(choice, "w"))
    if choice.exists():
        m = json.load(open(choice))["mode"]
        out.append(f"Escolha (desenvolvimento): {m or 'nenhuma — defesa refutada, teste não roda'}\n")
        if m:
            ok, rows = check(m, TEST, ref_none)
            out += [f"## TESTE — {m}: {'ACEITA' if ok else ('incompleto' if ok is None else 'REJEITADA')}", head] + rows
    (OUT / "RESULTADO.md").write_text("\n".join(out) + "\n"); print("\n".join(out))


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
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    for cfg in jobs():
        if (OUT / "raw" / f"{cfg.name}.json").exists():
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        if not claim(cfg.name):
            continue
        print(f"\n=== {cfg.name} pid={os.getpid()}", flush=True)
        json.dump(run(cfg), open(OUT / "raw" / f"{cfg.name}.json", "w"))
        (OUT / "raw" / f"{cfg.name}.lock").unlink(missing_ok=True)
        report()   # grava choice.json quando o desenvolvimento termina, liberando o teste
        if args.one:
            return
    if args.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
