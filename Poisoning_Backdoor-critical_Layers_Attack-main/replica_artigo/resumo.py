#!/usr/bin/env python
"""Lê os treinos da réplica (save/rep_*) e imprime uma tabela: ASR (sucesso do backdoor) e
acurácia, ambos média das últimas 10% das rodadas. Critério: ASR <= 10% com acurácia perto
do FedAvg = defesa segurou o ataque. Uso: python replica_artigo/resumo.py
"""
import glob
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)


def tail_mean(txt, key):
    g = re.search(rf"{key}=(\[.*?\])", txt)
    if not g:
        return None
    v = eval(g.group(1))
    if not v:
        return None
    k = max(1, len(v) // 10)
    return sum(v[-k:]) / k, len(v)


rows = []
for d in sorted(glob.glob("save/rep_*")):
    accf = glob.glob(os.path.join(d, "accuracy_file*"))
    tag = os.path.basename(d)
    if not accf:
        rows.append((tag, None, None, None))
        continue
    t = Path(accf[0]).read_text()
    ma = tail_mean(t, "main_task_accuracy")
    ba = tail_mean(t, "backdoor_accuracy")
    rows.append((tag, ba[0] if ba else None, ma[0] if ma else None, ba[1] if ba else 0))

if not rows:
    print("(nenhum treino em save/rep_* ainda)")
else:
    print(f"{'treino':28s} {'rodadas':>7s} {'ASR(backdoor)':>14s} {'acuracia':>9s}")
    print("-" * 62)
    for tag, asr, acc, n in rows:
        a = f"{asr:.2f}%" if asr is not None else "—"
        c = f"{acc:.2f}%" if acc is not None else "—"
        print(f"{tag:28s} {n:>7d} {a:>14s} {c:>9s}")
    print("\nLeitura: 'avg' = sem defesa (o ataque deve pegar, ASR alto). 'ours' = nossa proposta")
    print("(ASR baixo = segurou). Compare mesmo malicious/seed. Defesa OK: ASR<=10% e acuracia ~ a do avg.")
