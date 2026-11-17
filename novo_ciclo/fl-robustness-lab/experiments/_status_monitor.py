#!/usr/bin/env python
"""Monitor de treinos (15/11/2026): gera results/fase10/STATUS.json a partir da fila em uso e dos logs dos jobs.

- rodando: tag, GPU, rodada atual/total, ASR e MA da última avaliação, s/rodada (últimas duas avaliações), previsão de término;
- contagem feitos / rodando / na fila (por grupo) lida das definições da fila mais recente (+ PRIO_JOBS.txt);
- previsão global: soma do trabalho restante / nº de vagas em uso (estimativa grosseira, marcada como tal).
Uso: python experiments/_status_monitor.py [--loop 120]   (sem dependências além da stdlib)
"""
import glob, json, os, re, subprocess, sys, time
from datetime import datetime, timedelta

ROOT = "/home/marcos.morais/IC/novo_ciclo/fl-robustness-lab"
OUT = f"{ROOT}/results/fase10/STATUS.json"
QLOG = f"{ROOT}/results/fase10/_fila_gpu.log"
DEADLINE = datetime(2026, 11, 21, 23, 0)
RX = re.compile(r"round (\d+)/(\d+) MA=([\d.]+)% BA=([\d.]+)%.*\((\d+)s\)\s*$")
# duração típica (h) de um job inteiro por prefixo, usada só para quem ainda não começou
TYP = {"a20": 30, "p20": 8, "f19": 0.3, "g13_escala100": 15, "l18": 7, "a17": 30, "m78": 0.3, "adp": 12, "a16": 14, "f8e": 7, "impf": 7, "g13": 8, "tsw": 7, "fp": 4, "rea": 1.5, "analise": 0.6, "c2": 1}


def group(tag):
    for p, name in (("a20", "Resposta ao adaptativo 2 (fase 20)"), ("p20", "Resposta ao adaptativo 2 (fase 20)"), ("f19", "Ataques das defesas citadas"), ("m78", "M7/M8 detecção"), ("l18", "Linha de base limpa"), ("a17", "Atacante intermediário"), ("a16", "Resposta ao adaptativo"), ("p16", "Resposta ao adaptativo"), ("adp", "Adversário adaptativo"),
                    ("f8e", "Tabela principal"), ("impf", "Modelos finais (especificidade)"), ("rea", "Reativação"),
                    ("g13_escala", "Escala"), ("g13_cifar", "CIFAR-100"), ("tsw", "Varredura de T"),
                    ("fp", "Falsos positivos"), ("analise", "Análises"), ("c2", "Cenário 2")):
        if tag.startswith(p):
            return name
    return "Outros"


def queue_defs():
    qs = sorted(glob.glob(f"{ROOT}/results/fase10/_fila_gpu_v*.sh"), key=lambda p: int(re.findall(r"_v(\d+)\.sh", p)[0]))
    q = qs[-1]
    src = open(q).read()
    m = re.search(r'^log "==== fila', src, re.M)
    body = "\n".join(l for l in src[:m.start()].splitlines() if not re.match(r"\s*(while|setsid)", l))
    body += r'''
for cls in ADP HEAVY LAST; do declare -p $cls >/dev/null 2>&1 || continue; declare -n L=$cls; for j in "${L[@]}"; do IFS='|' read -r t d l c u <<< "$j"; st=pend; [ -f "$d" ] && st=done; echo "$t|$st|$u"; done; done
for F in results/fase10/PRIO_JOBS.txt results/fase10/LIGHT_JOBS.txt; do [ -f $F ] && grep -v '^\s*#' $F | grep '|' | while IFS='|' read -r t d l c u; do st=pend; [ -f "$d" ] && st=done; echo "$t|$st|$u"; done; done
'''
    r = subprocess.run(["bash", "-c", body], cwd=ROOT, capture_output=True, text=True, timeout=120)
    jobs = {}
    for line in r.stdout.splitlines():
        t, st, u = line.split("|", 2)
        jobs[t] = (st, u)
    return os.path.basename(q), jobs


def launched():
    """tag -> (pid, gpu, log) da última vez que foi lançado (log da fila)."""
    d = {}
    for line in open(QLOG, errors="replace"):
        m = re.search(r"-> GPU (\d+) \(pid (\d+)\) :: (\S+).*log=(\S+)", line)
        if m:   # guarda o lançamento VIVO mais recente (relançamentos que saem na hora não escondem o que roda)
            v = (int(m.group(2)), int(m.group(1)), m.group(4))
            if alive(v[0]) or m.group(3) not in d or not alive(d[m.group(3)][0]):
                d[m.group(3)] = v
    return d


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def progress(log):
    pts = []
    try:
        for line in open(f"{ROOT}/{log}", errors="replace"):
            m = RX.search(line)
            if m:
                pts.append(tuple(float(x) for x in m.groups()))
    except FileNotFoundError:
        return None
    if not pts:
        return None
    r, tot, ma, ba, t = pts[-1]
    spr = None
    if len(pts) >= 2 and pts[-1][0] > pts[-2][0] and pts[-1][4] > pts[-2][4]:
        spr = (pts[-1][4] - pts[-2][4]) / (pts[-1][0] - pts[-2][0])
    elif r > 0:
        spr = t / r
    mtime = os.path.getmtime(f"{ROOT}/{log}")
    return dict(round=int(r), total=int(tot), ma=ma, asr=ba, spr=spr, mtime=mtime)


def _gate_failed(kind, seed):
    """Veredito em cache da fila (results/fase10/.gate_cache): 1 = não pegou."""
    for f in glob.glob(f"{ROOT}/results/fase10/.gate_cache/gate*_{kind}__lg_shadow__s{seed}.json_*"):
        return open(f).read().strip() == "1"
    return None


def waiting(tag):
    """'' = pronto para rodar; 'descartado' = depende de um portão que não pegou; outro texto = aguardando condição."""
    m = re.match(r"g13_(\w+?)__(\w+)__s(\d+)$", tag)
    if m and m.group(2) != "lg_shadow":
        kind, seed = m.group(1), m.group(3)
        if not os.path.exists(f"{ROOT}/results/fase13/raw/{kind}__lg_shadow__s{seed}.json"):
            return "portão"
        return "descartado" if _gate_failed(kind, seed) else ""
    m = re.match(r"rea_(.+)_(reativacao|dormencia)$", tag)
    if m and not os.path.exists(f"{ROOT}/results/fase10/models/{m.group(1)}.pt"):
        return "modelo"
    if tag.startswith("analise_"):
        return "dependências"
    return ""


def snapshot():
    now = datetime.now()
    qname, jobs = queue_defs()
    L = launched()
    running = []
    for tag, (pid, gpu, log) in L.items():
        if not alive(pid):
            continue
        p = progress(log) or {}
        eta = left_min = None
        if p.get("spr") and p.get("total"):
            left = (p["total"] - p["round"]) * p["spr"] - (time.time() - p["mtime"])
            eta = (now + timedelta(seconds=max(left, 0))).strftime("%d/%m %H:%M")
            left_min = round(max(left, 0) / 60)
        running.append(dict(tag=tag, group=group(tag), gpu=gpu, round=p.get("round"), total=p.get("total"),
                            asr=p.get("asr"), ma=p.get("ma"), spr=round(p["spr"]) if p.get("spr") else None, eta=eta, left_min=left_min))
    run_tags = {r["tag"] for r in running}
    groups = {}
    pend_hours = 0.0
    for t, (st, u) in jobs.items():
        g = groups.setdefault(group(t), {"feitos": 0, "rodando": 0, "fila": 0, "aguardando": 0, "descartados": 0, "expirados": 0})
        if st == "done":
            g["feitos"] += 1
        elif t in run_tags:
            g["rodando"] += 1
        else:
            try:
                exp = datetime.strptime(u, "%Y-%m-%d %H:%M") < now
            except ValueError:
                exp = False
            w = waiting(t)
            if exp:
                g["expirados"] += 1
            elif w == "descartado":
                g["descartados"] += 1
            elif w:
                g["aguardando"] += 1
            else:
                g["fila"] += 1
                pend_hours += next((v for k, v in TYP.items() if t.startswith(k)), 7)
    run_hours = sum(((r["total"] - r["round"]) * r["spr"] / 3600) for r in running if r.get("spr") and r.get("total"))
    slots = max(len(running), 1)
    eta_all = now + timedelta(hours=(pend_hours + run_hours) / slots)
    tot = {k: sum(g[k] for g in groups.values()) for k in ("feitos", "rodando", "fila", "aguardando", "descartados", "expirados")}
    gpus = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        for line in q.strip().splitlines():
            i, u, mu, mt = [x.strip() for x in line.split(",")]
            gpus.append(dict(gpu=int(i), util=int(u), mem=round(int(mu) / 1024, 1), memtot=round(int(mt) / 1024, 1),
                             jobs=sum(1 for r in running if r["gpu"] == int(i))))
    except Exception:
        pass
    return dict(updated=now.strftime("%d/%m/%Y %H:%M:%S"), queue=qname, deadline=DEADLINE.strftime("%d/%m %H:%M"),
                totals=tot, groups=groups, running=sorted(running, key=lambda r: (r["gpu"], r["tag"])),
                eta_all=eta_all.strftime("%d/%m %H:%M"), eta_note="estimativa grosseira: trabalho restante / vagas em uso",
                gpus=gpus)


def main():
    every = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--loop" else 0
    while True:
        try:
            s = snapshot()
            tmp = OUT + ".tmp"
            json.dump(s, open(tmp, "w"), ensure_ascii=False, indent=1)
            os.replace(tmp, OUT)
        except Exception as e:
            print("erro:", e, file=sys.stderr)
        if not every:
            break
        time.sleep(every)


if __name__ == "__main__":
    main()
