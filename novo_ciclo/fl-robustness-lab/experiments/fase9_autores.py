#!/usr/bin/env python
"""Fase 9A (28/10/2026) — a nossa proposta dentro do CÓDIGO ORIGINAL do LP (Zhuang et al., ICLR 2024).
Pergunta de revisor: "a sua versão do LP é mais fraca que a original?". Desenho e critério FIXADOS ANTES DE RODAR.

Repositório dos autores (~/Documentos/IC/Poisoning_Backdoor-critical_Layers_Attack-main), SEM alterar o código
deles: o wrapper semeia random/numpy/torch (como experiments/repro_lp_seeded.py) e, para a nossa proposta, troca
utils.defense.flame por um adaptador que chama src/defenses/layer_guard.py::layer_guard_sn (limiar 3, sem mudar
nada) e roda com --defence flame. O adaptador agrega por tensor do state_dict (a mesma granularidade do LP) com
pesos iguais, como o FedAvg dos autores. Linha de base: --defence avg (FedAvg dos autores).
Cenário: o da reprodução da Tab. 2 (CNN rlr_mnist, Fashion-MNIST não-IID q = 0,5, 100 clientes, 10 por rodada,
200 rodadas, LP = --attack lp_attack, hiperparâmetros do Apêndice A-7), sementes 1, 2, 3, com
  - 10% de maliciosos (padrão do artigo: 1 atacante por rodada — sem conluio, o sinal de semelhança não ajuda);
  - 20% de maliciosos (2 por rodada — o regime em que a proposta foi desenhada).
Treinos: 2 defesas × 2 frações × 3 sementes = 12 (~70 min cada sozinho).
Critério por caso (o da Fase 4, adaptado): sucesso final (média dos últimos 10% das rodadas) <= 10% com acurácia
>= FedAvg − 3 pp; FedAvg com sucesso < 20% = LP não pegou (inconclusivo).
Expectativa declarada antes: com 10% (1 atacante) a proposta pode falhar; o resultado vai para o artigo como for.
Uso (venv do repositório): <repo>/.venv-lp/bin/python experiments/fase9_autores.py --one | --report
"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent
# o repo dos autores já esteve em ~/Documentos/IC/...; nesta máquina (dl-16) está em ~/IC/... — auto-detecta.
_CANDS = [Path.home() / "Documentos/IC/Poisoning_Backdoor-critical_Layers_Attack-main",
          Path.home() / "IC/Poisoning_Backdoor-critical_Layers_Attack-main"]
REPO = next((p for p in _CANDS if p.exists()), _CANDS[0])
OUT = LAB / "results/fase9"
BASE = ("--dataset fashion_mnist --model rlr_mnist --lr 0.01 --poison_frac 0.5 --local_ep 2 --local_bs 64 "
        "--attack_begin 0 --epochs 200 --attack_label 5 --attack_goal -1 --trigger square --triggerX 23 "
        "--triggerY 23 --gpu 0 --iid 0 --attack lp_attack")
JOBS = [(d, m, s) for m in (0.2, 0.1) for s in (1, 2, 3) for d in ("avg", "ours")]
WRAPPER = """
import faulthandler, signal, random, runpy, sys, json as _json
import numpy as np, torch
faulthandler.register(signal.SIGUSR1, file=open('{save}/pilha.txt', 'w'), all_threads=True)
random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed}); torch.cuda.manual_seed_all({seed})
if {ours}:
    sys.path.append({lab!r})
    from src.defenses.layer_guard import layer_guard_sn
    import utils.defense as D
    _EVID = {evid!r}; _RC = [0]          # evidência (item 470): loga a CONTAGEM DE CORROBORAÇÃO por round
    def _logcorr(deltas, groups, keys, args):
        c = deltas - deltas.median(0).values
        def rz(v):
            mu = np.median(v); return (v - mu) / (1.4826 * np.median(np.abs(v - mu)) + 1e-12)
        cnt = np.zeros(len(deltas))
        for k in keys:                    # mesmo sinal do layer_guard_sn: z_sim + z_norm por tensor
            x = c[:, groups[k]]
            u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cm = u @ u.T; cm.fill_diagonal_(-2.0)
            z = np.maximum(rz(cm.max(1).values.cpu().numpy()), 0.0) + np.maximum(rz(x.norm(dim=1).cpu().numpy()), 0.0)
            cnt += (z > 3.0)
        nmal = int(args.malicious * len(deltas))      # atacantes = primeiras nmal linhas do w_updates
        ben = cnt[nmal:]
        rec = dict(round=_RC[0], n=int(len(deltas)), nmal=nmal, count_mal=cnt[:nmal].tolist(),
                   count_ben_med=float(np.median(ben)) if len(ben) else None,
                   count_ben_p90=float(np.percentile(ben, 90)) if len(ben) else None,
                   count_ben_max=float(ben.max()) if len(ben) else None)
        open(_EVID, 'a').write(_json.dumps(rec) + '\\n'); _RC[0] += 1
    def adapter(w_locals, w_updates, w_glob, args, debug=False):
        keys = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                and k.split('.')[-1] not in ('running_mean', 'running_var')]
        groups, pos = {{}}, 0
        for k in keys:
            groups[k] = slice(pos, pos + w_glob[k].numel()); pos += w_glob[k].numel()
        dev = w_glob[keys[0]].device
        deltas = torch.stack([torch.cat([u[k].reshape(-1).float().to(dev) for k in keys]) for u in w_updates])
        if _EVID: _logcorr(deltas, groups, keys, args)
        agg, info = layer_guard_sn(deltas, torch.ones(len(w_updates)), {{'state': {{'tensor_groups': groups}}}}, {{}})
        for k in keys:
            w_glob[k] = w_glob[k] + agg[groups[k]].reshape(w_glob[k].shape).to(w_glob[k].dtype)
        return w_glob
    D.flame = adapter
sys.argv = ['main_fed.py'] + {argv!r}
runpy.run_path('main_fed.py', run_name='__main__')
"""
STALL_S = 45 * 60


def name(d, m, s):
    return f"f9_{d}_m{m}_s{s}"


def read_acc(save):
    files = list(save.glob("accuracy_file*.txt"))
    if not files:
        return [], []
    t = files[0].read_text()
    get = lambda k: eval(g.group(1)) if (g := re.search(rf"{k}=(\[.*?\])", t)) else []
    return get("main_task_accuracy"), get("backdoor_accuracy")


PYLP = next((p for p in [REPO / ".venv-lp/bin/python", REPO / ".venv/bin/python", LAB / ".venv/bin/python"]
             if p.exists()), REPO / ".venv-lp/bin/python")


def run_job(d, m, s, epochs=None, evid="", tag=""):
    # tag != "" isola corridas auxiliares (ex.: --evidence) dos resultados reais de 200 rodadas
    rel = f"save/{name(d, m, s)}{tag}"
    save = REPO / rel
    save.mkdir(parents=True, exist_ok=True)
    base = BASE if epochs is None else re.sub(r"--epochs \d+", f"--epochs {epochs}", BASE)
    argv = (base + f" --malicious {m} --defence {'avg' if d == 'avg' else 'flame'} --seed {s} --save {rel}").split()
    code = WRAPPER.format(save=rel, seed=s, ours=d == "ours", lab=str(LAB), argv=argv, evid=evid)
    log = open(REPO / f"{rel}.log", "w")
    proc = subprocess.Popen([str(PYLP), "-u", "-c", code], cwd=REPO, stdout=log,
                            stderr=subprocess.STDOUT)
    last_n, last_t, status = -1, time.time(), "ok"
    while proc.poll() is None:
        time.sleep(30)
        n = len(read_acc(save)[0])
        if n != last_n:
            last_n, last_t = n, time.time()
        elif time.time() - last_t > STALL_S:
            proc.send_signal(signal.SIGUSR1); time.sleep(10); proc.kill()
            status = f"travou no round {n}"
            break
    if proc.returncode not in (0, None) and status == "ok":
        status = f"erro (código {proc.returncode}; ver {rel}.log)"
    ma, ba = read_acc(save)
    k = max(1, len(ba) // 10)
    r = dict(defesa=d, maliciosos=m, semente=s, status=status, rounds=len(ba),
             acc=float(np.mean(ma[-k:])) if ma else None, asr=float(np.mean(ba[-k:])) if ba else None)
    json.dump(r, open(OUT / "raw" / f"{name(d, m, s)}{tag}.json", "w"))
    return r


def report():
    R = {p.stem: json.load(open(p)) for p in (OUT / "raw").glob("f9_*.json")}
    out = ["# Fase 9A — nossa proposta no código original do LP (FMNIST, CNN, 200 rodadas)", "",
           "| maliciosos | semente | FedAvg: sucesso (acurácia) | nossa: sucesso (acurácia) | veredito |", "|---|---|---|---|---|"]
    for m in (0.1, 0.2):
        for s in (1, 2, 3):
            a, o = R.get(name("avg", m, s)), R.get(name("ours", m, s))
            f = lambda r: "—" if not r or r["asr"] is None else f"{r['asr']:.1f}% ({r['acc']:.1f}%)" + ("" if r["status"] == "ok" else f" [{r['status']}]")
            v = "—"
            if a and o and a["asr"] is not None and o["asr"] is not None:
                v = "inconclusivo" if a["asr"] < 20 else ("defende" if o["asr"] <= 10 and o["acc"] >= a["acc"] - 3 else "não defende")
            out.append(f"| {int(m * 100)}% | {s} | {f(a)} | {f(o)} | {v} |")
    txt = "\n".join(out) + "\n"
    (OUT / "AUTORES.md").write_text(txt); print(txt)


def evidence(epochs=40, seed=1, ms=(0.1, 0.2)):
    """Evidência do mecanismo de conluio (item 470): re-roda a nossa proposta no repo do LP com 10% (1 atacante)
    e 20% (2 atacantes), logando por round a CONTAGEM DE CORROBORAÇÃO (nº de tensores que acusam cada cliente)
    do(s) atacante(s) vs. os benignos. Hipótese: com 1 atacante a contagem dele ~ a dos benignos (indistinguível,
    pois o z_sim precisa de >= 2 maliciosos parecidos), logo a corr não purga e o backdoor passa; com 2 eles se
    corroboram e a contagem sobe acima do limiar. ~{e} rounds bastam (o LP ataca desde o round 0)."""
    ev = OUT / "evidencia"
    ev.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in ms:  # roda só as frações pedidas (--ms); as demais reaproveitam o jsonl já gravado
        jl = ev / f"corr_m{m}_s{seed}.jsonl"
        jl.unlink(missing_ok=True)
        print(f"=== evidência ours m={m} s={seed} ({epochs} rounds) -> {jl}", flush=True)
        run_job("ours", m, seed, epochs=epochs, evid=str(jl), tag="_evid")
    for m in (0.1, 0.2):
        jl = ev / f"corr_m{m}_s{seed}.jsonl"
        recs = [json.loads(l) for l in jl.read_text().splitlines()] if jl.exists() else []
        if recs:
            import numpy as _np
            cmal = _np.array([_np.mean(r["count_mal"]) for r in recs if r["count_mal"]])
            cben = _np.array([r["count_ben_med"] for r in recs if r["count_ben_med"] is not None])
            rows.append(dict(malicious=m, n_atacantes=recs[0]["nmal"], rounds=len(recs),
                             count_mal_medio=float(cmal.mean()) if len(cmal) else None,
                             count_ben_medio=float(cben.mean()) if len(cben) else None))
    (OUT / "EVIDENCIA.json").write_text(json.dumps(rows, indent=2))
    print("\n# Evidência — contagem de corroboração (tensores que acusam) por round")
    print("| maliciosos | nº atacantes | round | atacante (média) | benigno (mediana) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {int(r['malicious']*100)}% | {r['n_atacantes']} | {r['rounds']} | "
              f"{r['count_mal_medio']:.1f} | {r['count_ben_medio']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--evidence", action="store_true"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ms", type=float, nargs="+", default=[0.1, 0.2])
    a = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if a.report:
        return report()
    if a.evidence:
        return evidence(epochs=a.epochs, ms=tuple(a.ms))
    for d, m, s in JOBS:
        f, lock = OUT / "raw" / f"{name(d, m, s)}.json", OUT / "raw" / f"{name(d, m, s)}.lock"
        if f.exists() or (OUT / "STOP").exists():
            continue
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0)
            except (ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)   # lock de um processo que morreu: a próxima chamada refaz
            continue
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        print(f"=== {name(d, m, s)}", flush=True)
        print(run_job(d, m, s), flush=True)
        lock.unlink(missing_ok=True)
        if a.one:
            return
    sys.exit(3)


if __name__ == "__main__":
    main()
