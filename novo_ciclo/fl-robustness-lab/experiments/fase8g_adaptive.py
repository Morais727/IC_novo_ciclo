#!/usr/bin/env python
"""Fase 8G (10/11/2026) — adversário adaptativo CIENTE DA CORROBORAÇÃO (item 468 do painel).

Pergunta de revisor: "um atacante que conhece a sua defesa a derruba?". Desenho FIXADO ANTES DE RODAR.

Mesmo cenário do artigo da Fase 8E (ResNet18-GN / CIFAR-10, 20 nós, 6 maliciosos, Dirichlet α 0,9, 205 rodadas,
ataque 80–200, gatilho BadNet, alvo 4), MUDANDO só o atacante: liga `evade_corr` no laysel_attack. O atacante é
onisciente — conhece a regra da layer_guard_corr, o limiar T e os deltas honestos do round — e, a cada rodada,
CONCENTRA o veneno em m tensores (revertendo os demais sensíveis ao próprio delta limpo) e DESSINCRONIZA os
maliciosos (ruído nu) para manter a contagem de corroboração ABAIXO de T (ver src/attacks/recent.py::_evade_corr).

Comparação:
  - PORTÃO (lg_shadow, sem defesa): o ataque concentrado ainda PEGA? (ASR final >= 20%). Se não pega, o próprio
    custo da evasão já contém o backdoor — resultado para o artigo.
  - só se pegar: layer_guard_corr (a corr CEDE ao adaptativo?), + client e corr-soft (variantes de purga).
O JSON guarda attack_params["_evade_corr_log"] — m escolhido, nu, contagem dos maliciosos e se evadiu, por round:
a evidência de SE/QUANDO a corr cede (m p/ evadir vs. m p/ o backdoor pegar).
Uso: python experiments/fase8g_adaptive.py --one [--gate-only] | --report | --missing [--seed 42] [--T 8]
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import fase6, fase8e
from experiments.fase4_alpha import mem_ok
from src.federated.simulate import run
# 13/11/2026: salva o modelo final de toda corrida (artigo 2)
os.environ.setdefault("FL_SAVE_FINAL", "results/models_final")
from src.data import datasets

datasets.set_trigger("badnet_cifar")

OUT = Path("results/fase8g")
SH = fase6.SH
ALPHA, SEED, T = 0.9, 42, 8
DEFENSES = [SH, "layer_guard_corr", "layer_guard_client", "layer_guard_corr_soft"]
AGG_AWARE = ("layer_guard_corr_soft_tm", "layer_guard_corr_soft_clip")   # resposta ao adaptativo (15/11)


def cfg(defense):
    c = fase8e.cfg(defense)                      # reusa o cenário fiel do artigo
    c.attack_params["evade_corr"] = True         # <<< liga o adversário adaptativo ciente da corroboração
    c.attack_params["evade_T"] = T               # o atacante mira exatamente o T da defesa
    c.attack_params["confirm_tensors"] = T       # a defesa corr usa o mesmo T
    if defense in AGG_AWARE:                     # 15/11 (fase16): atacante também ciente da AGREGAÇÃO da variante
        c.attack_params["evade_score_defense"] = defense
    c.name = f"laysel_adapt_T{T}__{defense}__a{ALPHA}_s{SEED}"
    return c


def path(defense):
    return OUT / "raw" / f"laysel_adapt_T{T}__{defense}__a{ALPHA}_s{SEED}.json"


def final(defense):
    f = path(defense)
    if not f.exists():
        return None
    h = json.load(open(f))["history"][-1]
    return dict(asr=h["backdoor_asr"], acc=h["main_accuracy"])


def gate():
    r = final(SH)
    return None if r is None else r["asr"] >= 20


def jobs(gate_only=False):
    j = [SH] if final(SH) is None else []
    if gate_only:
        return j
    if gate():
        j += [d for d in DEFENSES if d != SH and final(d) is None]
    return j


def report():
    out = ["# Fase 8G — adversário adaptativo ciente da corroboração",
           f"Cenário fiel do LaySelFL (8E) com evade_corr ligado, T={T}. Atacante concentra/dessincroniza p/ ficar < T.", ""]
    g = final(SH)
    out.append("Portão (sem defesa): " + ("ainda não rodou" if g is None else
               f"ASR {g['asr']:.1f}%, acurácia {g['acc']:.1f}% → "
               f"{'PEGOU (>= 20%) — evasão não se autocontém' if g['asr'] >= 20 else 'NÃO pegou — o custo da evasão já contém o backdoor'}"))
    if g and g["asr"] >= 20:
        out += ["", "| defesa | ASR | acurácia | cede? |", "|---|---|---|---|"]
        for d in DEFENSES:
            r = final(d)
            if r:
                nome = "sem defesa" if d == SH else d
                cede = "—" if d == SH else ("CEDE" if r["asr"] >= 20 else "contém")
                out.append(f"| {nome} | {r['asr']:.1f}% | {r['acc']:.1f}% | {cede} |")
        # resumo da evasão (m/nu médios na janela de ataque) a partir do portão
        f = path(SH)
        if f.exists():
            ev = json.load(open(f)).get("attack_params", {}).get("_evade_corr_log", [])
            if ev:
                ms = [e["m_kept"] for e in ev]
                evd = sum(e["evades"] for e in ev)
                out += ["", f"Evasão (portão): m mantido médio {sum(ms)/len(ms):.1f} de {ev[0]['n_sens']} sensíveis; "
                            f"evadiu em {evd}/{len(ev)} rodadas de ataque."]
    txt = "\n".join(out) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTADO.md").write_text(txt)
    print(txt)


def main():
    global SEED, T
    ap = argparse.ArgumentParser()
    for f in ("--one", "--report", "--missing", "--gate-only"):
        ap.add_argument(f, action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--T", type=int, default=T)
    ap.add_argument("--no-gate", action="store_true", help="13/11: roda a defesa sem esperar o portão (prazo da máquina dl-16)")
    ap.add_argument("--defense", help="roda só esta defesa (12/11: permite paralelizar as defesas entre GPUs; defesas só rodam se o portão já pegou)")
    a = ap.parse_args()
    SEED, T = a.seed, a.T
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if a.report:
        return report()
    if a.missing:
        j = jobs(a.gate_only); print(len(j)); [print(x) for x in j]; return
    todo = jobs(a.gate_only)
    if a.defense:
        todo = [d for d in todo if d == a.defense]
    if a.no_gate and a.defense and final(a.defense) is None:
        todo = [a.defense]
    for d in todo:
        f = path(d)
        if f.exists():
            continue
        if (OUT / "STOP").exists():
            print("STOP encontrado", flush=True); return
        while not mem_ok():
            time.sleep(30)
        lock = f.with_suffix(".lock")
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
        c = cfg(d)
        print(f"\n=== {c.name} pid={os.getpid()} device={c.device}", flush=True)
        json.dump(run(c), open(f, "w"), default=str)
        lock.unlink(missing_ok=True)
        if a.one:
            return
    if a.one:
        sys.exit(3)


if __name__ == "__main__":
    main()
