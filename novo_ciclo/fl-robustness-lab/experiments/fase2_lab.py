#!/usr/bin/env python
"""Fase 2 (16/10/2026, depois da busca bibliográfica) — desenho e critérios FIXADOS ANTES DE RODAR.

Contexto: o LayerGuard v2 (layer_guard_sn) barrou o LP com ruído na semente 42 e falhou nas sementes 44 e 45.
A busca achou o FedSurrogate (arXiv 2605.11122), defesa por camada crítica avaliada contra o LP só com α = 0,5.

Observação feita ANTES desta fase (logs _lp_log dos treinos já existentes): na semente 42 o Multi-Krum simulado
pelo atacante rejeitava L* inteiro e o LP encolhia L* para só fc1.weight; nas sementes 44 e 45 o Multi-Krum
aceitava, e L* ficou com 3 a 6 tensores (features.3/6.weight, fc1.weight, fc2...).

Fase A — FedSurrogate como linha de base (src/defenses/fedsurrogate.py), CIFAR-10, 20 clientes, 40 rounds.
  Q-A: o FedSurrogate barra o LP com α = 0,1?  "Resolve" se ASR <= 10% em lp_full_indiv_nu1.0 nas sementes 42, 44
  e 45 E acurácia limpa (semente 42) >= FedAvg - 2 pp. Também: lp_full e lp_full_indiv (semente 42).
  Verificação da reimplementação: lp_full com α = 0,5 (semente 42) — o artigo relata ASR ~1%; se aqui der > 10%,
  a reimplementação ou o cenário diferem e isso é reportado.
Fase B — diagnóstico (defesa lg_shadow: agrega como FedAvg e registra z_sim/z_norm por round, tensor e cliente).
  Sementes de DESENVOLVIMENTO 42, 44, 45, lp_full_indiv_nu1.0; mais lp_full_indiv (ν = 0) na 44.
  H5: nas sementes 44/45 o escore de cada atacante fica <= 3 em cada tensor de L* na maioria dos rounds (sinal
  diluído em vários tensores), enquanto na 42 passa de 3 em fc1.weight. Descritivo; decide o desenho da Fase C.
  Sementes de TESTE 46, 47, 48 (lg_shadow = FedAvg; dá a ASR sem defesa). Os registros dessas sementes NÃO
  serão olhados antes de a correção ser fixada (só a ASR/acurácia do FedAvg).
Fase C — correção: escolhida com os registros das sementes 42/44/45 (fora do circuito), fixada num adendo
  aqui antes de rodar, avaliada em circuito fechado nas sementes de desenvolvimento e depois nas de teste.
  Critério final (sementes de teste, α = 0,1): para cada semente com ASR(FedAvg) >= 40% em lp_full_indiv_nu1.0,
  ASR(correção) <= 10%; e acurácia limpa >= FedAvg - 2 pp. Reporta-se tudo, inclusive falhas.

ADENDO 16/10 (tarde) — resultado da Fase B e Fase C redefinida ANTES de rodar a Fase C:
  H5 CONFIRMADA (experiments/fase2_diag.py): nas sementes 44/45 o escore de cada atacante fica abaixo de 3 em
  todos os tensores de L* em >= 58% dos rounds (fc1.weight: 0%); o cosseno entre os dois atacantes (0,13-0,41)
  fica abaixo do máximo de pares honestos parecidos (p95 do máx. por cliente 0,38-0,70).
  Correções fora do circuito sobre os registros de 42/44/45 (limiar menor, média exponencial, soma no cliente,
  evidência acumulada por par): NENHUMA marca >= 50% da massa de L* dos atacantes nas três sementes com <= 5%
  dos parâmetros honestos (melhor na 45: 14% com 15% de honestos). Por isso nenhuma correção por estatística de
  atualização vai para o circuito fechado — conclusão: sob α = 0,1, o LP evasivo com ruído e L* em vários
  tensores não é separável pelas atualizações, como os backdoors treinados em separado.
  Fase C nova — outra fonte de informação: REPARO DO MODELO NO SERVIDOR com as 100 amostras limpas que o
  laboratório já supõe (server_holdout, 10 por classe; mesma suposição de BOBA/RAGNAR/FLTrust), aplicado ao
  modelo global final, sem saber gatilho nem classe-alvo (src/defenses/repair.py, experiments/fase2_repair.py):
    FP    fine-pruning (Liu et al., 2018): poda neurônios de fc1 em ordem crescente de ativação média nas amostras
          limpas, em passos de 5%, enquanto a acurácia nessas amostras não cair mais que 5 pp;
    FT    ajuste fino: 5 épocas de SGD (lr 0,01, lote 32) nas 100 amostras;
    FP+FT as duas, nessa ordem.
  Desenvolvimento (escolha da variante, pela menor ASR média com perda de acurácia <= 2 pp): semente 42 com
  os 7 backdoors + sem ataque, sementes 44/45 com ν = 1 (treinos FedAvg que salvam o modelo final).
  Teste (critério final, sementes 46, 47, 48, α = 0,1, variante escolhida):
    F1 para cada (ataque em {ν = 1, pixel, DBA}, semente) com ASR(FedAvg) >= 40%: ASR após o reparo <= 10%;
    F2 acurácia limpa após o reparo >= FedAvg limpo - 2 pp em cada semente.
  ADENDO 2 (16/10, depois de ver SÓ o primeiro modelo de desenvolvimento, semente 44 ν = 1: FP 59,7% -> 14,2%
  de ASR com -5,3 pp de acurácia; FT com lr 0,01 perde 5 pp sozinho): acrescentadas, ainda no desenvolvimento,
  as variantes FP2 e FP3 (limite de queda de 2 e 3 pp nas amostras limpas), FTb (lr 0,001) e FP2+FTb. A regra de
  escolha e os critérios F1/F2 nas sementes de teste não mudam.
  ADENDO 3 (16/10, mesmo modelo): curva FP x fração podada na semente 44 — ASR 60 -> 15% só com ~60% podado e
  -5 pp no teste; o holdout de 100 amostras não enxerga essa perda e o FT nas 100 amostras não a recupera.
  Nova variante, em circuito fechado (src/defenses/repair.py::fedavg_prune): FedAvg; no round 30 o servidor
  aplica FP (limite 5 pp no holdout) ao modelo global e mantém os neurônios podados em zero até o round 40 —
  os 10 rounds restantes recuperam a acurácia com os honestos, e os atacantes continuam atacando (e o LP
  escolhe L* já no modelo podado). Desenvolvimento: ν = 1 nas sementes 42/44/45, sem ataque, pixel e DBA na 42.
  Se passar no desenvolvimento (ASR <= 10% e acurácia >= FedAvg - 2 pp na maioria), vai às sementes de teste
  com F1/F2 inalterados.
  ADENDO 4 (16/10): FP pós-treino nas sementes 42/45 também falha (só ~33-42% podado antes de o holdout cair;
  pelas curvas, a ASR só cai com ~80% podado, com a acurácia já em ~47%). Testada inversão de gatilho no servidor
  (Neural Cleanse enxuto, src/defenses/nc.py, experiments/fase2_nc.py e fase2_compact.py) nos modelos de
  desenvolvimento: toda classe é alcançável com máscara de ~13-30 pixels (a CNN pequena não é robusta); a classe-
  alvo tem a menor máscara em 2 de 3 sementes, mas o índice de anomalia (1,0-2,5) e a compacidade (z 1,5-2,9) não
  separam de forma confiável. A máscara da classe-alvo cobre parte do gatilho real (17-40% da massa no canto),
  as outras classes nada — o sinal existe, mas é fraco. Não vai às sementes de teste.
  Avaliação entre sementes (descritiva, para o artigo): FedAvg, LayerGuard v2 e FedSurrogate em ν = 1 nas seis
  sementes 42, 44-48; Multi-Krum e FLAME em ν = 1 nas 44/45.
  ADENDO 5 (16/10) — VARIANTE FINAL FIXADA ANTES DE OLHAR QUALQUER SEMENTE DE TESTE:
  Desenvolvimento (semente 42): o NC detecta pixel (índice 5,1) e DBA (4,6) com a classe certa (0) e não detecta
  nada no modelo sem ataque (1,2); não detecta o LP (1,0-2,5; com orçamento fixo de 8 pixels, z 1,4-3,7, e o
  modelo limpo já dá 2,0 na classe 0). Poda direcionada custa 3-5 pp no pixel; desaprendizado (ajuste fino com
  o gatilho reconstruído em metade das amostras e rótulo verdadeiro) zera a ASR: pixel 97 -> 1,2%, DBA 48 -> 0,4%,
  com lr 0,001 e 10 épocas (-2,0 pp).
  NC-U = detect (índice > 2) + nc_unlearn(lr 0,001, 10 épocas, frac 0,5), src/defenses/nc.py; sem alvo, nada.
  Defesa combinada avaliada = LayerGuard v2 (layer_guard_sn) no treino + NC-U no modelo final.
  Teste (sementes 46, 47, 48): F1 e F2 como acima, para a combinada; também NC-U sobre o FedAvg (sem LayerGuard).
  ADENDO 6 (16/10, desenvolvimento, antes do teste): NC-U detecta também o PGD (índice 3,6) mas só reduz a ASR de
  94% para 71%; no LP da semente 45 (detectado, 2,5) leva 76% -> 8%. Variante "forçada" (desaprende sempre a
  classe de menor máscara, sem limiar; experiments/fase2_forced.py) REJEITADA: custa 4,8 pp no modelo limpo e
  acerta a classe só em 2 de 3 LP (s44: ASR 60 -> 40%). NC-U do adendo 5 segue como a variante final.
  As linhas de base sem reparo, o FedSurrogate e o LayerGuard v2 seguem reportados ao lado.

Execução: vários processos dividem a fila (trava .lock por treino); cada um só começa um treino se houver
>= 2,5 GB de RAM disponível.
    nohup .venv/bin/python -u experiments/fase2_lab.py >> results/fase2.log 2>&1 &
    python experiments/fase2_lab.py --analyze-only
Saídas: results/fase2/{raw/*.json, summary.md}.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.final_lab import cfg_for
from src.federated.simulate import run

OUT = Path("results/fase2")
DEV, TEST = (42, 44, 45), (46, 47, 48)


def C(attack, defense, alpha=0.1, seed=42, **dp):
    cfg = cfg_for(attack, defense, alpha, seed=seed)
    cfg.name += f"_s{seed}"
    cfg.defense_params = dict(dp)
    return cfg


def jobs():
    # lg_shadow 2ª versão (matriz de cossenos completa); os registros v1 de 44/45 ficaram como *.v1json
    j = [C("lp_full_indiv_nu1.0", "lg_shadow", seed=s) for s in DEV]
    j += [C("lp_full_indiv_nu1.0", "fedsurrogate", seed=s) for s in DEV]
    j += [C("lp_full_indiv", "lg_shadow", seed=44)]
    return j + phase_c()


BACKDOORS = ["lp_full", "lp_full_indiv", "lp_full_indiv_nu1.0", "lp_full_indiv_nu1.0k", "backdoor_pixel", "backdoor_pgd", "dba"]


def phase_c():
    """Treinos FedAvg que salvam o modelo final (FL_SAVE_FINAL), para o reparo fora do circuito."""
    j = [C("lp_full_indiv_nu1.0", "fedavg", seed=s) for s in (42, 44, 45)]
    j += [C(a, "fedavg") for a in ["none", "backdoor_pixel", "dba", "lp_full", "lp_full_indiv", "lp_full_indiv_nu1.0k", "backdoor_pgd"]]
    j += [C("lp_full_indiv_nu1.0", "fedavg_prune", seed=s) for s in (44, 42)]   # em andamento
    # teste da defesa final primeiro (adendo 5), depois o resto
    for s in TEST:
        j += [C(a, "fedavg", seed=s) for a in ("none", "lp_full_indiv_nu1.0", "backdoor_pixel", "dba")]
        j += [C(a, "layer_guard_sn", seed=s) for a in ("none", "lp_full_indiv_nu1.0", "backdoor_pixel", "dba")]
    j += [C("lp_full_indiv_nu1.0", "fedsurrogate", seed=s) for s in TEST]
    j += [C("lp_full_indiv_nu1.0", "fedavg_prune", seed=45)] + [C(a, "fedavg_prune") for a in ("none", "backdoor_pixel", "dba")]
    j += [C("lp_full_indiv_nu1.0", d, seed=s) for s in (44, 45) for d in ("multi_krum", "flame")]
    j += [C("none", "fedsurrogate"), C("lp_full", "fedsurrogate"), C("lp_full_indiv", "fedsurrogate"),
          C("lp_full", "fedsurrogate", alpha=0.5)]
    j += [C("lp_full_indiv_nu1.0", "lg_shadow", seed=s) for s in TEST]   # só se sobrar tempo (registros)
    return j


def claim(name):
    lock = OUT / "raw" / f"{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0)
            return False
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True)
            return claim(name)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def mem_ok(gb=2.5):
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6 >= gb
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--one", action="store_true", help="roda o primeiro treino livre e sai (código 3 se não há)")
    args = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FL_SAVE_FINAL", str(OUT / "models"))
    if not args.analyze_only:
        t0 = time.time()
        for cfg in jobs():
            if (OUT / "raw" / f"{cfg.name}.json").exists():
                continue
            if (OUT / "STOP3").exists():
                print("STOP encontrado; saindo entre treinos", flush=True); break
            while not mem_ok():
                time.sleep(30)
            if not claim(cfg.name):
                continue
            print(f"\n=== {cfg.name} pid={os.getpid()} (decorrido {(time.time() - t0) / 60:.1f} min)", flush=True)
            res = run(cfg)
            json.dump(res, open(OUT / "raw" / f"{cfg.name}.json", "w"))
            (OUT / "raw" / f"{cfg.name}.lock").unlink(missing_ok=True)
            if args.one:
                return
        if args.one:
            sys.exit(3)
    from experiments.fase2_analyze import summarize
    print(summarize())


if __name__ == "__main__":
    main()
