# fl-robustness-lab

Ambiente de simulação de Aprendizado Federado (FL) para avaliação de ataques
e defesas, construído a partir da revisão sistemática
`revisao-ataques-defesas-fl-v2`. Ver `../entregavel_02/cenario-avaliacao-fl.tex` (ou o
PDF correspondente) para a definição completa do cenário de avaliação.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU (CUDA) é detectada automaticamente; funciona também em CPU.

## Validar o ambiente

```bash
source .venv/bin/activate
python tests/smoke_test.py
```

Executa todas as combinações relevantes de ataque x defesa em modo reduzido
(poucos clientes/amostras/rounds) só para confirmar que o pipeline roda
ponta-a-ponta sem erros. Não mede robustez real.

## Rodar um cenário

```bash
source .venv/bin/activate
python experiments/run_scenario.py --scenario scenarios/minmax_boba_labelskew.yaml
# ou em modo rápido para checar rapidamente:
python experiments/run_scenario.py --scenario scenarios/minmax_boba_labelskew.yaml --quick
```

Resultados em `results/<nome_do_cenario>_s<seed>[_quick].json`. Use `--seed N`
para sobrescrever a semente do YAML.

## Rodar a matriz completa e consolidar

```bash
source .venv/bin/activate
python tests/test_fixes.py                          # testes unitários (dados sintéticos, segundos)
python experiments/run_matrix.py --seeds 42 --only baseline_fedavg_no_attack   # mede s/round
python experiments/run_matrix.py                    # 13 cenários x sementes 42 43 44 (retomável)
python experiments/aggregate_results.py             # tabelas + figuras em results/summary/
```

`run_matrix.py` pula execuções cujo JSON já existe, então pode ser interrompido
e retomado. `aggregate_results.py` gera `summary.{csv,md,tex}` (média ± desvio
entre sementes) e as figuras `final_ma.png`, `backdoor_ba.png`,
`detection_f1.png` e `curves.png`.

## Notas de implementação

- `backdoor_pgd`: treino malicioso projetado (PGD) na bola L2 de raio ε/scale
  em torno do modelo global e depois escalado por `scale` — norma final ≤ ε.
  ε = `attack_params.max_norm` ou, por padrão, a mediana das normas benignas
  do round × `norm_multiplier`.
- BOBA: os vértices do simplex são *deltas* de referência por classe (treino
  local no holdout da classe k, mesmo lr e nº de passos de um cliente típico),
  com o mesmo sinal e escala dos deltas dos clientes.
- RAGNAR: o bootstrap exclui a família do ataque avaliado
  (`defense_params.bootstrap_exclude_eval_attack`, padrão `true`), e a
  referência do cosseno é configurável (`cos_ref: mean_delta | global_model`).

## Estrutura

- `src/config.py` — `ScenarioConfig` (contrato de definição de cenário).
- `src/models/cnn.py` — `SmallCNN` para CIFAR-10.
- `src/data/` — carregamento do CIFAR-10, particionamento não-IID (Dirichlet), datasets de ataque.
- `src/attacks/` — envenenamento de dados, backdoor, ataques de modelo "oniscientes" (IPM, LIE, MinMax/MinSum, etc).
- `src/defenses/` — FedAvg, Median, Trimmed Mean, Krum/Multi-Krum, FoolsGold, multi-métrica com branqueamento, BOBA, RAGNAR.
- `src/federated/` — treino local do cliente e orquestração dos rounds.
- `scenarios/*.yaml` — 13 cenários pré-definidos cobrindo a taxonomia da revisão.
- `experiments/run_scenario.py` — CLI de execução.
- `tests/smoke_test.py` — validação do ambiente.
