#!/usr/bin/env python3
"""Gera o Relatório 4 (mês 04: conjunto de ataques injetáveis funcional) no padrão visual H.IAAC.

Reaproveita os blocos XML e a montagem de ../entregavel_03/build_relatorio.py (modelo = .docx do Relatório 2:
cabeçalho com logo, rodapé "Página X de Y", títulos laranja, Arial 11). Só o conteúdo muda. Os números vêm de
../fl-robustness-lab/results/ (attack_effect, fase5, fase7, fase8, fase8a, repro_lp_seeded).

    python3 build_relatorio.py      # -> [HIAAC][M04] Relatório 4.docx e .pdf
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m3", HERE.parent / "entregavel_03" / "build_relatorio.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
p, h1, h2, bullet, table, box, empty, run, num = m3.p, m3.h1, m3.h2, m3.bullet, m3.table, m3.box, m3.empty, m3.run, m3.num
BORDERS, TEXT_W = m3.BORDERS, m3.TEXT_W


def info_table():
    rows = [("Pesquisador:", "Marcos Gilliard Oliveira Morais"), ("Orientador:", "Leandro A. Villas"),
            ("Vínculo", "Pesquisador Visitante Convidado - Iniciação Científica"),
            ("Período tratado no relatório:", "01/10/2026 a 31/10/2026")]
    trs = "".join(
        f'<w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="2736"/></w:tcPr><w:p>{run(a, bold=True, size=20)}</w:p></w:tc>'
        f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="6624"/></w:tcPr><w:p>{run(b, size=20)}</w:p></w:tc></w:tr>'
        for a, b in rows)
    return (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{TEXT_W}"/>{BORDERS.replace("{c}", "000000")}'
            f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid><w:gridCol w:w="2736"/><w:gridCol w:w="6624"/>'
            f"</w:tblGrid>{trs}</w:tbl>")


# ----------------------------------------------------------------------------- dados
# Fase 7 (results/fase7/RESULTADO.md, linha "Sem defesa"): ASR médio das sementes 46, 47, 48 e acurácia média
ALPHAS = ("0,5", "0,4", "0,3", "0,2", "0,1")
LP_VARIANTS = [
    ("LP original (`lp_full`)", (57.9, 33.0, 45.1, 58.9, 71.9)),
    ("LP evasivo (`lp_full_indiv`)", (58.9, 32.4, 45.1, 57.0, 70.9)),
    ("LP + ruído (`lp_full_indiv_nu1.0`)", (55.2, 35.5, 45.3, 57.0, 73.3)),
]
CLEAN_ACC = (71.5, 71.6, 70.6, 69.0, 66.5)


EFEITO = HERE.parent / "fl-robustness-lab" / "results" / "m4_efeito" / "efeito.csv"
HIDE = {"Ruído gaussiano (m = 3)", "Ruído gaussiano (m = 10)", "Ruído gaussiano (m = 30)"}
RENAME = {"Ruído gaussiano": "Ruído gaussiano (desvio 5)", "Ruído gaussiano (m = 1)": "Ruído gaussiano (m = 1, padrão)",
          "Edge-case PGD, imagens externas *": "Edge-case PGD, aviões externos *"}


def effect_rows():
    import csv
    br = lambda x: x.replace(".", ",").replace("-", "−").replace("(−", "(−").replace("–−", " a −")
    rows = []
    for r in csv.DictReader(open(EFEITO)):
        if r["ataque"] in HIDE:
            continue
        drop = r["medida"] == "queda de acurácia"
        rows.append([RENAME.get(r["ataque"], r["ataque"]), r["maliciosos"], br(r["acc"]), br(r["dacc"].split(" (")[0]),
                     r["medida"].replace("Southwest", "externos"), "—" if drop else br(r["efeito"]),
                     "—" if drop else br(r["base"])])
    return rows


def content():
    B = []
    A = B.append
    A(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/>'
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 04</w:t></w:r></w:p>')
    A(h1("Conjunto de Ataques Injetáveis em Aprendizado Federado", center=True))
    A(empty()); A(info_table()); A(empty())

    # 1 ---------------------------------------------------------------
    A(h1("1. Objetivo e Escopo"))
    A(p("Este relatório é o quarto entregável do plano de 12 meses. O plano pede, para o mês 4, a **implementação de "
        "ataques controlados (envenenamento)** e, como entregável, um **conjunto de ataques injetáveis funcional**: "
        "ataques que podem ser ligados em qualquer cenário do laboratório por configuração, com efeito medido e "
        "documentado. Esses ataques são a base de validação do agente de detecção (meses 5 a 10)."))
    A(p("O conjunto partiu dos 8 ataques avaliados até o mês 2 e dos 5 ataques recentes implementados no mês 3. Em "
        "outubro ele cresceu em duas direções: (i) o **LP** [Zhuang et al. 2024], o ataque por camadas mais forte da "
        "revisão, ganhou uma versão fiel ao código dos autores e três variantes mais difíceis de detectar, incluindo um "
        "atacante adaptativo que conhece a defesa; (ii) foram implementados os **dois ataques por camada publicados em "
        "2026**, LSA [Foroughi et al. 2026] e LaySelFL [Li et al. 2026]."))
    A(h2("1.1. Entregável desta Etapa"))
    A(bullet("**Este documento**: catálogo dos 21 ataques (seção 2), como injetá-los (seção 3), validação de cada um "
             "sem defesa (seção 4), testes automatizados (seção 5) e limitações (seção 6)."))
    A(bullet("**Código** em `fl-robustness-lab/`: `src/attacks/` (ataques e registro único `registry.py`), "
             "`src/federated/simulate.py` (ponto de injeção no round), `tests/test_recent_attacks.py` (8 testes) e os "
             "experimentos de validação `experiments/attack_effect.py`, `m4_efeito.py`, `fase7.py`, `fase8_r18.py` e `fase8a.py`."))
    A(h2("1.2. Modelo de Ameaça"))
    A(bullet("O atacante controla uma fração fixa dos clientes (2 de 20 na maioria dos backdoors; 4 de 20 no DBA e nos "
             "ataques não direcionados), que agem em conluio durante todo o treino."))
    A(bullet("Ataques de dados alteram só os dados locais; ataques de modelo e por camada controlam a atualização enviada."))
    A(bullet("Os ataques de modelo são *oniscientes* (veem as atualizações honestas do round), como no Blades e no BOBA; "
             "isso torna a avaliação de qualquer defesa conservadora."))
    A(bullet("Só o LP adaptativo conhece a regra de defesa; todos os outros ignoram o detector."))

    # 2 ---------------------------------------------------------------
    A(h1("2. Catálogo de Ataques"))
    A(p("Cada ataque tem um nome no registro (`src/attacks/registry.py`) e um *tipo*, que diz em que ponto do round ele "
        "age. Asterisco: implementado ou ampliado em outubro."))
    A(table(["Nome no registro", "Ataque", "Tipo", "Referência", "O que faz"], [
        ["`label_flip`", "Troca de rótulos", "dados", "Tolpegin et al. 2020", "troca o rótulo das amostras locais"],
        ["`backdoor_pixel`", "Backdoor por gatilho", "dados", "Gu et al. 2017", "gatilho 4×4 no canto e rótulo-alvo em parte das amostras"],
        ["`dba`", "DBA", "dados", "Xie et al. 2020", "gatilho dividido em 4 faixas, uma por atacante"],
        ["`backdoor_pgd`", "Backdoor com PGD", "dados + pós", "Wang et al. 2020", "gatilho + projeção na bola de norma dos honestos + escala"],
        ["`edge_case_pgd` *", "Edge-case PGD", "dados + pós", "Wang et al. 2020", "amostras raras rotuladas como outra classe + PGD; opção de aviões externos"],
        ["`gaussian_noise` *", "Ruído gaussiano", "modelo", "Blanchard et al. 2017", "ruído com desvio 5 ou, calibrado, com norma m × mediana honesta"],
        ["`sign_flip`", "Inversão de sinal", "modelo", "clássico (Blades)", "envia −1 × média honesta"],
        ["`ipm`", "IPM", "modelo", "Xie et al. 2019", "média honesta com sinal invertido e escala ε"],
        ["`lie`", "LIE", "modelo", "Baruch et al. 2019", "média − z·desvio, dentro da variação honesta"],
        ["`minmax` / `minsum`", "MinMax / MinSum", "modelo", "Shejwalkar e Houmansadr 2021", "maior desvio que não supera a distância máx./soma dos honestos"],
        ["`mimic`", "Mimic", "modelo", "Karimireddy et al. 2022", "todos copiam o mesmo honesto mais extremo"],
        ["`lp` / `lf`", "LP / LF simplificados", "por camada", "Zhuang et al. 2024", "envenena (LP) ou inverte (LF) só as camadas críticas"],
        ["`lp_full` *", "LP como no código dos autores", "por camada", "Zhuang et al. 2024", "camadas críticas por tensor, λ, controle adaptativo de camadas"],
        ["`lp_full_indiv` *", "LP evasivo", "por camada", "variante nossa", "cada atacante usa o próprio delta limpo como base"],
        ["`lp_full_indiv_nu1.0` *", "LP + ruído", "por camada", "variante nossa", "LP evasivo + ruído independente nas camadas críticas"],
        ["`lp_adapt` *", "LP adaptativo", "por camada", "variante nossa", "escolhe força e ruído a cada round para passar pela defesa"],
        ["`lsa` *", "LSA", "por camada", "Foroughi et al. 2026", "camadas críticas por substituição + ajuste fino só nelas"],
        ["`laysel` *", "LaySelFL", "por camada", "Li et al. 2026", "camadas sensíveis por distância L1, treino preso ao servidor, corte das demais"],
    ], [2100, 1600, 1060, 1800, 2800], size=15, bold_first_col=False))
    A(p("Contando `minmax`/`minsum` e `lp`/`lf` separadamente, são **21 ataques**: 5 de dados, 7 de modelo e 9 por camada."))
    A(h2("2.1. Os ataques novos de outubro"))
    A(bullet("**LP fiel ao código dos autores** (`lp_full`). Reproduz `models/Attacker.py` do repositório público: cada "
             "atacante treina limpo; um treina com backdoor; as camadas críticas são achadas por substituição *tensor a "
             "tensor* (a granularidade do código); um controle adaptativo simula a defesa do servidor e tira camadas até "
             "o vetor passar; o envio é λ·envenenado + (1 − λ)·média limpa nas camadas críticas."))
    A(bullet("**Variantes mais difíceis de detectar** (não estão no artigo): no *evasivo*, cada atacante parte do próprio "
             "delta limpo, então os atacantes só coincidem nas camadas críticas; no *LP + ruído*, cada um ainda soma ruído "
             "independente de mesma norma nessas camadas; no *adaptativo*, o atacante conhece a regra e o limiar da "
             "defesa e, a cada round, varre força e ruído e envia a combinação mais forte que passa."))
    A(bullet("**LSA** [Foroughi et al. 2026, IEEE ICC]: a partir do LP, ajusta as camadas críticas nos dados envenenados "
             "com as outras congeladas, para que se encaixem no resto do modelo (a \"suavização\"); refaz a busca de "
             "camadas a cada 5 rounds. τ = 0,8 e λ = 1, os melhores valores do artigo."))
    A(bullet("**LaySelFL** [Li et al. 2026, CVPR]: escolhe as camadas sensíveis pela distância L1 entre os treinos "
             "limpo e com backdoor; treina, alternadamente, metade delas com a perda do backdoor mais 20·‖θ − θ_servidor‖²; "
             "e corta em ±0,05, elemento a elemento, a diferença das demais camadas em relação ao servidor. Cada "
             "atacante faz isso sozinho, com os próprios dados."))
    A(p("Nenhum dos dois artigos de 2026 tem código público; ambos foram implementados a partir do texto completo. "
        "Onde o texto não diz, as escolhas foram fixadas **antes** de rodar e estão na tabela abaixo."))
    A(table(["Ataque", "Ponto não especificado no artigo", "Escolha"], [
        ["LSA", "o que é uma \"camada\"", "um tensor de parâmetros (como o código do LP, em que a LSA se baseia)"],
        ["LSA", "épocas do ajuste fino", "1, a época local do cenário"],
        ["LSA", "simulação da defesa", "não usada (o artigo não a descreve)"],
        ["LaySelFL", "nº de camadas sensíveis k", "total de tensores − 2 (= 60 de 62 na ResNet18, valor do artigo)"],
        ["LaySelFL", "nº de grupos alternados P / épocas N da seleção", "P = 2 / N = 2"],
        ["LaySelFL", "ponto de partida do treino com backdoor", "o modelo já treinado limpo; âncora = modelo do servidor"],
        ["LaySelFL", "rodada inicial do ataque", "1ª rodada (o artigo começa na 80ª de 205)"],
    ], [1500, 3700, 4160], size=16))

    # 3 ---------------------------------------------------------------
    A(h1("3. Como Injetar um Ataque"))
    A(p("Um cenário é um `ScenarioConfig` (ou um YAML em `scenarios/`). Três campos injetam o ataque: `attack` (o nome no "
        "registro), `num_malicious` e `attack_params`. Os maliciosos são sorteados uma vez, pela semente, e ficam fixos."))
    A(box("Exemplo: LSA contra a defesa do servidor, 2 atacantes em 20", [
        "`attack: lsa`", "`num_malicious: 2`", "`attack_params: {target_label: 0, tau: 0.8, lambda: 1.0, period: 5}`",
        "`defense: layer_guard_sn`", "`python experiments/run_scenario.py --scenario scenarios/<arquivo>.yaml`"]))
    A(p("Os dois parâmetros novos de outubro seguem o mesmo caminho: `attack_params: {norm_mult: 1}` no ruído gaussiano "
        "(seção 4.2) e `attack_params: {edge_source: southwest}` no Edge-case com aviões externos (imagens em "
        "`data_cache/southwest/`)."))
    A(p("Em cada round (`_train_round` em `simulate.py`), os honestos treinam primeiro; depois cada tipo age assim:"))
    A(bullet("**dados**: troca o conjunto local do atacante (ex.: com gatilho) antes do treino normal;"))
    A(bullet("**dados + pós**: idem, com o treino projetado numa bola de norma e o delta escalado no fim;"))
    A(bullet("**modelo**: ignora o treino e sintetiza o vetor a partir das atualizações honestas do round;"))
    A(bullet("**por camada**: recebe a função de treino local, o modelo global e os dados de todos os atacantes, e "
             "monta os deltas tensor a tensor; pode enviar o mesmo vetor por todos ou um vetor por atacante."))
    A(p("A métrica de sucesso é definida pelo próprio ataque (`eval` no registro): para os backdoors, a taxa de sucesso "
        "(ASR) é a fração das imagens de teste com o gatilho classificadas na classe-alvo; para a troca de rótulos, a "
        "fração da classe de origem classificada na classe de destino; para os demais, a queda de "
        "acurácia em relação ao treino sem ataque."))

    # 4 ---------------------------------------------------------------
    A(h1("4. Validação: os Ataques Funcionam?"))
    A(p("Cenário padrão: CIFAR-10, SmallCNN, 20 clientes, Dirichlet α = 0,5, 40 rounds, **sem defesa** (FedAvg). "
        "Um backdoor é considerado funcional se a ASR final passar de **20%**, regra fixada antes de cada experimento."))
    A(h2("4.1. Efeito de cada ataque sem defesa (4 sementes)"))
    A(p("Os 13 ataques dos meses 2 e 3, com as versões novas da seção 4.2, nas sementes 42, 46, 47 e 48 (75 treinos). "
        "Valores como média (mínimo–máximo). Δ = diferença média de acurácia em relação ao modelo treinado sem atacantes "
        "(72,0%, de 71,6 a 73,0). Para os ataques que só derrubam a acurácia, o efeito é o próprio Δ. Asterisco: "
        "ataques recentes do mês 3."))
    A(table(["Ataque", "Malic.", "Acurácia (%)", "Δ (p.p.)", "Medida de efeito", "Efeito (%)", "Sem atacantes (%)"],
            effect_rows(), [2000, 640, 1500, 900, 1820, 1300, 1200], size=15))
    A(bullet("**Todos os ataques têm efeito mensurável.** Os de modelo derrubam de 2,4 a 4,9 p.p. (o ruído antigo, 62 p.p.); "
             "os backdoors por gatilho passam de 65% de sucesso, com a acurácia a menos de 1,2 p.p. do normal."))
    A(bullet("**A variação entre sementes é grande** nos backdoors mais fracos: o LP vai de 50% a 86% e o DBA de 42% a 92%. "
             "Por isso a semente 42 sozinha, usada no mês 3, subestimava DBA, LP e Edge-case."))
    A(h2("4.2. Pendências do mês 3 resolvidas"))
    A(p("O Relatório 3 deixou três ataques com efeito mal medido. Em outubro, cada um ganhou uma medida ou uma versão "
        "nova, com regra de decisão fixada antes de rodar. Cenário da seção 4, sem defesa, sementes 42, 46, 47 e 48; "
        "valores como média (mínimo–máximo)."))
    A(table(["Ataque", "Medida", "Com ataque", "Modelo limpo", "Veredito"], [
        ["Troca de rótulos (classe 0 → 2, 4 atacantes)", "imagens da classe 0 classificadas como 2 (%)",
         "17,5 (13,5–20,2)", "4,9 (4,2–5,9)", "efeito visível"],
        ["", "acurácia da classe 0 (%)", "59,0 (56,6–61,5)", "78,4 (77,0–81,6)", ""],
        ["", "acurácia geral (%)", "70,9 (70,3–72,1)", "72,0 (71,6–73,0)", ""],
        ["Edge-case PGD, aviões externos (2 atacantes)", "aviões classificados como caminhão (%)",
         "24,7 (13,8–36,7)", "2,7 (2,0–3,6)", "funcional (média ≥ 20%)"],
    ], [2700, 2260, 1500, 1500, 1400], size=16))
    A(bullet("**Troca de rótulos.** A queda da acurácia geral (cerca de 1 p.p.) esconde o efeito. A medida certa é a "
             "taxa de troca na classe atacada: 17,5% das imagens de avião viram pássaro, contra 4,9% no modelo limpo, e a "
             "acurácia da classe cai 19 p.p. Essas duas medidas passam a acompanhar o ataque no laboratório."))
    A(bullet("**Edge-case.** A versão do mês 3 usava como amostras raras a cauda da própria classe avião do CIFAR-10 e "
             "ficou em 8% na semente 42. A nova usa as imagens externas de aviões do artigo original [Wang et al. 2020] (784 "
             "de treino, 196 de teste), rotuladas como caminhão, com o resto igual. Passou da regra de 20% pela média, "
             "mas varia muito entre sementes: em 2 das 4 ficou abaixo de 20%. Com as 4 sementes, a versão antiga também "
             "chega a 24,0% (8,0 a 36,0): o problema do mês 3 era a semente 42, não a fonte das imagens. As duas versões "
             "ficam no conjunto, com força parecida."))
    A(p("**Ruído gaussiano.** A versão antiga (desvio 5 por coordenada) derruba o modelo a 10%, o mesmo que chutar, e "
        "não serve para medir detecção. A nova calibra o tamanho do ruído pelo round: norma = m × mediana das normas "
        "dos clientes honestos. Regra fixada: o padrão é o menor m com queda média de pelo menos 2 p.p. e acurácia "
        "acima de 20%."))
    A(table(["m", "1", "3", "10", "30", "desvio 5 (antigo)"], [
        ["Queda de acurácia (p.p.)", "2,4 (1,7–3,2)", "2,6 (1,8–3,3)", "5,3 (2,6–7,1)", "28,1 (19,1–35,1)", "62,0"],
    ], [2160, 1400, 1400, 1400, 1500, 1500], size=16))
    A(p("Pela regra, o padrão passa a ser **m = 1** (`attack_params: {norm_mult: 1}`), que fica na faixa dos ataques "
        "LIE, MinMax e MinSum (3 a 4 p.p.). É um valor no limite: numa das sementes a queda foi de 1,7 p.p. Para "
        "testes com efeito mais claro, m = 10 dobra a queda sem derrubar o modelo."))
    A(h2("4.3. LP e variantes (3 sementes, 5 níveis de heterogeneidade)"))
    A(p("ASR média sem defesa nas sementes 46, 47 e 48, para cada α (quanto menor o α, mais heterogêneos os dados). "
        "São 45 treinos, parte da grade completa de outubro (seção 7)."))
    A(table(["Ataque"] + [f"α = {a}" for a in ALPHAS],
            [[n] + [num(v, 1) for v in vals] for n, vals in LP_VARIANTS]
            + [["Acurácia sem ataque"] + [num(v, 1) for v in CLEAN_ACC]],
            [2760, 1320, 1320, 1320, 1320, 1320], size=17))
    A(bullet("As três versões funcionam em todos os α, com a acurácia praticamente intacta (≤ 1 p.p. abaixo do treino "
             "sem ataque). Em α = 0,4 a ASR média cai para ~33%: em 3 dos 9 casos o LP não passou de 20%."))
    A(bullet("Disfarçar o LP (evasivo, + ruído) **não custa força**: a ASR fica a ±3 p.p. do LP original."))
    A(h2("4.4. LP adaptativo"))
    A(p("Com a defesa ligada (α = 0,5 e 0,3; sementes 42, 46, 47, 48), o atacante passa pelo filtro em 80–95% dos pares "
        "(atacante, camada crítica). O preço é real: sem defesa, a ASR desse ataque cai de 23–72% (LP evasivo) para "
        "7–47%. É o ataque mais forte do conjunto contra defesas por camada: a defesa testada resistiu em só 2 de 6 casos."))
    A(h2("4.5. Ataques de 2026"))
    A(table(["Ataque", "ASR (%)", "Acurácia (%)", "Veredito (ASR ≥ 20%)"], [
        ["LSA", "63,6", "72,6", "funcional"],
        ["LaySelFL", "1,5", "72,5", "não funcional neste cenário"],
    ], [2400, 2000, 2000, 2960], size=18))
    A(p("Semente 42, α = 0,5, sem defesa, 40 rounds (executados no processador). A LSA sobe de 5,7% (round 5) para 63,6%."))
    A(p("A LaySelFL não pega com 2 atacantes de 20. O ataque troca força por disfarce: a penalidade prende cada "
        "atualização perto do servidor e o corte zera o resto, então a contribuição de 2 atacantes se dilui na média dos "
        "18 honestos. O artigo usa 6 atacantes de 19 e começa a atacar na rodada 80, com o modelo já treinado. Pela regra "
        "fixada, nada foi reajustado. Em testes com poucos dados, o sucesso subiu de 17% para 23% em 3 rounds, então o "
        "mecanismo funciona; falta força neste cenário."))
    A(p("Para separar o efeito do cenário, um segundo teste (fixado antes de rodar) mudou só as duas diferenças em "
        "relação ao artigo: **6 atacantes de 20** e ataque **a partir da rodada 16 de 40** (a mesma proporção de 80 em "
        "205), com o modelo já treinado. O sucesso final foi de **3,9%** (acurácia 71,6%): a LaySelFL também não pega "
        "nesse cenário, e o teste parou ali, como fixado. Fica o registro de que a nossa implementação, sem código "
        "dos autores, não reproduz a força relatada no artigo, que usa a ResNet18 e 205 rodadas."))
    A(p("**Atualização (novembro de 2026).** A pista acima — ResNet18 e 205 rodadas — foi confirmada. Ao reproduzir o "
        "cenário **exato** do artigo (ResNet18 com GroupNorm, CIFAR-10, 20 nós com 6 maliciosos, Dirichlet α = 0,9, "
        "205 rodadas, ataque nas rodadas 80 a 200, gatilho BadNet, alvo 4), a **LaySelFL pega de forma decisiva: ASR "
        "final de 82%** (pico de 88% na rodada 200). Ou seja, a LaySelFL **é funcional**; o 1,5% / 3,9% de outubro era "
        "efeito do cenário reduzido (SmallCNN, 40 rodadas), não da implementação. Com isso, a LaySelFL passa para a "
        "coluna *funcional* e abre a fase de defesa (mês 5), resumida na seção 7.1."))
    A(h2("4.6. LP fora do cenário padrão"))
    A(table(["Cenário", "Resultado sem defesa"], [
        ["Fashion-MNIST, SmallCNN, α 0,5, 0,3 e 0,1 (3 sementes)", "ASR < 20% em 16 de 18 casos: não pega"],
        ["ResNet-20 com GroupNorm, CIFAR-10 (3 sementes)", "ASR < 20% em todos os casos: não pega"],
        ["ResNet18 com GroupNorm, CIFAR-10, λ = 1 e 0,5 (semente 42)", "ASR 3,5% e 3,8%, acurácia 62,5%: não pega"],
        ["Código original dos autores, FMNIST, sob FLAME (3 sementes)", "BSR 0,3–0,6% (artigo: 74,9%): não reproduzido"],
    ], [5000, 4360], size=17))
    A(p("O LP, e portanto os ataques derivados dele, só se mostrou forte no cenário padrão (CIFAR-10 com a SmallCNN). "
        "A ResNet18 é uma das redes do artigo do LP, mas o nosso cenário difere em pontos que podem pesar: 40 rounds "
        "contra 200, 20 clientes contra 100 e GroupNorm em vez de BatchNorm. Isso vale como resultado do nosso cenário, "
        "não como conclusão geral sobre o ataque."))

    # 5 ---------------------------------------------------------------
    A(h1("5. Testes Automatizados"))
    A(p("`tests/test_recent_attacks.py` roda em segundos, na CPU, com dados sintéticos. **8 de 8 testes passam.**"))
    A(table(["Teste", "O que verifica"], [
        ["Mimic", "todos os rounds copiam o mesmo cliente honesto"],
        ["DBA", "as 4 faixas formam exatamente o gatilho global"],
        ["Edge-case", "as amostras raras são a cauda da classe de origem"],
        ["Grupos de camadas", "os grupos cobrem o vetor de parâmetros sem sobra nem sobreposição"],
        ["LP (Eq. 4)", "interpolação λ nas camadas críticas e média limpa no resto; λ > 1 zera a média"],
        ["Treino congelado *", "no treino da LSA/LaySelFL, só os tensores escolhidos mudam"],
        ["LSA *", "fora das camadas críticas, o envio é exatamente a média limpa dos atacantes"],
        ["LaySelFL *", "k = total − 2 camadas sensíveis e camadas normais dentro de ±0,05"],
    ], [2600, 6760], size=17))

    # 6 ---------------------------------------------------------------
    A(h1("6. Limitações e Pendências"))
    A(bullet("**Pendências do mês 3 resolvidas, com ressalvas** (seção 4.2): o *Edge-case* com imagens externas passa "
             "dos 20% só pela média (13,8% a 36,7% entre sementes), e o ruído gaussiano com m = 1 tem queda média de "
             "2,4 p.p., perto do limite de 2 p.p. da regra."))
    A(bullet("**Resultados por semente variam muito** nos backdoors fracos (Edge-case de 8% a 37%, DBA de 42% a 92%): "
             "uma semente só não basta para dizer se um ataque funciona."))
    A(bullet("**A LaySelFL só pega no cenário fiel do artigo** (resolvido em novembro, seção 4.5): com a ResNet18-GN e "
             "205 rodadas, a ASR chega a 82%; nos cenários reduzidos de outubro (SmallCNN, 40 rodadas) ficava em "
             "1,5–3,9%. A diferença vinha da rede e do número de rodadas, não da implementação."))
    A(bullet("**LSA e LaySelFL sem código dos autores**: as escolhas da seção 2.1 podem diferir das originais."))
    A(bullet("**Os backdoors por camada dependem do cenário**: fortes no CIFAR-10 com a SmallCNN, fracos no "
             "Fashion-MNIST e nas ResNets com 40 rounds."))
    A(bullet("**Uma semente** na validação dos 2 ataques de 2026; os 13 antigos agora têm 4, e o LP e as variantes têm 3 "
             "(mais 2 já programadas)."))

    # 7 ---------------------------------------------------------------
    A(h1("7. Uso do Conjunto em Outubro e Próximos Passos"))
    A(p("O conjunto já foi usado em escala: a grade de outubro avaliou **8 formas de agregar** (sem defesa, nossa "
        "proposta, LASA, AlignIns, RFLDefense, FLAME, Multi-Krum, FedSurrogate) × 5 valores de α × 4 cenários (LP "
        "original, evasivo, + ruído, sem ataque) × 3 sementes, num total de 480 treinos. Os resultados de detecção "
        "pertencem aos relatórios dos meses 7 a 10, conforme o plano."))
    A(bullet("**Em execução desde 29/10**: +2 sementes (49 e 50) em toda a grade (320 treinos); a defesa dentro do código original "
             "do LP (12 treinos); a LSA contra as 8 formas de agregar, em 3 valores de α e 5 sementes (120 treinos)."))
    A(bullet("**M5**: protótipo do agente de detecção, usando este conjunto como base de validação."))

    A(h2("7.1. Prévia dos resultados de defesa (mês 5, novembro de 2026)"))
    A(p("Com a LaySelFL agora funcional no cenário do artigo (seção 4.5), a fase de defesa começou. Resultados "
        "preliminares no mesmo cenário (ResNet18-GN, CIFAR-10, 6 de 20 atacantes, 205 rodadas, semente 42); o rigor com "
        "três sementes e o comparativo completo com média ± desvio estão em andamento."))
    A(bullet("**Diagnóstico.** A nossa defesa por tensor (corte cirúrgico, `layer_guard_sn`) detecta o atacante "
             "(recall 1,0) nos tensores de GroupNorm e atalho, mas ~70% da massa maliciosa vaza pelos tensores "
             "convolucionais — por isso, sozinha, não contém o LaySelFL (ASR final 76,7%)."))
    A(bullet("**A corroboração entre camadas contém o ataque.** Exigir que ≥ T tensores acusem o cliente antes de "
             "purgá-lo das demais camadas leva a ASR de 76,7% para **7,9%** (`corr`), **2,2%** (`client`, purga o "
             "cliente inteiro) e **8,8%** (`corr-soft`, substituição pelo consenso benigno) — enquanto as seis defesas "
             "SOTA do comparativo ficam em 72–87% e só a LASA [WACV 2025] contém (13,5%, e apenas depois que o ataque "
             "para). As variantes de corroboração seguram **durante toda a janela** do ataque."))
    A(bullet("**Generaliza de arquitetura e de ataque.** Contém também na VGG19-GN (com T baixo: 9–13%) e na "
             "EfficientNet-GN (13–19%), e contra o ataque de gatilho adaptativo A3FL [NeurIPS 2023] no CIFAR-10 (12,6%, "
             "ante 97,7% sem defesa). O overhead de agregação fica em ≤ 6,1% sobre o FedAvg (a LASA chega a 93%)."))
    A(bullet("**Posicionamento honesto.** Os ingredientes têm precedente (detecção por camada, corroboração por "
             "contagem, z robusto de cosseno+norma); o vizinho mais próximo é a FedSurrogate [2026], que combina "
             "criticalidade de camada e substituição cirúrgica e **falha** contra o LaySelFL (71,9%). A contribuição é "
             "a combinação específica do sinal com a corroboração e, sobretudo, **conter os ataques camada-seletivos de "
             "2026 onde a classe inteira — inclusive as defesas cirúrgicas recentes — falha**."))

    # refs ------------------------------------------------------------
    A(h1("Referências Bibliográficas"))
    for ref in [
        "[Zhuang et al. 2024] Zhuang, H. et al. Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
        "[Foroughi et al. 2026] Foroughi, M. H.; Rastegar, S. H.; Sabokrou, M.; Khonsari, A. Exploiting Layer-Specific Vulnerabilities to Backdoor Attack in Federated Learning. IEEE ICC, 2026. arXiv:2602.15161.",
        "[Li et al. 2026] Li, X. et al. Eliminate Distance Differences Induced by Backdoor Attacks: Layer-Selective Training and Clipping to Mask Backdoor Models. CVPR, 2026, p. 13336–13345.",
        "[Tolpegin et al. 2020] Tolpegin, V. et al. Data Poisoning Attacks Against Federated Learning Systems. ESORICS, 2020.",
        "[Gu et al. 2017] Gu, T.; Dolan-Gavitt, B.; Garg, S. BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. arXiv:1708.06733, 2017.",
        "[Xie et al. 2020] Xie, C.; Huang, K.; Chen, P.-Y.; Li, B. DBA: Distributed Backdoor Attacks against Federated Learning. ICLR, 2020.",
        "[Wang et al. 2020] Wang, H. et al. Attack of the Tails: Yes, You Really Can Backdoor Federated Learning. NeurIPS, 2020.",
        "[Blanchard et al. 2017] Blanchard, P. et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS, 2017.",
        "[Xie et al. 2019] Xie, C.; Koyejo, O.; Gupta, I. Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. UAI, 2019.",
        "[Baruch et al. 2019] Baruch, G.; Baruch, M.; Goldberg, Y. A Little Is Enough: Circumventing Defenses for Distributed Learning. NeurIPS, 2019.",
        "[Shejwalkar e Houmansadr 2021] Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. NDSS, 2021.",
        "[Karimireddy et al. 2022] Karimireddy, S. P.; He, L.; Jaggi, M. Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing. ICLR, 2022.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


if __name__ == "__main__":
    m3.content = content
    m3.HERE = HERE
    m3.OUT = HERE / "[HIAAC][M04] Relatório 4.docx"
    m3.main()
