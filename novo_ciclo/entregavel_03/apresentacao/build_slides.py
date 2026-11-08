"""Conteúdo da apresentação do mês 3 (slides + notas do apresentador).

Separado de make_pptx.py (que tem os utilitários de layout) para facilitar a edição do texto.
Cada slide termina com notes(s, "..."): o texto vai para as notas do apresentador no PowerPoint.
"""
from make_pptx import (GRAY, LIGHT, MSO_ANCHOR, MSO_CONNECTOR, MSO_SHAPE, NAVY, ORANGE, BLACK, PP_ALIGN,
                       Inches, Pt, callout, center_picture, content, cross_phase_rows, etree, open_template,
                       ragnar_table_rows, rich, section, table, textbox, title_slide, HERE, OUT)

ARROW = '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>'


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = " ".join(text.split())


def box(s, x, y, w, h, lines, sizes=(18, 15)):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = LIGHT; b.line.color.rgb = NAVY; b.line.width = Pt(1.5)
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        rich(p, line, sizes[0] if i == 0 else sizes[1], NAVY if i == 0 else BLACK, bold=(i == 0))


def arrow(s, x1, y1, x2, y2, dashed=False):
    a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    a.line.color.rgb = ORANGE; a.line.width = Pt(3)
    if dashed:
        a.line.dash_style = 4
    a.line._get_or_add_ln().append(etree.fromstring(ARROW))


REFERENCES = [
    "Bagdasaryan, E. et al. How To Backdoor Federated Learning. AISTATS, 2020.",
    "Bao, W.; Wu, J.; He, J. BOBA: Byzantine-Robust Federated Learning with Label Skewness. AISTATS, 2024.",
    "Baruch, G.; Baruch, M.; Goldberg, Y. A Little Is Enough: Circumventing Defenses for Distributed Learning. NeurIPS, 2019.",
    "Blanchard, P. et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS, 2017.",
    "Gu, T.; Dolan-Gavitt, B.; Garg, S. BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. arXiv:1708.06733, 2017.",
    "Huang, S. et al. Multi-metrics Adaptively Identifies Backdoors in Federated Learning. ICCV, 2023.",
    "Karimireddy, S. P.; He, L.; Jaggi, M. Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing. ICLR, 2022.",
    "Li, L. et al. RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets. AAAI, 2019.",
    "Morais, M. et al. RAGNAR: Lightweight Defense against Poisoning Attacks in Federated LLMs and IoT. CoUrb, 2024.",
    "Ozdayi, M. S.; Kantarcioglu, M.; Gel, Y. R. Defending against Backdoors in Federated Learning with Robust Learning Rate. AAAI, 2021.",
    "Page, E. S. Continuous Inspection Schemes. Biometrika, 1954.",
    "Rousseeuw, P. J.; Van Driessen, K. A Fast Algorithm for the Minimum Covariance Determinant Estimator. Technometrics, 1999.",
    "Shejwalkar, V.; Houmansadr, A. Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. NDSS, 2021.",
    "Tolpegin, V. et al. Data Poisoning Attacks Against Federated Learning Systems. ESORICS, 2020.",
    "Tran, B.; Li, J.; Madry, A. Spectral Signatures in Backdoor Attacks. NeurIPS, 2018.",
    "Vovk, V.; Gammerman, A.; Shafer, G. Algorithmic Learning in a Random World. Springer, 2005.",
    "Wald, A. Sequential Tests of Statistical Hypotheses. Annals of Mathematical Statistics, 1945.",
    "Wang, H. et al. Attack of the Tails: Yes, You Really Can Backdoor Federated Learning. NeurIPS, 2020.",
    "Weiszfeld, E. Sur le point pour lequel la somme des distances de n points donnés est minimum. Tôhoku Math. J., 1937.",
    "Xie, C.; Huang, K.; Chen, P.-Y.; Li, B. DBA: Distributed Backdoor Attacks against Federated Learning. ICLR, 2020.",
    "Xie, C.; Koyejo, O.; Gupta, I. Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. UAI, 2019.",
    "Xu, J. et al. SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering. ICDCS, 2022.",
    "Zhuang, H. et al. Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
]


def build():
    prs, _ = open_template()
    fig = HERE / "figuras"

    s = title_slide(prs, "Agente para Detecção de Ataques em Aprendizado Federado",
                    "Evoluindo o RAGNAR: ataques mais recentes e detecção sem treino offline",
                    ["**Aluno:** Marcos Gilliard Oliveira Morais", "**Orientador:** Leandro A. Villas",
                     "**H.IAAC — Meta 3 · Setembro de 2026**"])
    notes(s, """Apresentação do terceiro mês da IC. Além do avanço do cronograma, a mensagem principal é que estamos
    evoluindo o RAGNAR, nosso método base, em duas frentes: avaliá-lo contra ataques mais recentes do que os do
    artigo original e torná-lo independente do treino offline prévio, que hoje exige simular de antemão os ataques
    que o classificador deve reconhecer.""")

    # ---------------- contexto
    s = section(prs, "Contexto", "O problema e o plano de 12 meses")
    notes(s, "Primeiro, o problema que o projeto ataca e onde estamos no cronograma.")

    s = content(prs, "O problema")
    textbox(s, 0.8, 2.6, 10.6, 7, [
        "No Aprendizado Federado (FL) o servidor **não vê os dados**, só as atualizações dos clientes.",
        "Clientes maliciosos podem:",
        (1, "**degradar** o modelo (ruído, IPM, LIE, MinMax…)"),
        (1, "**inserir comportamento oculto** (backdoor)"),
        (1, "**envenenar os dados** (label flip)"),
        "Detectar exige inferir intenção a partir do que é observável."], size=22)
    callout(s, 12.2, 2.6, 7.0, 3.0, "Objetivo do projeto",
            ["Um **agente de detecção** que combine múltiplos sinais (atualizações e histórico dos clientes) e **fundamente suas decisões**."])
    callout(s, 12.2, 6.0, 7.0, 2.6, "Caso base",
            ["**RAGNAR** (Morais et al., 2024): filtro leve com features geométricas + XGBoost."])
    notes(s, """No FL o servidor só recebe atualizações de modelo, nunca os dados. Isso abre espaço para clientes
    maliciosos que degradam o modelo, inserem backdoors ou envenenam os próprios dados. O objetivo do plano de
    bolsa é um agente que combine vários sinais observáveis e explique suas decisões. O ponto de partida é o
    RAGNAR, nosso método publicado no CoUrb 2024.""")

    s = content(prs, "Plano de 12 meses: onde estamos")
    table(s, 1.2, 2.6, 17.6, [
        ["Mês", "Atividade", "Entregável", "Situação"],
        ["1 (jul)", "Revisão de ataques e defesas", "Revisão e taxonomia", "concluído"],
        ["2 (ago)", "Ambiente e cenário de avaliação", "Ambiente validado + cenário", "concluído"],
        ["**3 (set)**", "**Sinais observáveis**", "**Especificação dos sinais e da lógica**", "^^**em andamento**^^"],
        ["4 (out)", "Ataques controlados", "Ataques injetáveis", "adiantado"],
        ["5 (nov)", "Agente (versão inicial)", "Protótipo funcional", "—"],
        ["6 (dez)", "Integração à agregação", "Pipeline + relatório parcial", "—"],
        ["7–10", "Detecção, falsos positivos, robustez", "Relatórios de avaliação", "—"],
        ["11–12", "Artigo e relatório final", "Manuscrito + repositório", "—"]],
        col_w=[2.0, 6.2, 6.6, 2.8], size=18, row_h=0.62, highlight=3)
    notes(s, """Cronograma oficial do plano de bolsa. Meses 1 e 2 estão concluídos. Setembro é o mês 3, cujo
    entregável é a especificação dos sinais observáveis e da lógica de detecção. O mês 4, ataques controlados,
    já está adiantado: os 9 ataques estão implementados e validados.""")

    s = content(prs, "Objetivo deste ciclo: evoluir o RAGNAR")
    callout(s, 0.8, 2.6, 8.9, 4.3, "Frente 1 · Ataques mais recentes", [
        "O artigo original (CoUrb 2024) avaliou ataques simples de pesos: ruído, zeros, inversão, embaralhamento.",
        "Agora: ataques da literatura 2017–2021 (IPM, LIE, MinMax, backdoor PGD) e, em seguida, os de 2020–2024 da revisão (LP/LF, Mimic, DBA)."], size=18)
    callout(s, 10.3, 2.6, 8.9, 4.3, "Frente 2 · Sem treino offline", [
        "Hoje o classificador é treinado antes, com ataques simulados de antemão: só reconhece o que viu.",
        "Meta: detectar com sinais normalizados no próprio round, sem classificador pré-treinado."], head=ORANGE, size=18)
    textbox(s, 0.8, 7.4, 18.4, 1.6, ["Os experimentos deste mês diagnosticam onde o método convencional falha e dão a base para as duas frentes."], size=20, bullets=False)
    notes(s, """Este é o fio condutor da apresentação. Não estamos só executando o cronograma: estamos evoluindo o
    método convencional em duas frentes. A primeira é a avaliação: o artigo original testou o RAGNAR com ataques
    simples de pesos, e estamos levando-o para ataques da literatura recente, primeiro os de 2017 a 2021 e depois os
    de 2020 a 2024 mapeados na revisão. A segunda é a arquitetura: hoje o classificador depende de um treino offline
    com ataques simulados de antemão; a meta é detectar sem esse treino, com sinais normalizados no próprio round. Os
    experimentos deste mês diagnosticam as falhas do método convencional e sustentam essas duas frentes.""")

    # ---------------- o que foi feito
    s = section(prs, "O que foi feito", "Revisão, ambiente experimental e auditoria")
    notes(s, "O que foi construído até aqui: a revisão, o ambiente de experimentos e a auditoria desse ambiente.")

    s = content(prs, "Meses 1 e 2: revisão e ambiente")
    textbox(s, 0.8, 2.5, 8.6, 0.6, ["**Revisão sistemática (M1)**"], size=22, bullets=False, color=NAVY)
    textbox(s, 0.8, 3.2, 8.6, 6, [
        "16 artigos: BOBA, Blades, FedSecurity, FLGuard, Multi-metrics, XI-DDaBA, RAGNAR…",
        "Taxonomia por objetivo, canal (dados/modelo) e furtividade (camadas críticas)."], size=20)
    textbox(s, 10.2, 2.5, 9.2, 0.6, ["**fl-robustness-lab (M2)**"], size=22, bullets=False, color=NAVY)
    textbox(s, 10.2, 3.2, 9.2, 6.5, [
        "FL simulado em PyTorch: CIFAR-10, 20 clientes, não-IID (Dirichlet).",
        "**9 ataques**: ruído, sign flip, IPM, LIE, MinMax, MinSum, label flip, backdoor pixel e PGD.",
        "**9 defesas**: FedAvg, Mediana, Trimmed Mean, (Multi-)Krum, FoolsGold, Multi-métrica, BOBA, RAGNAR.",
        "Métricas: acurácia (MA), sucesso do backdoor (BA), precisão e recall de detecção."], size=20)
    notes(s, """No mês 1 revisamos 16 artigos e consolidamos uma taxonomia de ataques e defesas. No mês 2 montamos
    o fl-robustness-lab, um ambiente de FL simulado em PyTorch com CIFAR-10, dados não-IID, nove ataques e nove
    defesas plugáveis, incluindo BOBA e RAGNAR. As métricas são acurácia principal, sucesso do backdoor e
    precisão e recall da detecção de clientes maliciosos.""")

    s = content(prs, "Os ataques avaliados e suas origens")
    table(s, 0.6, 2.4, 18.8, [
        ["Ataque", "Tipo", "Mecanismo", "Referência"],
        ["Ruído gaussiano", "modelo", "substitui a atualização por ruído N(0, σ²)", "Blanchard et al., NeurIPS 2017"],
        ["Sign flip / IPM", "modelo", "envia −ε × média das atualizações honestas", "Li et al., AAAI 2019 · Xie et al., UAI 2019"],
        ["LIE", "modelo", "média + z × desvio-padrão dos honestos", "Baruch et al., NeurIPS 2019"],
        ["MinMax / MinSum", "modelo", "perturbação máxima que ainda parece honesta", "Shejwalkar & Houmansadr, NDSS 2021"],
        ["Label flip", "dados", "troca os rótulos de uma classe (0 → 2)", "Tolpegin et al., ESORICS 2020"],
        ["Backdoor (pixel)", "dados", "gatilho 4×4 + rótulo alvo em 50% das amostras", "Gu et al., 2017 · Bagdasaryan et al., AISTATS 2020"],
        ["Backdoor (PGD)", "dados + modelo", "backdoor com treino projetado e escala (norma limitada)", "Wang et al., NeurIPS 2020"]],
        col_w=[3.3, 2.4, 6.9, 6.2], size=15, row_h=0.72)
    textbox(s, 0.6, 8.4, 18.8, 1, ["Ataques de modelo são “oniscientes”: o atacante vê as atualizações honestas do round (modelagem do BOBA/Blades). O PGD é uma simplificação do Edge-case PGD, sem a mineração de amostras raras."],
            size=14, bullets=False, color=GRAY)
    notes(s, """Estes são os ataques usados em todos os experimentos, com a referência de onde cada um vem. Os
    quatro primeiros são de envenenamento de modelo e seguem a modelagem onisciente do BOBA e do Blades: o
    atacante observa as atualizações honestas do round e sintetiza a sua. IPM com epsilon 1 e sign flip com
    escala 1 produzem o mesmo vetor, por isso só IPM entra na avaliação. Label flip e backdoor pixel atuam nos
    dados. O backdoor PGD combina o gatilho com treino projetado e escala, para ficar com norma igual à de um
    cliente honesto; é uma simplificação do Edge-case PGD de Wang et al., sem a mineração de amostras raras.""")

    s = content(prs, "Auditoria do ambiente antes de medir", "5 problemas encontrados e corrigidos, todos com testes unitários")
    table(s, 0.8, 4.1, 18.4, [
        ["Problema", "Impacto", "Correção"],
        ["backdoor_pgd escalava e voltava à própria norma", "idêntico ao backdoor simples", "PGD real: treino projetado + escala"],
        ["BOBA: vértices eram gradientes (sinal/escala opostos)", "descartava clientes honestos", "deltas de referência por classe"],
        ["RAGNAR treinado com o ataque avaliado", "resultado otimista", "treino sem a família avaliada"],
        ["Parâmetros eram views do vetor global", "em CPU o treino sobrescrevia o global", "cópia explícita"],
        ["Braço “ataque visto” sorteava o treino", "o ataque podia nunca ser visto", "treino só com o ataque"]],
        col_w=[7.4, 5.6, 5.4], size=17, row_h=0.72)
    notes(s, """Antes de medir, revisamos o código e corrigimos cinco problemas. O ataque PGD era, na prática,
    igual ao backdoor simples. O BOBA usava referências com sinal e escala errados. O RAGNAR era treinado com o
    próprio ataque avaliado, o que inflava o resultado. Em CPU, o treino local sobrescrevia o modelo global. E o
    braço de ataque visto podia nunca ter visto o ataque. Cada correção tem um teste unitário.""")

    # ---------------- o RAGNAR funciona?
    s = section(prs, "Diagnóstico do RAGNAR convencional", "Ataques da literatura e dependência do treino offline")
    notes(s, """Primeiro, o diagnóstico: como o RAGNAR convencional se comporta diante de ataques da literatura, e o
    quanto ele depende do treino offline.""")

    s = content(prs, "RAGNAR original: como funciona")
    xs = [0.9, 5.7, 10.5, 15.3]
    labels = [["Pesos enviados", "pelo cliente"], ["54 features", "L1, L2, L3 dos pesos e sua", "variação, por camada"],
              ["MinMax + XGBoost", "exclui se P > 0,5"], ["Agrega os demais", "(a partir do round 2)"]]
    for k, (x, lines) in enumerate(zip(xs, labels)):
        box(s, x, 3.0, 3.9, 1.9, lines)
        if k < 3:
            arrow(s, x + 3.9, 3.95, xs[k + 1], 3.95)
    box(s, 8.9, 5.9, 7.1, 1.3, ["Treino offline", "simulações prévias completas; depois congelado"], sizes=(17, 15))
    arrow(s, 12.45, 5.9, 12.45, 4.9, dashed=True)
    textbox(s, 0.9, 7.7, 18.2, 1.2, ["Configuração do artigo (CoUrb 2024): **54 features** (9 tensores × 6), **limiar 0,5**, Flower + CNN Keras. É a referência usada em todo o projeto."], size=18, bullets=False)
    notes(s, """O RAGNAR calcula, para cada camada do modelo enviado pelo cliente, as normas L1, L2 e L3 dos pesos e
    quanto cada uma mudou em relação ao modelo global anterior. Essas features passam por um MinMaxScaler e um
    XGBoost; quem tem probabilidade acima do limiar é excluído, e o resto é agregado. O filtro começa no round 2.
    O classificador é treinado antes, com simulações completas, e fica congelado. Usamos como referência a
    configuração do artigo publicado no CoUrb 2024: 54 features, que vêm de 9 tensores com 6 medidas cada, e
    limiar 0,5.""")

    s = content(prs, "Reimplementação v1 do laboratório vs. original", "A primeira rodada avaliou uma versão diferente do método")
    table(s, 2.0, 4.2, 16.0, [
        ["", "RAGNAR original", "Reimplementação v1"],
        ["Features", "54, por camada (pesos)", "4, no vetor inteiro (delta vs. média)"],
        ["Normalização", "MinMax", "nenhuma"],
        ["Limiar", "0,5", "0,5"],
        ["Treino", "simulações completas", "5 rounds a partir do modelo inicial"],
        ["Início", "round 2", "round 1"]], col_w=[3.4, 6.2, 6.4], size=19, row_h=0.62)
    textbox(s, 2.0, 8.4, 16, 1, ["Os resultados dos Experimentos 1 a 3 valem para a **v1**. A versão fiel está implementada no laboratório e no framework do CoUrb."], size=17, bullets=False)
    notes(s, """Um aviso importante: a primeira rodada de experimentos usou a reimplementação v1 do laboratório, que
    difere do RAGNAR original em quase tudo: features, normalização, limiar, forma de treino e round de início.
    Os resultados dos Experimentos 1 a 3 valem para essa v1. A versão fiel já está implementada, tanto no
    laboratório quanto no framework do CoUrb.""")

    s = content(prs, "Experimento 1: v1 contra todos os ataques")
    textbox(s, 0.8, 2.5, 18.4, 1, ["8 ataques, semente 42: 20 clientes, 20% maliciosos (10% nos backdoors), α = 0,5, 40 rounds."], size=21)
    table(s, 2.5, 3.6, 15.0, [
        ["Braço", "Pergunta"],
        ["FedAvg (sem defesa)", "Quanto estrago o ataque faz?"],
        ["RAGNAR v1, ataque **não visto**", "Generaliza para um ataque novo?"],
        ["Sem ataque", "Teto de acurácia"]], col_w=[7.0, 8.0], size=20, row_h=0.65)
    callout(s, 2.5, 7.0, 15.0, 1.9, "Por que “não visto” importa", ["Na prática ninguém sabe de antemão qual ataque vai sofrer."])
    notes(s, """Protocolo do Experimento 1: oito ataques, mesma semente e mesmos dados para todos os braços. Comparamos
    o FedAvg sem defesa, que mede o estrago do ataque, com o RAGNAR v1 treinado sem a família do ataque avaliado,
    e uma execução sem ataque como teto de acurácia. O caso não visto é o que importa, porque na prática ninguém
    sabe qual ataque vai sofrer.""")

    s = content(prs, "Resultados: RAGNAR v1 vs. sem defesa")
    center_picture(s, fig / "ragnar_vs_fedavg.png", 2.4, 18.4, 8.0)
    notes(s, """À esquerda, a acurácia final do modelo global com e sem RAGNAR; a linha tracejada é o teto sem ataque.
    À direita, a detecção do RAGNAR: em azul os atacantes pegos, em laranja os honestos excluídos por engano.
    Nos backdoors o RAGNAR exclui muitos honestos e a acurácia cai; nos ataques de modelo ele quase não detecta
    ninguém.""")

    s = content(prs, "Resultados em números")
    rows, clean = ragnar_table_rows()
    table(s, 1.2, 2.5, 17.6, rows, col_w=[3.6, 2.2, 2.6, 2.2, 2.6, 2.0, 2.4], size=18, row_h=0.6, num_cols=(1, 2, 3, 4, 5, 6))
    textbox(s, 1.2, 2.5 + 0.6 * len(rows) + 0.3, 17.6, 1.2, [
        f"Sem ataque (FedAvg): MA = {clean:.1f}%".replace(".", ",") if clean is not None else "",
        "MA = acurácia (%) · BA = sucesso do backdoor (%, menor é melhor) · Falsos pos. = fração de honestos excluídos · uma semente"],
        size=14, bullets=False, color=GRAY)
    notes(s, """Os mesmos resultados em tabela. Vale olhar as últimas duas linhas: nos backdoors, o RAGNAR v1 derruba
    a acurácia de cerca de 72% para 57% e o backdoor fica mais forte, porque cerca de metade dos honestos é
    excluída e os atacantes ganham peso. É uma semente só, então diferenças de 1 a 2 pontos podem ser ruído.""")

    s = content(prs, "O que o Experimento 1 mostra")
    callout(s, 0.8, 2.5, 18.4, 4.6, "RAGNAR v1, ataque não visto: não protege, e às vezes piora", [
        "**Backdoors**: exclui ~50% dos **honestos** ⇒ MA 72% → 57% e o backdoor fica **mais** forte (BA 75% → 94%).",
        "**Ataques de modelo** (IPM, LIE, MinMax, MinSum): recall 0.",
        "**Label flip**: nenhuma detecção.",
        "**Ruído gaussiano**: todos colapsam para 10%, inclusive com o RAGNAR."], head=ORANGE, size=20)
    textbox(s, 0.8, 7.6, 18.4, 1.2, ["Próxima pergunta: **por quê?** Falta informação nas features, ou o classificador aprende a regra errada?"], size=21, bullets=False)
    notes(s, """Resumo do Experimento 1: com ataque não visto, a v1 não protege e às vezes piora. A pergunta seguinte
    é por quê: as features não carregam informação suficiente, ou o classificador aprende uma regra errada? A
    análise de sinais responde a isso.""")

    s = content(prs, "Classificador pré-treinado do artigo", "Modelo do CoUrb 2024 (54 features)")
    table(s, 0.8, 4.1, 18.4, [
        ["Ambiente", "Features fora da faixa do MinMax", "Honestos marcados como atacantes", "Leitura"],
        ["framework do CoUrb (Flower + Keras)", "**0%**", "—", "compatível"],
        ["fl-robustness-lab (mesma CNN, Adam)", "**0%**", "0–5% (rounds 1–3)", "compatível"]],
        col_w=[6.0, 4.6, 4.6, 3.2], size=17, row_h=0.7, num_cols=(1, 2))
    callout(s, 0.8, 6.6, 18.4, 2.4, "Leitura", [
        "O modelo do artigo pode ser avaliado **diretamente**, nos dois ambientes, sem retreino.",
        "A questão em aberto é se ele reconhece ataques que **não viu** no treino: ele foi treinado com os ataques do artigo."], head=ORANGE, size=18)
    notes(s, """Testamos o classificador pré-treinado do artigo publicado no CoUrb, com 54 features. No framework do
    próprio CoUrb, nenhuma feature cai fora da faixa do MinMax. Levando o mesmo modelo para o laboratório, com a
    mesma CNN e o mesmo otimizador, também nenhuma feature cai fora da faixa, e nos três primeiros rounds só de 0
    a 5% dos clientes honestos são marcados como atacantes. Ou seja, o modelo do artigo pode ser avaliado
    diretamente nos dois ambientes. A questão em aberto é outra: ele foi treinado com os ataques do artigo,
    e precisamos medir se reconhece os ataques mais recentes, que nunca viu. Este teste cobriu só clientes
    honestos e os três primeiros rounds.""")

    s = content(prs, "Bug no cliente do CoUrb-2024", "cliente_principal.py, linha 165 (e atak_cliente.py, linha 163)")
    code = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.2), Inches(18.4), Inches(1.1))
    code.fill.solid(); code.fill.fore_color.rgb = LIGHT; code.line.fill.background()
    tf = code.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Inches(0.3)
    r = tf.paragraphs[0].add_run()
    r.text = "elif modo=='EMBARALHA' or 'NORMAL' and server_round >= self.round_inicio and self.cid <= self.per_cents_atacantes:"
    r.font.name, r.font.size, r.font.color.rgb = "Courier New", Pt(20), NAVY
    textbox(s, 0.8, 5.7, 18.4, 3.6, [
        "A string **'NORMAL'** sozinha vale como verdadeiro: a condição é **sempre verdadeira**.",
        "Em qualquer modo que não seja ALTERNA_INICIO ou ATACANTES, **todos os clientes** (inclusive os honestos) executam EMBARALHA em todos os rounds, com rótulo de atacante.",
        "Os ramos INVERTE_*, ZEROS, RUIDO_GAUSSIANO e o cliente honesto nunca rodam. Se os dados de treino do classificador vieram desse cliente, podem estar contaminados.",
        "Na nossa cópia (courb_eval/) a lógica foi corrigida."], size=19)
    notes(s, """Ao estudar o código do CoUrb encontramos este bug. Em Python, a string NORMAL sozinha é avaliada como
    verdadeira, então essa condição é sempre verdadeira. Na prática, em quase todos os modos de ataque, todos os
    clientes, inclusive os honestos, embaralham os pesos em todos os rounds e são rotulados como atacantes. Os
    outros ataques e o cliente honesto nunca rodam. Se os dados usados para treinar o classificador vieram desse
    cliente, eles podem estar contaminados. Corrigimos isso na nossa cópia do framework.""")

    s = content(prs, "Quanto o treino offline importa?", "No framework do CoUrb, com os 8 ataques")
    textbox(s, 0.8, 4.1, 10.2, 5.5, [
        "Framework do artigo: Flower 1.7, CNN Keras, Adam, 1 época, partição Dirichlet (α = 0,5, 1000 amostras/cliente), 20 clientes.",
        "RAGNAR como no servidor do CoUrb: 54 features, MinMax, limiar 0,5, a partir do round 2.",
        "Braços: FedAvg · offline completo (pré-treinado) · treino curto de 5 rounds (ataque visto e não visto) · próximo: **sem treino** (Frente 2).",
        "Testes de validação: todos os caminhos rodam; ~10 s por round."], size=19)
    callout(s, 11.6, 4.1, 7.6, 3.2, "Situação", [
        "Código implementado e validado.",
        "Rodada completa (33 execuções): **a iniciar**; resultados entram na próxima versão."], head=ORANGE, size=18)
    notes(s, """Para comparar de forma justa o classificador pré-treinado com o treinado em 5 rounds, portamos os oito
    ataques do laboratório para o framework do artigo: Flower com a CNN em Keras, o mesmo otimizador, a mesma
    partição de dados e o RAGNAR exatamente como no servidor do CoUrb. Os testes de validação passaram em todos
    os caminhos. A rodada completa ainda precisa rodar, e os resultados entram na próxima versão destes slides.""")

    # ---------------- por que falha
    s = section(prs, "Por que falha?", "Análise de discriminabilidade dos sinais")
    notes(s, "Para entender a falha da v1, medimos quanto cada sinal observável separa maliciosos de honestos.")

    s = content(prs, "Experimento 2: quanto cada sinal separa?")
    textbox(s, 0.8, 2.5, 10.5, 0.7, ["19 sinais por cliente e por round, em 6 famílias:"], size=21, bullets=False)
    textbox(s, 0.8, 3.3, 10.5, 6, [
        "geometria do RAGNAR v1 (L1, L2, L3, cosseno)", "geometria robusta (mediana, Mahalanobis, Krum)",
        "conluio (similaridade entre clientes)", "camada crítica (fc1)",
        "referência no servidor (holdout, FLTrust)", "histórico entre rounds"], size=20)
    callout(s, 11.8, 2.5, 7.4, 5.0, "Protocolo", [
        "**Agregação oráculo**: mede detectabilidade, não estrago.",
        "**AUC dentro de cada round**, média entre rounds.",
        "0,5 = não separa · 1,0 = separa perfeitamente."], size=18)
    notes(s, """Registramos 19 sinais por cliente em cada round, em seis famílias. Para isolar a detecção do estrago,
    o modelo global é atualizado só com os clientes honestos. Medimos a AUC dentro de cada round, que é o tipo de
    decisão que o servidor precisa tomar, e tiramos a média entre rounds. AUC 0,5 significa que o sinal não
    separa; 1,0 significa separação perfeita.""")

    s = content(prs, "Mapa de discriminabilidade")
    center_picture(s, fig / "heatmap_signals.png", 2.3, 18.0, 8.4)
    notes(s, """Cada célula é a AUC de um sinal contra um ataque, lida na direção que declaramos antes de ver os
    dados: a seta ao lado do nome diz se esperamos o malicioso com valor maior ou menor. Azul quer dizer que o
    sinal detecta; vermelho, que o ataque inverte o sinal. Corrigimos a versão anterior, que escolhia a direção
    por ataque depois de ver o resultado e por isso superestimava os sinais. As duas últimas colunas são a média
    e o pior caso.""")

    s = content(prs, "Achados da análise de sinais")
    textbox(s, 0.8, 2.5, 18.4, 5, [
        "**Nenhum sinal cobre todos os ataques**: no pior caso, só 2 sinais passam de 0,6 (conluio 0,60; Mahalanobis 0,64).",
        "**LIE/MinMax/MinSum se escondem no centro**: o malicioso fica **mais** alinhado à mediana que um honesto (cosseno: AUC 0,00 na direção esperada); o IPM faz o oposto. Lido como desvio **bilateral**, o cosseno sobe para 0,87.",
        "**Normas falham no PGD** (0,46–0,58); o holdout pega (1,00). **Label flip**: nenhum sinal passa de 0,76."], size=19)
    callout(s, 0.8, 7.4, 18.4, 1.7, "Consequência", ["A informação existe, mas **em sentidos diferentes por ataque**: é preciso combinar famílias e ler cada sinal no modo certo (uni ou bilateral)."], size=18)
    notes(s, """Três achados. Primeiro, nenhum sinal sozinho cobre todos os ataques. Segundo, os ataques de modelo
    como LIE e MinMax constroem o vetor para ficar no centro da nuvem honesta, e acabam mais alinhados que um
    honesto típico; o IPM faz o oposto. Um cosseno lido numa direção só falha em metade deles; lido como desvio
    em qualquer sentido, funciona. Terceiro, normas falham no PGD, que limita a norma, e o label flip segue sem
    sinal confiável. A consequência é que a informação existe, mas em sentidos diferentes: o detector precisa
    combinar famílias e ler cada sinal no modo certo.""")

    s = content(prs, "A regra muda ao longo do treino")
    textbox(s, 0.8, 2.6, 9.6, 0.8, ["Cosseno com a média (feature da v1), backdoor pixel:"], size=20, bullets=False)
    table(s, 0.8, 3.5, 9.6, [["Fase", "AUC", "Malicioso tem cosseno"],
                             ["rounds 2–7", "**0,21**", "menor"], ["rounds 8–15", "**0,87**", "maior"]],
          col_w=[3.0, 2.0, 4.6], size=19, row_h=0.62)
    textbox(s, 0.8, 5.6, 9.6, 1, ["a média (0,55) esconde um sinal forte que **inverte de sentido**"], size=16, bullets=False, color=GRAY)
    callout(s, 11.0, 2.6, 8.2, 5.8, "Por que isso quebra a v1", [
        "A v1 treinava só com **5 rounds** a partir do modelo inicial.",
        "Aprende a regra da fase inicial e a aplica depois, quando ela já se inverteu.",
        "Features **absolutas**: a escala muda com a fase do treino."], head=ORANGE, size=18)
    notes(s, """Este é o achado mais importante. No backdoor pixel, o cosseno com a média separa bem, mas inverte de
    sentido: no começo do treino o malicioso tem cosseno menor, e depois tem cosseno maior. A média esconde isso.
    Como a v1 treinava o classificador só nos primeiros rounds, ela aprendia a regra do começo e a aplicava depois,
    quando a regra já tinha se invertido. É um efeito de escala: as features absolutas mudam de valor conforme o modelo treina.""")

    s = content(prs, "Experimento 3: treinado cedo, testado tarde")
    table(s, 0.8, 2.6, 10.0, cross_phase_rows(), col_w=[2.6, 2.5, 1.5, 1.6, 1.8], size=16, row_h=0.5, num_cols=(2, 3, 4))
    textbox(s, 0.8, 6.9, 10.0, 1, ["Treino: rounds 2–5 · teste: ≥ 8 · ataque não visto · limiar 0,5 · média de 8 ataques"], size=14, bullets=False, color=GRAY)
    textbox(s, 11.4, 2.6, 7.8, 6.5, [
        "**Features da v1**, sem normalização: AUC ^^**0,48**^^ (acaso) e ^^**44%**^^ de honestos excluídos.",
        "**z-score robusto no round**: AUC 0,68 e 3% de falsos positivos, com as mesmas 4 features.",
        "**Sinais adicionais**: AUC 0,91.",
        "Label flip continua sem detecção."], size=19)
    notes(s, """Para testar essa explicação, treinamos um classificador igual ao do RAGNAR nos rounds iniciais e
    testamos nos rounds finais, com ataque não visto. Com as features da v1 sem normalização, a AUC fica em 0,48,
    nível do acaso, e 44% dos honestos são excluídos, o que reproduz o Experimento 1. Normalizando as mesmas
    quatro features dentro do round com um z-score robusto, a AUC sobe para 0,68 e os falsos positivos caem para
    3%. Com os sinais adicionais, a AUC chega a 0,91. Label flip continua sem detecção.""")

    # ---------------- frente 1: ataques mais recentes
    s = section(prs, "Frente 1: ataques mais recentes", "Levar o RAGNAR além dos ataques do artigo original")
    notes(s, "A primeira frente de evolução: a cobertura de ataques.")

    s = content(prs, "Evolução da cobertura de ataques")
    stages = [("CoUrb 2024", ["artigo original", "ruído, zeros, inversão,", "embaralhamento de pesos"], "avaliado no artigo"),
              ("Este ciclo", ["literatura 2017–2021", "IPM, LIE, MinMax/MinSum,", "label flip, backdoor, PGD"], "8 ataques avaliados"),
              ("Mês 4", ["revisão 2020–2024", "LP/LF, Mimic, DBA,", "Edge-case PGD completo"], "a implementar")]
    for k, (head, lines, status) in enumerate(stages):
        x = 0.9 + k * 6.3
        box(s, x, 3.0, 5.2, 2.6, [head] + lines, sizes=(22, 17))
        tb = textbox(s, x, 5.8, 5.2, 0.6, [f"**{status}**"], size=17, bullets=False, align=PP_ALIGN.CENTER,
                     color=NAVY if k < 2 else ORANGE)
        if k < 2:
            arrow(s, x + 5.2, 4.3, x + 6.3, 4.3)
    textbox(s, 0.9, 7.2, 18.2, 1.8, [
        "Cada etapa é mais furtiva: de pesos claramente anômalos, para perturbações dentro da variância honesta, para ataques que só mexem nas camadas críticas.",
        "O Experimento 1 já mostra que a passagem da 1ª para a 2ª etapa quebra o método convencional."], size=18)
    notes(s, """Esta é a evolução da cobertura. O artigo original testou ataques simples de pesos, que produzem
    atualizações claramente anômalas. Neste ciclo passamos a oito ataques da literatura de 2017 a 2021, que
    ficam dentro da variância dos clientes honestos ou atuam nos dados. O próximo passo, no mês 4, são os ataques
    mais recentes da revisão, de 2020 a 2024, como o LP/LF, que só modifica as camadas críticas. Cada etapa é
    mais furtiva, e o Experimento 1 já mostra que a segunda etapa quebra o método convencional.""")

    s = content(prs, "Ataques recentes da revisão: implementados", "src/attacks/recent.py · FedAvg, 40 rounds, sem defesa")
    table(s, 0.6, 3.9, 18.8, [
        ["Ataque", "Referência", "Mecanismo", "Efeito (sem defesa)"],
        ["**LP** (Layer-wise Poisoning)", "Zhuang et al., ICLR 2024", "envenena só as camadas críticas (achou fc1 e conv3, como o artigo)", "backdoor **50%** (base 2,5%), MA −0,2 p.p."],
        ["**LF** (Layer-wise Flipping)", "Zhuang et al., ICLR 2024", "inverte a direção só das camadas mais críticas", "MA **−4,0 p.p.**"],
        ["**DBA** (backdoor distribuído)", "Xie et al., ICLR 2020", "cada atacante treina 1/4 do gatilho", "backdoor **42%** (base 2,5%)"],
        ["**Mimic**", "Karimireddy et al., ICLR 2022", "copia sempre a atualização real de um honesto", "MA **−2,9 p.p.**"],
        ["**Edge-case PGD**", "Wang et al., NeurIPS 2020", "aviões raros (escuros) → caminhão, sem gatilho, norma limitada", "^^fraco: 8% (base 2%)^^"]],
        col_w=[4.2, 4.0, 6.6, 4.0], size=14, row_h=0.8)
    textbox(s, 0.6, 8.9, 18.8, 0.8, ["Edge-case: 3 versões documentadas; com amostras do próprio CIFAR-10 e CNN pequena, 2 atacantes em 20 não sustentam a exceção. Versão com imagens externas: mês 4."], size=14, bullets=False, color=GRAY)
    notes(s, """Implementamos os cinco ataques recentes da revisão que faltavam. O LP envenena só as camadas críticas e,
    como no artigo, a análise de substituição apontou a fc1 e a última convolução; ele chega a 50% de sucesso do
    backdoor sem mexer na acurácia. O LF é a versão não direcionada e tira 4 pontos de acurácia. O DBA divide o
    gatilho entre quatro atacantes e chega a 42%. O Mimic copia um cliente honesto real e tira 3 pontos. O
    Edge-case, que usa amostras raras em vez de gatilho, ficou fraco nesta configuração: 8% contra 2% de base,
    depois de três versões documentadas. Com amostras do próprio CIFAR-10 e uma CNN pequena, dois atacantes não
    conseguem sustentar a exceção; a versão com imagens externas, como no artigo, fica para o mês 4.""")

    s = content(prs, "Ataques recentes: o que os sinais mostram", "Teste fora da amostra (regra congelada antes)")
    textbox(s, 0.8, 4.0, 10.4, 5.5, [
        "**Cada ataque escapa de uma família diferente:**",
        "Mimic: escapa dos cossenos e do holdout; pego por norma, conluio e espectral (0,97).",
        "LF: escapa das normas e **inverte o conluio** (0,07); pego pelos cossenos (0,92–0,98).",
        "LP e DBA: invisíveis no holdout; pegos por normas e dispersão (0,94–0,97).",
        "Edge-case: escapa das normas; pego pelo holdout (1,00)."], size=17)
    table(s, 11.6, 4.1, 7.8, [["Regra (5 ataques novos)", "AUC", "Recall", "FP"],
                             ["Proposta", "0,88", "0,72", "15%"], ["Proposta + conluio", "0,93", "0,85", "15%"],
                             ["RAGNAR-4", "0,70", "0,34", "9%"]],
          col_w=[3.6, 1.3, 1.5, 1.4], size=15, row_h=0.6, num_cols=(1, 2, 3))
    callout(s, 11.6, 6.9, 7.8, 2.4, "Leitura", [
        "A vantagem da proposta se mantém fora da amostra, mas cai (AUC 0,95 → 0,88) e os falsos positivos sobem: o limiar precisa de calibração (mês 9)."], head=ORANGE, size=15)
    notes(s, """A regra de detecção foi congelada antes destes ataques existirem, então eles são um teste fora da
    amostra. Primeiro achado: cada ataque escapa de uma família diferente de sinais, o que confirma que o
    detector precisa combinar famílias. O LF chega a inverter o sinal de conluio. Segundo: a proposta continua
    bem acima da regra com as features do RAGNAR, mas cai em relação aos ataques usados no desenho, de AUC 0,95
    para 0,88, e os falsos positivos sobem para 15%, principalmente no Mimic. Isso mostra quanto do número
    anterior vinha de desenhar e avaliar com os mesmos dados, e reforça a calibração do limiar no mês 9.""")

    # ---------------- frente 2: sem treino offline
    s = section(prs, "Frente 2: RAGNAR sem treino offline", "Sinais normalizados no round, sem classificador pré-treinado")
    notes(s, """A segunda frente: tirar a dependência do treino offline. Um classificador supervisionado só reconhece o
    que viu no treino, e os diagnósticos mostraram duas fragilidades: ataques não vistos e mudança das features com
    a fase do treino.""")

    s = content(prs, "Conceitos matemáticos mais robustos")
    table(s, 0.8, 2.5, 18.4, [
        ["Problema observado", "Conceito", "Situação"],
        ["Referência = média (ruptura 0)", "**Mediana geométrica** (Weiszfeld, 1937), ruptura 50%", "implementado"],
        ["Escala muda com o treino/ambiente", "**z-score robusto** (mediana/MAD) ou rank", "validado"],
        ["Mahalanobis contaminado", "**MCD** (Rousseeuw & Van Driessen, 1999)", "implementado"],
        ["Backdoors em conluio", "**Assinaturas espectrais** (Tran et al., 2018)", "implementado"],
        ["PGD limita a norma", "**Concordância de sinais** (SignGuard, 2022; RLR, 2021)", "implementado"],
        ["Label flip afeta uma classe", "**Perda por classe** no holdout, divergência JS", "implementado"],
        ["Sinal fraco mas persistente", "**CUSUM** (Page, 1954) / SPRT (Wald, 1945)", "implementado"],
        ["Limiar fixo não controla FP", "**Predição conforme** (Vovk et al., 2005)", "mês 9"]],
        col_w=[6.0, 9.6, 2.8], size=17, row_h=0.72)
    textbox(s, 0.8, 9.2, 18.4, 0.7, ["“Implementado”: código e testes unitários prontos; a coleta com esses sinais ainda não rodou."], size=14, bullets=False, color=GRAY)
    notes(s, """Cada problema observado tem um conceito mais robusto na literatura. A média como referência pode ser
    deslocada por um único atacante; a mediana geométrica aguenta até metade de atacantes. O z-score robusto
    resolve a mudança de escala e já foi validado no Experimento 3. MCD evita que os próprios atacantes
    contaminem o Mahalanobis. Assinaturas espectrais pegam backdoors em conluio. Concordância de sinais ignora a
    norma, que é justamente o que o PGD controla. Perda por classe no holdout mira o label flip. CUSUM acumula
    evidência fraca ao longo dos rounds. Predição conforme controla falsos positivos e fica para o mês 9. Esses
    sinais estão implementados e testados, mas a coleta com eles ainda não rodou.""")

    s = content(prs, "Proposta: detector sem treino offline", "Da evidência do Experimento 3 para o desenho da Frente 2")
    steps = [["Sinais do round", "7 sinais de 4 famílias:", "geometria, fc1, holdout, histórico"],
             ["Normalização", "z robusto (mediana/MAD)", "dentro do round"],
             ["Evidência", "uni ou bilateral por sinal;", "escore = máximo (explica)"],
             ["Decisão", "z > 3 ou CUSUM; no máx.", "(n−1)/2 excluídos, desde o round 1"]]
    xs = [0.8, 5.6, 10.4, 15.2]
    for k, (x, lines) in enumerate(zip(xs, steps)):
        box(s, x, 4.1, 4.0, 2.1, lines, sizes=(19, 15))
        if k < 3:
            arrow(s, x + 4.0, 5.15, xs[k + 1], 5.15)
    callout(s, 0.8, 6.8, 11.6, 2.3, "Evidência preliminar (sinais coletados, 8 ataques)", [
        "Regra proposta: AUC 0,95, recall 0,86, 9% de falsos positivos; mesma regra com as 4 features do RAGNAR: AUC 0,78, recall 0,37.",
        "Sem os sinais de conluio: recall 0,86 → não depende de atacantes idênticos. Label flip segue fraco (recall 0,33)."], size=15)
    callout(s, 12.9, 6.8, 6.3, 2.3, "Por que sem treino", [
        "Não depende de simular ataques nem do ambiente de treino."], head=ORANGE, size=16)
    notes(s, """Esta é a lógica especificada no entregável do mês 3. Em cada round, o servidor calcula sete sinais
    de quatro famílias, normaliza cada um dentro do round com um z-score robusto e transforma em evidência: as
    normas numa direção só, os cossenos e a consistência entre rounds nos dois sentidos, porque o atacante pode
    ser diferente demais ou típico demais. O escore do cliente é a maior evidência, e o sinal que a produziu é a
    explicação. Exclui-se quem passa de três desvios robustos ou acumula evidência no CUSUM, nunca mais da metade
    dos clientes e desde o primeiro round. Aplicada aos sinais já coletados, a regra chega a AUC 0,95 e recall
    0,86 com 9% de falsos positivos, contra recall 0,37 da mesma regra com as features do RAGNAR. É evidência de
    viabilidade, não validação: os modos foram escolhidos com esses mesmos dados, e a validação é com outra
    semente, com o agente atuando na agregação.""")

    s = content(prs, "Do RAGNAR convencional ao RAGNAR evoluído")
    table(s, 0.6, 2.4, 18.8, [
        ["Dimensão", "RAGNAR convencional", "RAGNAR evoluído (proposta)", "Situação"],
        ["Treino", "offline, com ataques simulados de antemão", "**sem treino offline**: sinais normalizados no round", "sinais prontos; avaliação a rodar"],
        ["Features", "normas absolutas dos pesos, por camada", "direção robusta, camada crítica, holdout por classe, histórico", "implementado"],
        ["Decisão", "limiar fixo (0,5)", "CUSUM entre rounds + limiar conforme (FP ≤ α)", "CUSUM pronto; conforme no mês 9"],
        ["Ataques avaliados", "simples de pesos (ruído, zeros, inversão)", "**2017–2021** (8 ataques) e **2020–2024** (LP/LF, Mimic, DBA, Edge-case)", "13 implementados; Edge-case fraco"],
        ["Explicação", "nenhuma", "contribuição de cada sinal na decisão", "mês 5"]],
        col_w=[2.8, 5.4, 6.8, 3.8], size=15, row_h=0.95)
    notes(s, """Esta tabela resume a evolução. Em treino, saímos de um classificador offline que depende de ataques simulados de antemão para
    uma detecção sem treino prévio. Em features, de normas absolutas para sinais de direção robusta, camada
    crítica, holdout e histórico. Em decisão, de um limiar fixo para evidência acumulada com falsos positivos
    controlados. Em avaliação, de ataques simples de pesos para ataques da literatura recente. E acrescentamos
    explicação. A coluna de situação mostra o que já está pronto e o que falta.""")

    # ---------------- próximos passos
    s = section(prs, "Próximos passos", "Entregável do mês 3 e rumo do agente")
    notes(s, "Por fim, o que falta para o entregável deste mês e para onde o agente caminha.")

    s = content(prs, "Entregável do mês 3 e rumo do agente")
    textbox(s, 0.8, 2.5, 8.8, 0.6, ["**Especificação dos sinais (M3)**"], size=22, bullets=False, color=NAVY)
    textbox(s, 0.8, 3.2, 8.8, 6.3, [
        "✓ Catálogo de 29 sinais, direção declarada a priori.", "✓ Evidência por ataque (19 sinais medidos).",
        "✓ Lógica de decisão + explicação, avaliação preliminar.",
        "Documento: especificacao-sinais-deteccao.pdf.",
        "✓ 5 ataques recentes implementados e analisados.",
        "Comparação no framework do CoUrb (courb_eval/)."], size=19)
    textbox(s, 10.2, 2.5, 9.2, 0.6, ["**Agente = RAGNAR evoluído (M5–M6)**"], size=22, bullets=False, color=NAVY)
    textbox(s, 10.2, 3.2, 9.2, 6.5, [
        "**Sinais**: + direção robusta, camada crítica, holdout por classe, histórico.",
        "**Normalização** dentro do round (independe do ambiente).",
        "**Treino** cobrindo todas as fases do FL.",
        "**Evidência acumulada** entre rounds (CUSUM).",
        "**Explicação**: contribuição de cada sinal."], size=20)
    notes(s, """O entregável do mês 3 é a especificação dos sinais: o catálogo com a evidência de cada sinal por
    ataque, a comparação entre a formulação atual e a robusta, a lógica de decisão do agente e a comparação no
    framework do CoUrb. Para os meses 5 e 6, a proposta é que o agente seja um RAGNAR evoluído: mais sinais,
    normalização dentro do round, treino cobrindo todas as fases, acúmulo de evidência entre rounds e explicação
    de cada decisão.""")

    s = content(prs, "Limitações")
    textbox(s, 0.8, 2.6, 18.4, 7, [
        "**Uma semente**: diferenças de 1–2 pontos podem ser ruído.",
        "**Cenário único**: CIFAR-10, α = 0,5, 20% de atacantes; falta label skew severo (α = 0,1).",
        "**Experimentos 1–3 com a v1**, não com o RAGNAR original.",
        "**Análise de sinais** nos rounds iniciais e com agregação oráculo.",
        "**Ataques não adaptativos**: vetores idênticos facilitam a detecção por conluio.",
        "**Ataques recentes da revisão fora da avaliação**: LP/LF (2024), Mimic (2022), DBA (2020), Edge-case PGD completo.",
        "**Sem comparação** com BOBA, Multi-Krum etc."], size=21)
    notes(s, """As limitações que precisam ficar claras. Uma semente só. Um único cenário de dados, sem o regime de
    label skew severo. Os Experimentos 1 a 3 usaram a v1, não o RAGNAR original. A análise de sinais cobre só os
    rounds iniciais, com agregação oráculo. Os ataques não são adaptativos, e os mais recentes da revisão, como LP/LF, Mimic e DBA, ainda não foram
    avaliados. E ainda não comparamos o RAGNAR com
    as outras defesas da literatura.""")

    half = (len(REFERENCES) + 1) // 2
    for part, refs in enumerate((REFERENCES[:half], REFERENCES[half:]), 1):
        s = content(prs, f"Referências ({part}/2)")
        textbox(s, 0.8, 2.5, 18.4, 7.5, refs, size=14)
        notes(s, "Referências dos ataques, defesas e conceitos citados na apresentação.")

    s = title_slide(prs, "Obrigado!", None, ["Código, resultados e esta apresentação: fl-robustness-lab/"], icon=False)
    notes(s, "Obrigado. Fico à disposição para perguntas.")
    prs.save(str(OUT))
    print(f"{OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
