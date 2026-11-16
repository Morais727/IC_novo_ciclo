#!/usr/bin/env python3
"""Gera o Relatório 5 (mês 05) no padrão visual dos relatórios H.IAAC.

Reaproveita o motor de montagem e os blocos XML de ../entregavel_03/build_relatorio.py (modelo = .docx do
Relatório 2: cabeçalho/logo, rodapé, estilos, títulos laranja, tabelas com cabeçalho cinza, caixas sombreadas).
Entregável M5 do plano de 12 meses: "Protótipo do agente — perfil por camada, concentração e sonda".
Todo o conteúdo vem de resultados JÁ RODADOS (results/ do fl-robustness-lab); não roda treino.

    python3 build_relatorio.py      # -> [HIAAC][M05] Relatório 5.docx (PDF se houver soffice)
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m3", HERE.parent / "entregavel_03" / "build_relatorio.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
p, h1, h2, bullet, table, box, empty, run = m3.p, m3.h1, m3.h2, m3.bullet, m3.table, m3.box, m3.empty, m3.run
formula, caption = m3.formula, m3.caption
BORDERS, TEXT_W = m3.BORDERS, m3.TEXT_W


def info_table():
    rows = [("Pesquisador:", "Marcos Gilliard Oliveira Morais"), ("Orientador:", "Leandro A. Villas"),
            ("Vínculo", "Pesquisador Visitante Convidado - Iniciação Científica"),
            ("Período tratado no relatório:", "01/11/2026 a 30/11/2026")]
    trs = "".join(
        f'<w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="2736"/></w:tcPr><w:p>{run(a, bold=True, size=20)}</w:p></w:tc>'
        f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="6624"/></w:tcPr><w:p>{run(b, size=20)}</w:p></w:tc></w:tr>'
        for a, b in rows)
    return (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{TEXT_W}"/>{BORDERS.replace("{c}", "000000")}'
            f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid><w:gridCol w:w="2736"/><w:gridCol w:w="6624"/>'
            f"</w:tblGrid>{trs}</w:tbl>")


def content():
    B = []
    A = B.append
    A(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/>'
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 05</w:t></w:r></w:p>')
    A(h1("Protótipo do Agente de Detecção: Perfil por Camada, Concentração e Sonda", center=True))
    A(empty()); A(info_table()); A(empty())

    # 1 ----------------------------------------------------------------------
    A(h1("1. Objetivo e Escopo"))
    A(p("Este é o quinto entregável do plano de 12 meses. O plano pede, para o mês 5, o **protótipo do agente de "
        "detecção**, organizado em três componentes: **perfil por camada**, **concentração** e **sonda**. O agente "
        "observa as atualizações enviadas por cada cliente a cada rodada, perfila o comportamento de cada camada, "
        "concentra a evidência dispersa entre as camadas para decidir sobre o cliente e, quando necessário, confirma "
        "a suspeita com uma sonda executada no servidor. A base de validação é o conjunto de ataques injetáveis "
        "entregue no mês 4 (LP, LaySelFL e o gatilho adaptativo A3FL)."))
    A(p("O protótipo está implementado no laboratório (`src/defenses/layer_guard.py`, `src/signals.py`, "
        "`src/defenses/trigger_scan.py`) e ligável por configuração em qualquer cenário. Este relatório especifica "
        "os três componentes, a lógica de resposta e a evidência preliminar de que o agente **contém** os ataques "
        "camada-seletivos de 2026 onde a classe inteira de defesas publicadas falha. A integração com a agregação "
        "real e o relatório parcial completo são o entregável do mês 6."))

    # 2 ----------------------------------------------------------------------
    A(h1("2. Arquitetura do Agente"))
    A(p("O agente opera em três estágios encadeados, do sinal local por camada ao veredito por cliente, com uma "
        "verificação independente de gatilho quando a geometria não basta:"))
    A(table(["Estágio", "O que faz", "Saída", "Código"],
            [["1. Perfil por camada", "Para cada tensor, mede quão atípica é a contribuição de cada cliente "
              "(similaridade de direção e norma, em z robusto entre clientes).", "Matriz z[cliente, tensor]",
              "signals.py / layer_guard.py"],
             ["2. Concentração", "Conta em quantos tensores cada cliente é marcado; corrobora a evidência dispersa e "
              "decide o cliente por um limiar de contagem T.", "Veredito por cliente", "layer_guard.py (corr)"],
             ["3. Sonda", "Quando pedida, varre o modelo de cada cliente por um gatilho compartilhado usando as "
              "amostras limpas do servidor; confirma o backdoor sem depender da geometria.", "Confirmação + gatilho",
              "trigger_scan.py"]],
            [1900, 4360, 1700, 1400], size=18, bold_first_col=True))
    A(p("A **resposta** à marcação é modular (seção 6): cortar só o tensor suspeito, purgar o cliente inteiro ou "
        "substituir a contribuição suspeita pelo consenso benigno. O agente dispensa dados rotulados no servidor "
        "para os estágios 1–2 (só a sonda usa um pequeno conjunto limpo) e custa pouco (seção 7)."))

    # 3 ----------------------------------------------------------------------
    A(h1("3. Componente 1 — Perfil por Camada"))
    A(p("A entrada da rodada são os deltas de cada cliente, Δ_i = θ_i − θ_servidor, achatados e fatiados por tensor "
        "de parâmetro g (a granularidade das “camadas” no código dos autores do LP). Primeiro removemos o consenso, "
        "centrando pela mediana coordenada dos clientes:"))
    A(formula("C_i = Δ_i − mediana_j(Δ_j)"))
    A(p("Para cada tensor g, dois sinais descrevem o cliente i em relação aos demais — a **direção** (similaridade de "
        "cosseno com o vizinho mais próximo) e a **norma**:"))
    A(formula("s_g(i) = max_{j ≠ i} cos(C_i[g], C_j[g])      n_g(i) = ||C_i[g]||"))
    A(p("Cada sinal é padronizado por um **z robusto** entre clientes (mediana e desvio absoluto mediano, resistentes "
        "a outliers), e o escore do par (cliente, tensor) soma as duas evidências no sentido suspeito:"))
    A(formula("z_g(i) = max(z_sim[s_g(i)], 0) + max(z_norm[n_g(i)], 0);   marca se z_g(i) > z_thr (= 3)"))
    A(p("A intuição: um ataque por camadas constrói parte da atualização em comum entre os atacantes (direção parecida "
        "→ cosseno alto) e muitas vezes a infla em norma. O z robusto evita que o não-IID dos honestos seja marcado. "
        "O **diagnóstico central do mês** (medido com o agregador-sombra `lg_shadow`, que só registra o sinal) é que, "
        "contra o LaySelFL, o veneno **vaza pelos tensores convolucionais**: decidir tensor a tensor, isolado, perde "
        "a evidência cruzada — daí o segundo componente."))

    # 4 ----------------------------------------------------------------------
    A(h1("4. Componente 2 — Concentração (Corroboração por Contagem)"))
    A(p("Em vez de agir por tensor isolado, o agente **concentra** a evidência entre tensores. Para cada cliente, "
        "conta quantos tensores o marcam e confirma o cliente se a contagem atingir um limiar T:"))
    A(formula("contagem_i = Σ_g [ z_g(i) > z_thr ];   confirmado se contagem_i ≥ T  (confirm_tensors)"))
    A(p("A medição no `lg_shadow` mostra por que a contagem separa bem: no cenário fiel do LaySelFL (ResNet18-GN, "
        "CIFAR-10), o **atacante é acusado por ~22 tensores por rodada** (mediana; percentil 10 ≈ 14), enquanto um "
        "**cliente honesto é acusado por ~1** (percentil 90 ≈ 6). A tabela abaixo é a curva de detecção por T, medida "
        "offline a partir do sinal do portão (sem re-treinar):"))
    A(table(["Limiar T", "Precisão", "Recall", "F1", "Leitura"],
            [["4", "0,66", "1,00", "0,80", "pega todos, com mais falsos-positivos"],
             ["8", "0,86", "0,97", "0,91", "ponto de operação padrão"],
             ["11", "0,86", "1,00", "0,93", "pico de F1"],
             ["14", "0,93", "0,83", "0,88", "mais conservador, começa a perder atacante"]],
            [1500, 1700, 1700, 1500, 2960], size=18, bold_first_col=True))
    A(caption("Detecção do atacante por limiar de corroboração T (cenário fiel do LaySelFL, semente 42)."))
    A(p("O agente adota T = 8 como padrão (recall 0,97 com ~1 falso-positivo por rodada), com o pico de F1 em T = 11. "
        "Arquiteturas de corroboração mais fraca (VGG19, EfficientNet) exigem T menor — relação registrada para o "
        "mês 6, quando o T passa a ser calibrado por arquitetura."))

    # 5 ----------------------------------------------------------------------
    A(h1("5. Componente 3 — Sonda de Gatilho no Servidor"))
    A(p("A sonda é a verificação independente do sinal geométrico, para o caso em que o atacante disfarça a direção. "
        "A cada `every` rodadas, sobre as amostras limpas do servidor, ela mede o efeito de cada quadrado de cor "
        "sólida inserido na entrada do modelo de cada cliente (o ganho de confiança numa classe que não é a "
        "verdadeira), padroniza esse efeito em z robusto entre clientes e, se um mesmo candidato de gatilho for forte "
        "em ≥ 2 clientes, exclui esses clientes e **desaprende** o gatilho no modelo agregado."))
    A(bullet("**Não precisa de gatilho nem de classe-alvo conhecidos** — descobre o quadrado e a classe pela varredura."))
    A(bullet("**Complementa a geometria**: pega o backdoor pelo *efeito* no modelo, não pela forma da atualização."))
    A(bullet("**Limitação registrada**: supõe um gatilho ativável por um quadrado de cor sólida e ≥ 2 atacantes com o "
             "mesmo gatilho — a mesma premissa de conluio do componente 2 (ver seção 8)."))

    # 6 ----------------------------------------------------------------------
    A(h1("6. Lógica de Resposta"))
    A(p("Marcar não é o fim: o agente tem três respostas, de mais cirúrgica a mais drástica, com um trade-off claro "
        "entre contenção e acurácia preservada. Todas usam o mesmo detector (componentes 1–2)."))
    A(table(["Resposta", "O que faz com o suspeito", "Perfil"],
            [["corte por tensor (`corr`)", "remove o cliente dos tensores em que foi marcado; se confirmado por T, de "
              "todos os tensores.", "equilíbrio"],
             ["purga de cliente (`client`)", "se marcado em qualquer tensor, sai de toda a agregação (como "
              "FLAME/Multi-Krum).", "mais contenção"],
             ["substituição por consenso (`corr-soft`)", "troca a contribuição suspeita pela mediana dos benignos "
              "aceitos; o cliente mantém o peso.", "mais acurácia"]],
            [2600, 5260, 1500], size=18, bold_first_col=True))

    # 7 ----------------------------------------------------------------------
    A(h1("7. Evidência Preliminar"))
    A(p("O protótipo já foi validado no cenário fiel do LaySelFL (Li et al., CVPR 2026), onde o ataque pega de forma "
        "decisiva (ASR 82% sem defesa) e, com ele, **toda a classe de defesas publicadas falha**. O agente contém, e "
        "o resultado se sustenta com rigor estatístico em 4 sementes (ASR final do backdoor, média ± desvio):"))
    A(table(["Agregação", "ASR (4 sementes)", "Contém?"],
            [["sem defesa (portão)", "77,7 ± 10,4%", "— (ataque pega)"],
             ["agente — purga de cliente", "4,3 ± 1,3%", "sim"],
             ["agente — substituição por consenso", "5,5 ± 2,0%", "sim"],
             ["agente — corte por tensor", "10,5 ± 6,3%", "3 de 4 (falha na s44)"],
             ["Multi-Krum / AlignIns / FedSurrogate / FLAME / RFLDefense", "70–88%", "não"],
             ["FLD / CosDefense (as mais parecidas com o agente)", "67–88% / 54–81%", "não (4 de 4 sementes)"],
             ["LASA (a melhor concorrente)", "21,7 ± 7,6%", "não"]],
            [3600, 3360, 2400], size=18, bold_first_col=True))
    A(caption("ResNet18-GN, CIFAR-10, 20 nós (6 maliciosos), α 0,9, 205 rodadas, ataque 80–200. 13 agregações × 4 sementes. "
              "ASR medido na rodada 205; um modelo nunca atacado já marca ~7%."))
    A(bullet("**Falha da corr na semente 44**: o ASR final de 22,9% foi examinado com gatilhos de controle (o gatilho real "
             "funciona 14 p.p. acima de um gatilho falso e sequestra 14% das imagens de forma estável): é backdoor real, "
             "não ruído. Por isso a corr conta como contenção em 3 das 4 sementes."))
    A(bullet("**Pico durante o ataque** (rodadas 80–200), além do valor final: corr-soft 5,0–8,6% nas 4 sementes (a mais "
             "estável), purga de cliente até 18,2%, corte por tensor até 49,5% e LASA 25,8–42,4%."))
    A(bullet("**Variação entre corridas**: repetir a substituição por consenso na semente 44, com a mesma configuração, deu "
             "13,2% de ASR final (antes, 3,4%), por não determinismo da GPU. As médias acima são de uma corrida por "
             "semente; a dispersão entre corridas está sendo medida e entra no relatório de robustez."))
    A(bullet("**Outras arquiteturas** (1 semente): o agente reduz o LaySelFL a 9–13% no VGG19 e a 13–19% no "
             "EfficientNet, com T menor (corroboração mais fraca → limiar mais baixo). Valores acima de ~10% só serão "
             "chamados de contenção depois do teste de gatilho de controle."))
    A(bullet("**Gatilho adaptativo (A3FL), 1 semente**: no CIFAR-10, de 97,7% sem defesa para 5,3% (purga de cliente), "
             "12,6% (corte por tensor) e 16,7% (substituição por consenso). No FMNIST o agente **não contém**: de 87,0% "
             "para 33,1%, 68,2% e 69,8% (seção 8)."))
    A(bullet("**Custo**: o agente adiciona ≤ 6,5% de tempo por rodada sobre o FedAvg — contra, por exemplo, +93% de "
             "uma defesa concorrente com busca de vizinhos."))

    # 8 ----------------------------------------------------------------------
    A(h1("8. Limitações e Próximos Passos"))
    A(bullet("**Modelo de ameaça (≥ 2 atacantes coordenados)**: tanto a corroboração quanto a sonda dependem de ao "
             "menos dois maliciosos parecidos; contra um atacante solitário o sinal de semelhança não ajuda. É um "
             "limite estrutural da família, declarado como premissa (o LaySelFL usa 6 atacantes)."))
    A(bullet("**A3FL no FMNIST**: o agente não contém (33–70% contra 87% sem defesa), ao contrário do CIFAR-10. O "
             "diagnóstico da assimetria (análise do sinal, sem novo treino) está planejado; até lá é limitação declarada."))
    A(bullet("**Atacante que conhece a defesa (H4)**: implementado e medido antes do previsto. O adversário adaptativo "
             "conhece a regra e o T, concentra o veneno em poucos tensores e dessincroniza os atacantes para ficar abaixo "
             "de T. Ele **vence o agente**: corte por tensor 62–65% (4 sementes), substituição por consenso 51–57% "
             "(5 sementes); a purga de cliente chega a 27–62% durante o ataque. Uma memória temporal da contagem não "
             "resolve (testada offline). A resposta em teste age na agregação (média aparada e corte de norma por tensor, "
             "contra um atacante que também conhece a agregação) e é o tema do relatório de robustez (mês 10)."))
    A(bullet("**Mês 6**: integrar o agente à agregação real do laboratório como um módulo único (perfil → "
             "concentração → sonda → resposta), calibrar T por arquitetura e consolidar o relatório parcial."))

    # 9 ----------------------------------------------------------------------
    A(h1("9. Referências"))
    for ref in [
        "[Li et al. 2026] Li, Y. et al. Eliminate Distance Differences Induced by Backdoor Attacks: Layer-Selective Training and Clipping to Mask Backdoor Models (LaySelFL). CVPR, 2026.",
        "[Zhuang et al. 2024] Zhuang, H. et al. Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
        "[Zhang et al. 2023] Zhang, Z. et al. A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning. NeurIPS, 2023.",
        "[Lin et al. 2023] Lin, H., Wu, Z., Liao, X., Zhou, P. FLD: Federated Layer Detection (corroboração por contagem de camadas). arXiv:2303.00302, 2023.",
        "[Yan et al. 2023] CosDefense: filtragem de clientes por similaridade de cosseno. ICLR 2023 Workshop on Backdoor Attacks and Defenses, arXiv:2304.00160.",
        "[FedSurrogate 2026] Criticalidade de camada e substituição cirúrgica com reseleção dinâmica. 2026.",
        "[Nguyen et al. 2022] FLAME: Taming Backdoors in Federated Learning. USENIX Security, 2022.",
        "[Blanchard et al. 2017] Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum/Multi-Krum). NeurIPS, 2017.",
        "[AlignIns 2025] Alinhamento de atualizações para detecção de backdoor. 2025.",
        "[LASA 2025] Layer-wise sparsification and analysis para robustez em FL. 2025.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


if __name__ == "__main__":
    m3.content = content
    m3.info_table = info_table
    m3.IMAGES = []
    m3.HERE = HERE
    m3.OUT = HERE / "[HIAAC][M05] Relatório 5.docx"
    try:
        m3.main()
    except Exception as e:                       # soffice (PDF) pode faltar nesta máquina; o .docx já foi salvo
        out = HERE / "[HIAAC][M05] Relatório 5.docx"
        if out.exists():
            print(out, "(.docx gerado; PDF não — soffice indisponível:", type(e).__name__, ")")
        else:
            raise
