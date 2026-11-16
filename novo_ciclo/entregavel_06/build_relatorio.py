#!/usr/bin/env python3
"""Gera o Relatório 6 (mês 06) no padrão visual dos relatórios H.IAAC.

Reaproveita o motor de montagem/blocos XML de ../entregavel_03/build_relatorio.py (modelo = .docx do Relatório 2).
Entregável M6 do plano: "Integração e relatório parcial — agregação real e resposta cirúrgica". RASCUNHO montado
a partir de resultados JÁ RODADOS (results/ do fl-robustness-lab); não roda treino. É um relatório PARCIAL: consolida
os meses 1–6 e fixa a posição do trabalho frente ao estado da arte.

    python3 build_relatorio.py      # -> [HIAAC][M06] Relatório 6.docx (PDF se houver soffice)
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
            ("Período tratado no relatório:", "01/12/2026 a 31/12/2026")]
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
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 06</w:t></w:r></w:p>')
    A(h1("Integração do Agente à Agregação e Relatório Parcial", center=True))
    A(empty()); A(info_table()); A(empty())

    # 1
    A(h1("1. Objetivo e Escopo"))
    A(p("Este é o sexto entregável e o **relatório parcial** do plano de 12 meses. O plano pede, para o mês 6, a "
        "**integração do agente de detecção à agregação real** e a **resposta cirúrgica** aos clientes marcados, além "
        "da consolidação do que foi feito até aqui. O relatório (i) descreve como o agente especificado no mês 5 opera "
        "como o agregador do sistema federado, (ii) formaliza a resposta cirúrgica e suas variantes e (iii) consolida "
        "o resultado central do projeto, posicionando-o frente ao estado da arte."))

    # 2
    A(h1("2. Integração à Agregação Real"))
    A(p("O agente não é um filtro externo: ele **é a regra de agregação** do laboratório. A cada rodada, o servidor "
        "coleta os deltas dos clientes e chama o agente, que devolve o modelo agregado e o veredito de detecção. A "
        "troca de defesa é por configuração (um nome no registro de agregadores), o que permite comparar o agente, no "
        "mesmo cenário e sementes, com FedAvg e com as defesas do estado da arte."))
    A(table(["Propriedade da integração", "Como está"],
            [["Ponto de entrada", "regra de agregação do loop federado (um agregador entre os demais)"],
             ["Entrada / saída", "deltas dos clientes → modelo agregado + veredito (clientes/tensores removidos)"],
             ["Dados no servidor", "nenhum para perfil + concentração; só a sonda usa um pequeno conjunto limpo"],
             ["Seleção por configuração", "trocável por nome, comparável a FedAvg e às defesas SOTA no mesmo cenário"],
             ["Custo", "≤ 6,5% de tempo por rodada sobre o FedAvg"]],
            [3400, 5960], size=18, bold_first_col=True))

    # 3
    A(h1("3. Resposta Cirúrgica"))
    A(p("A resposta cirúrgica é o núcleo do mês 6: em vez de descartar o cliente inteiro (como FLAME/Multi-Krum), o "
        "agente age **por tensor**. O percurso de projeto, fixado antes de rodar e depois diagnosticado, foi:"))
    A(bullet("**Corte por tensor isolado (`sn`)** — remove o cliente só dos tensores em que é marcado. Falha contra o "
             "LaySelFL (ASR 76,7%): o veneno vaza pelos tensores convolucionais, onde cada tensor isolado fica abaixo "
             "do limiar."))
    A(bullet("**Corroboração (`corr`)** — concentra a evidência entre tensores (contagem ≥ T) e, para o cliente "
             "confirmado, faz o corte em todas as camadas, fechando o vazamento. Contém (ASR 7,9%)."))
    A(bullet("**Substituição por consenso (`corr-soft`)** — troca a contribuição suspeita pela mediana dos benignos, "
             "preservando a massa de votos e a acurácia (ASR 8,8%, maior acurácia das três)."))
    A(p("A resposta é, portanto, um dial entre contenção e acurácia: `client` contém mais, `corr-soft` preserva mais "
        "acurácia, `corr` equilibra — todas sobre o mesmo detector."))

    # 4
    A(h1("4. Resultado Consolidado (Meses 1–6)"))
    A(p("No cenário fiel do ataque de ponta (LaySelFL, Li et al., CVPR 2026: ResNet18-GN, CIFAR-10), o backdoor pega "
        "de forma decisiva sem defesa e **a classe inteira de defesas publicadas falha**. O agente contém, com rigor "
        "estatístico em 4 sementes (ASR final do backdoor, média ± desvio):"))
    A(table(["Agregação", "ASR (4 sementes)", "Contém?"],
            [["sem defesa (portão)", "77,7 ± 10,4%", "— (ataque pega)"],
             ["agente — purga de cliente", "4,3 ± 1,3%", "sim"],
             ["agente — substituição por consenso", "5,5 ± 2,0%", "sim"],
             ["agente — corte por tensor", "10,5 ± 6,3%", "sim (3/4)"],
             ["Multi-Krum / AlignIns / FedSurrogate / FLAME / RFLDefense", "70–88%", "não"],
             ["LASA (melhor concorrente)", "21,7 ± 7,6%", "não"]],
            [3600, 3360, 2400], size=18, bold_first_col=True))
    A(caption("11 agregações × 4 sementes; ataque 80–200, 205 rodadas. A LASA parecia conter com 1 semente (13,5%) — era sorte."))
    A(bullet("**Generalidade**: contém também no VGG19 (9–13%) e no EfficientNet (13–19%), e sob o gatilho adaptativo "
             "A3FL no CIFAR-10 (12,6% contra 97,7% sem defesa)."))
    A(bullet("**Detecção**: o atacante é acusado por ~22 tensores/rodada vs. ~1 do honesto; a curva por T dá F1 0,91 "
             "em T = 8 (pico 0,93 em T = 11)."))

    # 5
    A(h1("5. Posição Frente ao Estado da Arte"))
    A(p("Os ingredientes do agente têm precedentes isolados — cosseno de atualizações (CosDefense, FoolsGold, "
        "AlignIns), corroboração por contagem de camadas (FLD, 2023) e resposta por camada com substituição "
        "(FedSurrogate, 2026). A originalidade defensável é **empírica e diagnóstica**, não de ingrediente:"))
    A(bullet("o **sinal específico** — cosseno par-a-par de deltas centrados pela mediana + norma, por tensor, com "
             "gate de corroboração por contagem T;"))
    A(bullet("a **demonstração** de que isso **contém** os ataques camada-seletivos de 2026 (LaySelFL, A3FL) que "
             "derrubam a classe inteira — **inclusive a FedSurrogate e a FLD**, que são do mesmo mecanismo;"))
    A(bullet("o **diagnóstico do vazamento convolucional** e sua correção pela corroboração;"))
    A(bullet("a variante **corr-soft** (substituição por consenso) e a ausência de dados limpos no servidor, com "
             "overhead baixo."))

    # 6
    A(h1("6. Balanço do Plano e Próximos Passos"))
    A(table(["Mês", "Entregável", "Estado"],
            [["M1–M3", "Infraestrutura, ataques da revisão, especificação dos sinais", "concluído"],
             ["M4", "Conjunto de ataques injetáveis (LP completo, LaySelFL, A3FL)", "concluído"],
             ["M5", "Protótipo do agente (perfil, concentração, sonda)", "concluído"],
             ["M6", "Integração à agregação + resposta cirúrgica + relatório parcial", "este relatório"],
             ["M7–M8", "Envenenamento de dados (H3) e de modelo", "ataques prontos; grades a consolidar"],
             ["M9", "Falsos positivos e impacto; limiar calibrado (predição conforme)", "α 0,1 medido; calibração a fazer"],
             ["M10", "Atacante que conhece a defesa (H4)", "adaptativo implementado; em teste"],
             ["M11–M12", "Artigo submetido e relatório final reprodutível", "resultados prontos; escrita em curso"]],
            [1200, 6160, 2000], size=18, bold_first_col=True))
    A(p("O núcleo experimental do artigo já existe. Os meses seguintes consolidam as grades de ataque, calibram o "
        "limiar por arquitetura e por predição conforme, fecham o atacante adaptativo (H4) e convertem o conjunto em "
        "artigo e em repositório reprodutível."))

    # 7
    A(h1("7. Limitações"))
    A(bullet("**≥ 2 atacantes coordenados**: corroboração e sonda dependem de semelhança entre maliciosos; um atacante "
             "solitário não é separável pelo sinal — premissa declarada do modelo de ameaça."))
    A(bullet("**A3FL no FMNIST (68%)**: assimetria com o CIFAR em apuração (variantes de purga agressiva em teste)."))
    A(bullet("**Limiar por arquitetura**: corroboração mais fraca (VGG19, EfficientNet) exige T menor; a calibração "
             "automática é do mês 9."))

    # 8
    A(h1("8. Referências"))
    for ref in [
        "[Li et al. 2026] Eliminate Distance Differences Induced by Backdoor Attacks (LaySelFL). CVPR, 2026.",
        "[Zhuang et al. 2024] Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
        "[Zhang et al. 2023] A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning. NeurIPS, 2023.",
        "[FLD 2023] Federated Layer Detection (corroboração por contagem de camadas).",
        "[FedSurrogate 2026] Criticalidade de camada e substituição cirúrgica.",
        "[Nguyen et al. 2022] FLAME. USENIX Security, 2022.",
        "[Blanchard et al. 2017] Krum/Multi-Krum. NeurIPS, 2017.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


if __name__ == "__main__":
    m3.content = content
    m3.info_table = info_table
    m3.IMAGES = []
    m3.HERE = HERE
    m3.OUT = HERE / "[HIAAC][M06] Relatório 6 (parcial).docx"
    try:
        m3.main()
    except Exception as e:
        out = m3.OUT
        if out.exists():
            print(out, "(.docx gerado; PDF não — soffice indisponível:", type(e).__name__, ")")
        else:
            raise
