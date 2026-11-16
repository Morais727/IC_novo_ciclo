#!/usr/bin/env python3
"""Relatório 7 (mês 07), padrão H.IAAC. Reusa o motor de ../entregavel_03/build_relatorio.py.
Entregável M7: "Envenenamento de dados (label flip, pixel, DBA) e teste da hipótese H3 (sonda)".
RASCUNHO a partir de resultados já rodados (results/m4_efeito/efeito.csv, results/probe_h3/). Não roda treino.
    python3 build_relatorio.py   # -> [HIAAC][M07] Relatório 7.docx
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m3", HERE.parent / "entregavel_03" / "build_relatorio.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
p, h1, h2, bullet, table, box, empty, run = m3.p, m3.h1, m3.h2, m3.bullet, m3.table, m3.box, m3.empty, m3.run
caption = m3.caption
BORDERS, TEXT_W = m3.BORDERS, m3.TEXT_W


def info_table():
    rows = [("Pesquisador:", "Marcos Gilliard Oliveira Morais"), ("Orientador:", "Leandro A. Villas"),
            ("Vínculo", "Pesquisador Visitante Convidado - Iniciação Científica"),
            ("Período tratado no relatório:", "01/01/2027 a 31/01/2027")]
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
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 07</w:t></w:r></w:p>')
    A(h1("Ataques de Envenenamento de Dados e Teste da Hipótese H3 (Sonda)", center=True))
    A(empty()); A(info_table()); A(empty())

    A(h1("1. Objetivo e Escopo"))
    A(p("Sétimo entregável do plano. O mês 7 pede os **ataques de envenenamento de dados** — troca de rótulos "
        "(*label flip*), backdoor por *pixel* e o backdoor distribuído DBA — e o **teste da hipótese H3** (a sonda de "
        "gatilho). Os ataques já estão ligáveis por configuração (entregue no mês 4); aqui medimos o efeito de cada um "
        "no cenário do laboratório e reportamos o veredito de H3."))

    A(h1("2. Ataques de Envenenamento de Dados"))
    A(p("Envenenamento de dados altera o conjunto local do cliente malicioso (rótulo e/ou gatilho) e deixa o treino "
        "normal. Medidas no cenário do laboratório (SmallCNN/CIFAR-10, 20 clientes, 4 sementes); ASR = taxa de ativação "
        "do backdoor, base = mesmo valor sem ataque."))
    A(table(["Ataque", "Maliciosos", "Acurácia (p.p. de queda)", "Efeito", "Base"],
            [["Troca de rótulos (label flip)", "4", "70,9% (−1,0)", "troca 0→2: 17,5%", "4,9%"],
             ["Backdoor por pixel (gatilho 4×4)", "2", "71,3% (−0,7)", "ASR→classe 0: 85,9%", "2,5%"],
             ["Backdoor PGD (restrito+escalado)", "2", "71,4% (−0,6)", "ASR→classe 0: 83,4%", "2,5%"],
             ["DBA (gatilho distribuído)", "4", "70,7% (−1,2)", "ASR→classe 0: 75,6%", "2,5%"],
             ["Edge-case PGD (subpop. rara)", "2", "70,6% (−1,4)", "ASR→caminhão: 24,0%", "3,0%"]],
            [2900, 1300, 2360, 1900, 900], size=18, bold_first_col=True))
    A(caption("Efeito dos ataques de envenenamento de dados (results/m4_efeito/efeito.csv; intervalos nas 4 sementes no CSV)."))
    A(bullet("Os backdoors por **pixel** e **PGD** são os mais fortes (ASR 83–86%); o **DBA** divide o gatilho entre os "
             "4 atacantes e ainda chega a 75,6%. A **troca de rótulos** é um ataque de disponibilidade suave (eleva a "
             "confusão 0→2 de 4,9% para 17,5%, com perda mínima de acurácia)."))
    A(bullet("O **edge-case** (subpopulação rara sem gatilho) é o mais fraco nesta configuração (24%), como esperado "
             "para um backdoor semântico."))

    A(h1("3. Teste da Hipótese H3 — Sonda com Gatilhos Aleatórios"))
    A(p("**H3 (enunciado):** uma sonda no servidor que insere **gatilhos aleatórios** nas amostras limpas e mede o "
        "desvio de confiança consegue, por si só, separar clientes maliciosos de honestos — sem conhecer o gatilho real."))
    A(p("**Procedimento** (`experiments/probe_h3.py`, `results/probe_h3/`): para cada cliente, aplicam-se gatilhos "
        "aleatórios às amostras limpas do servidor e mede-se o ganho de confiança em classes não-verdadeiras; o escore "
        "é padronizado entre clientes (AUC de separação maliciosos × honestos)."))
    A(table(["Sinal", "α 0,5", "α 0,4", "α 0,3", "α 0,2", "α 0,1"],
            [["Sonda com gatilho aleatório (H3)", "0,44", "0,51", "0,51", "0,50", "0,32"],
             ["Sonda com o gatilho real (referência)", "0,83", "—", "—", "—", "—"]],
            [3760, 1120, 1120, 1120, 1120, 1120], size=18, bold_first_col=True))
    A(caption("AUC de separação (maliciosos × honestos). Gatilho aleatório ≈ acaso (0,5); gatilho real separa (0,83)."))
    A(box("Veredito — H3 REFUTADA", [
        "A sonda com gatilhos aleatórios fica no acaso (AUC 0,44; nunca acima de 0,51) — não separa maliciosos de "
        "honestos. Só com o gatilho REAL a sonda informa (0,83).",
        "Consequência de projeto: a sonda não pode varrer o espaço de gatilhos às cegas; precisa de um CANDIDATO "
        "estruturado. O agente (mês 5) resolve isso varrendo quadrados de cor sólida e exigindo corroboração entre "
        "clientes (≥ 2 com o mesmo gatilho), em vez de gatilhos aleatórios."]))

    A(h1("4. Implicação para o Agente"))
    A(p("H3 delimita o componente “sonda” do agente: a varredura é por candidatos estruturados (quadrados de "
        "cor sólida) e corroborada entre clientes, não por amostragem aleatória de gatilhos. Assim a sonda complementa "
        "o sinal geométrico (perfil + concentração) nos casos em que o atacante disfarça a direção, sem herdar a "
        "fragilidade do gatilho aleatório refutada aqui."))

    A(h1("5. Referências"))
    for ref in [
        "[Xie et al. 2020] DBA: Distributed Backdoor Attacks against Federated Learning. ICLR, 2020.",
        "[Bagdasaryan et al. 2020] How To Backdoor Federated Learning. AISTATS, 2020.",
        "[Wang et al. 2020] Attack of the Tails: Yes, You Really Can Backdoor Federated Learning. NeurIPS, 2020.",
        "[Gu et al. 2019] BadNets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access, 2019.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


if __name__ == "__main__":
    m3.content = content
    m3.info_table = info_table
    m3.IMAGES = []
    m3.HERE = HERE
    m3.OUT = HERE / "[HIAAC][M07] Relatório 7.docx"
    try:
        m3.main()
    except Exception as e:
        if m3.OUT.exists():
            print(m3.OUT, "(.docx gerado; sem PDF —", type(e).__name__, ")")
        else:
            raise
