#!/usr/bin/env python3
"""Relatório 8 (mês 08), padrão H.IAAC. Reusa o motor de ../entregavel_03/build_relatorio.py.
Entregável M8: "Envenenamento de modelo (LP, LF, IPM, LIE, MinMax) no repositório do LP".
RASCUNHO a partir de resultados já rodados (results/m4_efeito/efeito.csv, results/fase9/AUTORES.md). Não roda treino.
    python3 build_relatorio.py   # -> [HIAAC][M08] Relatório 8.docx
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
            ("Período tratado no relatório:", "01/02/2027 a 28/02/2027")]
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
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 08</w:t></w:r></w:p>')
    A(h1("Ataques de Envenenamento de Modelo e Validação no Repositório do LP", center=True))
    A(empty()); A(info_table()); A(empty())

    A(h1("1. Objetivo e Escopo"))
    A(p("Oitavo entregável do plano. O mês 8 pede os **ataques de envenenamento de modelo** — IPM, LIE, MinMax, "
        "inversão de sinal e os ataques por camadas LP/LF — e a **validação no repositório original do LP** "
        "[Zhuang et al. 2024]. Envenenamento de modelo manipula diretamente a atualização enviada (não os dados), o "
        "regime mais forte de ataque e o alvo principal do agente de defesa."))

    A(h1("2. Ataques Não-Direcionados (degradação de acurácia)"))
    A(p("Atacam a disponibilidade: empurram a média para longe do ótimo honesto. Medidos no cenário do laboratório "
        "(SmallCNN/CIFAR-10, 20 clientes, 4 maliciosos, 4 sementes); efeito = queda de acurácia em p.p. (sem defesa). "
        "Acurácia honesta de referência ≈ 71,9%."))
    A(table(["Ataque", "Acurácia", "Queda (p.p.)", "Mecanismo"],
            [["IPM (inner-product manipulation)", "67,1%", "−4,9", "inverte a média honesta com escala ótima"],
             ["Inversão de sinal (sign flip)", "67,0%", "−4,9", "envia −escala × delta honesto"],
             ["LF (layer-wise flipping)", "67,5%", "−4,5", "inverte só as camadas mais críticas p/ a tarefa"],
             ["MinSum", "68,3%", "−3,6", "maximiza a soma de distâncias dentro do limite"],
             ["MinMax", "68,5%", "−3,5", "maximiza a distância máxima dentro do limite"],
             ["LIE (little is enough)", "68,9%", "−3,1", "desvio pequeno mas coordenado, abaixo do filtro"]],
            [3200, 1300, 1500, 3360], size=18, bold_first_col=True))
    A(caption("Ataques de modelo não-direcionados (results/m4_efeito/efeito.csv). Em FedAvg a queda é moderada (3–5 p.p.); são a base de estresse das defesas robustas."))

    A(h1("3. Ataque Direcionado por Camadas (LP)"))
    A(p("O **LP** [Zhuang et al. 2024] é o ataque por camadas que motiva o projeto: treina um modelo limpo e um com "
        "backdoor, identifica as camadas críticas do backdoor por substituição e envia o delta limpo com só essas "
        "camadas trocadas. No cenário do laboratório atinge **ASR 65,7%** (base 2,5%) com perda de acurácia "
        "desprezível (−0,5 p.p.) — exatamente o perfil “invisível” que derrota as defesas por distância. É o "
        "antecessor direto do LaySelFL (CVPR 2026), o foco do agente."))

    A(h1("4. Validação no Repositório Original do LP"))
    A(p("Para responder à pergunta de revisor “a sua versão do LP é mais fraca que a original?”, rodamos a "
        "nossa proposta **dentro do código dos autores** (Zhuang et al., ICLR 2024), sem alterar o ataque: um "
        "adaptador substitui a agregação por `layer_guard_sn` (limiar 3), com o cenário da Tab. 2 do artigo "
        "(CNN rlr\\_mnist, Fashion-MNIST não-IID, 100 clientes, 10 por rodada, 200 rodadas), sementes 1–3."))
    A(table(["Maliciosos", "FedAvg: sucesso (acurácia)", "Nossa: sucesso (acurácia)", "Veredito"],
            [["10% (1 atacante/rodada)", "77,8–78,4% (≈89,6%)", "71,5–72,4% (≈89,6%)", "não defende"],
             ["20% (2 atacantes/rodada)", "88,4–90,4% (≈89,6%)", "0,1–0,3% (≈89,6%)", "defende"]],
            [2200, 2900, 2860, 1400], size=18, bold_first_col=True))
    A(caption("Nossa proposta no repositório original do LP (FMNIST, CNN, 200 rodadas), média das sementes 1–3."))
    A(box("Leitura — o conluio é a condição", [
        "Com 20% (2 atacantes por rodada) a proposta ZERA o backdoor (0,1–0,3%) sem custo de acurácia, no próprio "
        "código dos autores — confirma que não é artefato da nossa reimplementação.",
        "Com 10% (1 atacante por rodada) ela NÃO defende: um atacante solitário não corrobora com ninguém, e o sinal "
        "de semelhança não o distingue dos honestos. É o limite estrutural declarado (≥ 2 atacantes coordenados).",
        "A evidência do mecanismo (contagem de corroboração do atacante por rodada, 1 vs 2 atacantes) está instrumentada "
        "e entra assim que houver GPU livre."]))

    A(h1("5. Referências"))
    for ref in [
        "[Zhuang et al. 2024] Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
        "[Baruch et al. 2019] A Little Is Enough: Circumventing Defenses for Distributed Learning (LIE). NeurIPS, 2019.",
        "[Shejwalkar e Houmansadr 2021] Manipulating the Byzantine (IPM, MinMax, MinSum). NDSS, 2021.",
        "[Fang et al. 2020] Local Model Poisoning Attacks to Byzantine-Robust Federated Learning. USENIX Security, 2020.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


if __name__ == "__main__":
    m3.content = content
    m3.info_table = info_table
    m3.IMAGES = []
    m3.HERE = HERE
    m3.OUT = HERE / "[HIAAC][M08] Relatório 8.docx"
    try:
        m3.main()
    except Exception as e:
        if m3.OUT.exists():
            print(m3.OUT, "(.docx gerado; sem PDF —", type(e).__name__, ")")
        else:
            raise
