#!/usr/bin/env python3
"""Gera o Relatório 3 (mês 03) no padrão visual dos relatórios H.IAAC.

Usa o .docx do Relatório 2 como modelo: mantém cabeçalho (logo), rodapé ("Página X de Y"), estilos e
numeração, e substitui só o corpo. Os blocos XML (título, tabela de identificação, títulos laranja,
parágrafos Arial 11 sem itálico, marcadores, tabelas com cabeçalho cinza, caixas sombreadas) são os
mesmos do Relatório 2. Tabelas e figuras vêm de ../fl-robustness-lab/results/.

    python3 build_relatorio.py      # -> [HIAAC][M03] Relatório 3.docx e .pdf
"""
import csv
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "entregavel_02" / "[HIAAC][M02] Relatório 2.docx"
RES = HERE.parent / "fl-robustness-lab" / "results"
OUT = HERE / "[HIAAC][M03] Relatório 3.docx"

ORANGE = "FF9900"
TEXT_W = 9360  # largura útil (Letter, margens de 1")


# ----------------------------------------------------------------------------- blocos XML do modelo
def run(text, bold=False, italic=False, code=False, size=22, color=None):
    rpr = ['<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>' if code
           else '<w:rFonts w:ascii="Arial" w:hAnsi="Arial"/>',
           "<w:b/>" if bold else '<w:b w:val="0"/>',
           "<w:i/>" if italic else '<w:i w:val="0"/>',
           f'<w:color w:val="{color}"/>' if color else "",
           f'<w:sz w:val="{size}"/>']
    return f'<w:r><w:rPr>{"".join(rpr)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text, size=22, italic_all=False):
    """Marcação leve: **negrito**, *itálico*, `código`."""
    out = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append(run(tok[2:-2], bold=True, italic=italic_all, size=size))
        elif tok.startswith("*"):
            out.append(run(tok[1:-1], italic=True, size=size))
        elif tok.startswith("`"):
            out.append(run(tok[1:-1], code=True, size=size - 2))
        else:
            out.append(run(tok, italic=italic_all, size=size))
    return "".join(out)


def p(text, align=None, size=22, italic=False, after=None):
    ppr = (f'<w:jc w:val="{align}"/>' if align else "") + (f'<w:spacing w:after="{after}"/>' if after is not None else "")
    return f"<w:p><w:pPr>{ppr}</w:pPr>{rich(text, size, italic)}</w:p>"


def empty():
    return "<w:p><w:pPr/></w:p>"


def h1(text, center=False):
    jc = '<w:jc w:val="center"/>' if center else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading1"/>{jc}</w:pPr><w:r><w:rPr><w:color w:val="{ORANGE}"/></w:rPr>'
            f"<w:t>{escape(text)}</w:t></w:r></w:p>")


def h2(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:rPr><w:color w:val="{ORANGE}"/></w:rPr>'
            f"<w:t>{escape(text)}</w:t></w:r></w:p>")


def bullet(text, level=0):
    return (f'<w:p><w:pPr><w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="1"/></w:numPr></w:pPr>'
            f"{rich(text)}</w:p>")


def formula(text):
    return f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{run(text, italic=True, size=22)}</w:p>'


def caption(text):
    return f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{rich(text, size=18, italic_all=True)}</w:p>'


BORDERS = ('<w:tblBorders>' + "".join(
    f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="{{c}}"/>'
    for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")


def table(header, rows, widths, size=18, bold_first_col=False):
    assert abs(sum(widths) - TEXT_W) < 20, (sum(widths), header)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    def cell(text, w, head=False, bold=False):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if head else ""
        body = run(text, bold=True, size=size) if head else rich(f"**{text}**" if bold and text else text, size)
        return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{shd}</w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="left"/></w:pPr>{body}</w:p></w:tc>')
    trs = ['<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(cell(h, w, head=True) for h, w in zip(header, widths)) + "</w:tr>"]
    for r in rows:
        trs.append("<w:tr>" + "".join(cell(str(v), w, bold=(bold_first_col and k == 0))
                                      for k, (v, w) in enumerate(zip(r, widths))) + "</w:tr>")
    return (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{TEXT_W}"/>{BORDERS.replace("{c}", "000000")}'
            f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>' + empty())


def box(title, lines):
    """Caixa sombreada de uma célula (mesmo recurso do bloco de código do Relatório 2)."""
    paras = f'<w:p><w:pPr><w:spacing w:after="60"/></w:pPr>{run(title, bold=True, size=20, color="C55A11")}</w:p>'
    paras += "".join(f'<w:p><w:pPr><w:spacing w:after="60"/></w:pPr>{rich(l, 20)}</w:p>' for l in lines)
    return (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{TEXT_W}"/>{BORDERS.replace("{c}", "BFBFBF")}</w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{TEXT_W}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{TEXT_W}"/>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="F4F4F4"/></w:tcPr>{paras}</w:tc></w:tr></w:tbl>' + empty())


IMAGES = []  # (rId, nome no pacote, origem)


def image(path: Path, width_in=6.5):
    w_px, h_px = struct.unpack(">II", path.read_bytes()[16:24])
    cx = int(width_in * 914400)
    cy = int(cx * h_px / w_px)
    k = len(IMAGES) + 1
    rid, name = f"rIdFig{k}", f"fig{k}.png"
    IMAGES.append((rid, name, path))
    return (f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{100 + k}" name="Figura {k}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{100 + k}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            f"</wp:inline></w:drawing></w:r></w:p>")


def info_table():
    rows = [("Pesquisador:", "Marcos Gilliard Oliveira Morais"), ("Orientador:", "Leandro A. Villas"),
            ("Vínculo", "Pesquisador Visitante Convidado - Iniciação Científica"),
            ("Período tratado no relatório:", "01/09/2026 a 30/09/2026")]
    trs = "".join(
        f'<w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="2736"/></w:tcPr><w:p>{run(a, bold=True, size=20)}</w:p></w:tc>'
        f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="6624"/></w:tcPr><w:p>{run(b, size=20)}</w:p></w:tc></w:tr>'
        for a, b in rows)
    return (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{TEXT_W}"/>{BORDERS.replace("{c}", "000000")}'
            f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid><w:gridCol w:w="2736"/><w:gridCol w:w="6624"/>'
            f"</w:tblGrid>{trs}</w:tbl>")


# ----------------------------------------------------------------------------- dados dos resultados
def num(v, d=2):
    return f"{float(v):.{d}f}".replace(".", ",")


def effect_rows():
    label = {"gaussian_noise": "Ruído gaussiano", "ipm": "IPM", "lie": "LIE", "minmax": "MinMax", "minsum": "MinSum",
             "label_flip": "Label flip", "backdoor_pixel": "Backdoor (pixel)", "backdoor_pgd": "Backdoor (PGD)",
             "mimic": "Mimic *", "dba": "DBA *", "lp": "LP *", "lf": "LF *", "edge_case_pgd": "Edge-case PGD *"}
    rows = []
    for r in csv.DictReader(open(RES / "attack_effect" / "effect.csv")):
        f = lambda k, d=1: num(r[k], d) if r[k] not in ("", "None") else "–"
        rows.append([label[r["ataque"]], r["maliciosos"], f("ma"), f"{float(r['delta_ma']):+.1f}".replace(".", ","),
                     f("asr"), f("asr_base")])
    return rows


def detection_rows():
    lines = (RES / "detection_logic" / "summary.md").read_text().splitlines()
    rows = []
    for ln in lines:
        if ln.startswith("| ") and not ln.startswith("| Ataque"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            name = {"gaussian_noise": "Ruído gaussiano", "ipm": "IPM", "lie": "LIE", "minmax": "MinMax",
                    "minsum": "MinSum", "label_flip": "Label flip", "backdoor_pixel": "Backdoor (pixel)",
                    "backdoor_pgd": "Backdoor (PGD)", "mimic": "Mimic *", "lf": "LF *", "dba": "DBA *", "lp": "LP *",
                    "edge_case_pgd": "Edge-case PGD *", "média (desenho)": "Média (8 do desenho)",
                    "média (novos)": "Média (5 recentes)"}.get(cells[0], cells[0])
            vals = [num(c) for c in cells[1:]]
            # ordem no summary.md: AUC (RAGNAR-4, proposta, +conluio), Recall (idem), FP (idem)
            rows.append([name, " / ".join(vals[0:3]), " / ".join(vals[3:6]), " / ".join(vals[6:9])])
    return rows


# ----------------------------------------------------------------------------- conteúdo
def content():
    B = []
    A = B.append
    A(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:b/>'
      f'<w:i w:val="0"/><w:sz w:val="29"/></w:rPr><w:t>Relatório de Progresso referente ao mês 03</w:t></w:r></w:p>')
    A(h1("Especificação dos Sinais Observáveis e da Lógica de Detecção de Ataques em Aprendizado Federado", center=True))
    A(empty()); A(info_table()); A(empty())

    # 1 ---------------------------------------------------------------
    A(h1("1. Objetivo e Escopo"))
    A(p("Este relatório é o terceiro entregável do plano de 12 meses. O primeiro sistematizou ataques e defesas em "
        "Aprendizado Federado (FL) e o segundo validou o ambiente `fl-robustness-lab` e definiu o cenário de avaliação. "
        "O plano pede, para o mês 3, a **especificação dos sinais e da lógica de detecção**: o que o agente observa, como "
        "transforma essas observações numa decisão por cliente e como justifica essa decisão. O protótipo do agente é o "
        "entregável do mês 5 e a integração ao fluxo de agregação, o do mês 6."))
    A(p("O ponto de partida é o RAGNAR [Morais et al. 2024], a defesa proposta pelo grupo: um filtro pré-agregação que "
        "calcula normas e similaridades das atualizações e as classifica com um XGBoost treinado em simulações prévias. "
        "O agente aqui especificado é uma evolução do RAGNAR, e os sinais e a lógica foram escolhidos a partir da medição "
        "de por que a versão atual falha."))
    A(p("Neste mês também foram implementados os cinco ataques recentes da revisão que faltavam no laboratório (Mimic, "
        "DBA, LP, LF e Edge-case PGD; seção 3). Como a lógica de detecção já estava congelada quando eles foram "
        "implementados, eles funcionam como um **teste fora da amostra** da regra."))
    A(h2("1.1. Entregável desta Etapa"))
    A(bullet("**Este documento** — catálogo de sinais (seção 2), ataques recentes (seção 3), evidência de quanto cada "
             "sinal separa maliciosos de honestos (seção 4), especificação da lógica de detecção (seção 5) e sua "
             "avaliação preliminar (seção 6)."))
    A(bullet("**Código** em `fl-robustness-lab/`: `src/signals.py` (29 sinais, com a direção esperada de cada um "
             "declarada em `DIRECTION`), `src/attacks/recent.py` (5 ataques recentes, com testes), "
             "`experiments/signal_analysis.py` (medição dos sinais), `experiments/detection_logic.py` (a regra de decisão "
             "aplicada aos sinais coletados) e `experiments/attack_effect.py` (efeito de cada ataque sem defesa)."))

    A(h2("1.2. Modelo de Observação e de Ameaça"))
    A(p("A cada round t, o servidor recebe de cada cliente i a atualização Δᵢ = wᵢ − w, em que w é o modelo global que "
        "ele mesmo enviou. Todo sinal do agente é função de apenas quatro fontes: (i) as atualizações do round; (ii) o "
        "modelo global; (iii) um **holdout limpo** Dₛ no servidor (100 amostras rotuladas, o mesmo pressuposto do "
        "FLTrust); e (iv) o **histórico** dos clientes em rounds anteriores. O servidor não vê os dados dos clientes."))
    A(bullet("Até 20% dos clientes são maliciosos (10% no backdoor por pixel, no PGD, no LP e no Edge-case), fixos ao "
             "longo do treino e em conluio."))
    A(bullet("Os ataques de modelo são *oniscientes*: o atacante vê as atualizações honestas do round e sintetiza a sua "
             "(IPM, LIE, MinMax/MinSum, Mimic), como no BOBA e no Blades."))
    A(bullet("O atacante **não** conhece o detector; o caso adaptativo é o objeto do mês 10."))
    A(bullet("Saída do agente, por cliente e round: uma **decisão** (aceitar ou excluir da agregação), um **escore** "
             "contínuo e uma **explicação** (os sinais que motivaram a decisão). Os aceitos são agregados por FedAvg."))

    # 2 ---------------------------------------------------------------
    A(h1("2. Catálogo de Sinais"))
    A(p("Cada sinal é um escalar por cliente por round. Notação: Δ̄ é a média das atualizações do round, Δ̃ a mediana "
        "coordenada a coordenada e cos(a, b) o cosseno. A coluna *Dir.* traz a direção esperada para um malicioso, "
        "**declarada antes de ver os dados** (↑ valor maior, ↓ valor menor). A coluna *Ataques* indica em quantos dos "
        "13 ataques o sinal já foi medido."))
    cat = [
        ("l1/l2/l3_mean", "Geometria (RAGNAR v1)", "‖Δᵢ − Δ̄‖ₚ, p = 1, 2, 3", "atualizações", "↑", "13"),
        ("cos_mean", "Geometria (RAGNAR v1)", "cos(Δᵢ, Δ̄)", "atualizações", "↓", "13"),
        ("cos_global", "Geometria (RAGNAR v1)", "cos(Δᵢ, w)", "atualizações, modelo", "↓", "13"),
        ("norm", "Geometria robusta", "‖Δᵢ‖₂", "atualizações", "↑", "13"),
        ("l2_median, cos_median", "Geometria robusta", "distância e cosseno à mediana Δ̃", "atualizações", "↑ / ↓", "13"),
        ("mahalanobis", "Geometria robusta", "Mahalanobis de (l1, l2, cos) à média", "atualizações", "↑", "13"),
        ("krum_score", "Geometria robusta", "soma das n−f−2 menores distâncias² (Krum)", "atualizações", "↑", "13"),
        ("max_cos_peer", "Conluio", "maior cosseno com outro cliente do round", "atualizações", "↑", "13"),
        ("fc1_l2/cos_median", "Camada crítica", "l2_median e cos_median só nos pesos de fc1", "atualizações", "↑ / ↓", "13"),
        ("holdout_loss_gain", "Referência no servidor", "ℓ(w + Δᵢ; Dₛ) − ℓ(w; Dₛ)", "holdout", "↑", "13"),
        ("holdout_acc", "Referência no servidor", "acurácia de w + Δᵢ em Dₛ", "holdout", "↓", "13"),
        ("fltrust_cos", "Referência no servidor", "cos(Δᵢ, Δₛ), Δₛ = treino do servidor em Dₛ", "holdout", "↓", "13"),
        ("hist_cos_self", "Histórico", "cosseno com a própria atualização anterior", "histórico", "↑", "13"),
        ("hist_max_cos_cum", "Histórico", "max_cos_peer sobre a soma histórica (FoolsGold)", "histórico", "↑", "13"),
        ("hist_ema_l2_median", "Histórico", "média móvel exponencial do z de l2_median", "histórico", "↑", "13"),
        ("l2/cos_geomed", "Robusto (v2)", "distância e cosseno à mediana geométrica (ruptura 50%)", "atualizações", "↑ / ↓", "5"),
        ("mcd_mahalanobis", "Robusto (v2)", "Mahalanobis com média/covariância do MCD", "atualizações", "↑", "5"),
        ("spectral_score", "Robusto (v2)", "projeção² no 1º vetor singular (assinaturas espectrais)", "atualizações", "↑", "5"),
        ("sign_agreement", "Robusto (v2)", "fração de coordenadas com o sinal de Δ̃ (SignGuard)", "atualizações", "↓", "5"),
        ("holdout_max_class_gain, holdout_class_excess, holdout_js", "Robusto (v2)", "piora de perda por classe em Dₛ; divergência JS", "holdout", "↑", "5"),
        ("cusum_cos_median, cusum_l2_median", "Robusto (v2)", "CUSUM de |z| de cos_median e l2_median", "histórico", "↑", "5"),
    ]
    A(table(["Sinal", "Família", "Definição", "Requer", "Dir.", "Ataques"], cat,
            [1900, 1600, 3060, 1300, 550, 950], size=16))

    # 3 ---------------------------------------------------------------
    A(h1("3. Ataques Recentes Implementados"))
    A(p("A revisão (entregável 01) cobre ataques mais recentes do que os 8 avaliados até o mês 2. Os cinco que faltavam "
        "foram implementados em `src/attacks/recent.py`, seguindo o mecanismo de cada artigo."))
    A(table(["Ataque", "Referência", "Mecanismo implementado", "Simplificação"], [
        ["Mimic", "Karimireddy et al., ICLR 2022", "os 4 atacantes enviam, em todos os rounds, a atualização real do honesto "
         "que mais se projeta na direção de maior variância dos honestos", "nenhuma relevante"],
        ["DBA", "Xie et al., ICLR 2020", "gatilho global 4×4 dividido em 4 faixas 1×4; cada atacante usa só a sua; "
         "avaliação com o gatilho completo", "4 atacantes fixos (no artigo, rodízio)"],
        ["LP", "Zhuang et al., ICLR 2024", "treino limpo e com backdoor; camadas críticas achadas por substituição de "
         "camadas; envia o delta limpo com só essas camadas envenenadas", "sem o ajuste adaptativo de λ"],
        ["LF", "Zhuang et al., ICLR 2024", "versão não direcionada: inverte a direção do delta só nas camadas mais "
         "críticas para a acurácia", "idem"],
        ["Edge-case PGD", "Wang et al., NeurIPS 2020", "aviões raros (extremo do 1º componente principal: aviões "
         "escuros) rotulados como caminhão, retirados dos honestos; mistura 1:1 com dados limpos; PGD + escala ×3 com "
         "norma limitada", "modo raro do próprio CIFAR-10 em vez de imagens externas"],
    ], [1300, 1900, 4060, 2100], size=16, bold_first_col=True))
    A(h2("3.1. Os ataques funcionam?"))
    A(p("Cada ataque foi executado sem defesa (FedAvg, 40 rounds, semente 42). A taxa de sucesso dos ataques "
        "direcionados (ASR) é comparada com a de um modelo limpo, treinado sem atacantes. Asterisco: ataques recentes."))
    A(table(["Ataque", "Maliciosos", "MA (%)", "ΔMA (p.p.)", "ASR (%)", "ASR base (%)"], effect_rows(),
            [2560, 1360, 1360, 1360, 1360, 1360], size=18))
    A(bullet("**LP e DBA são backdoors funcionais e furtivos.** O LP chega a 50% de sucesso (base 2,5%) sem tocar a "
             "acurácia; as camadas críticas escolhidas foram quase sempre `fc1` e a última convolução (64 e 61 de 80 "
             "escolhas), o mesmo padrão do artigo. O DBA chega a 42%, com cada atacante treinando só um quarto do gatilho."))
    A(bullet("**Mimic e LF degradam a acurácia** em 3–4 pontos, na mesma faixa de LIE, MinMax e MinSum."))
    A(bullet("**O Edge-case é fraco nesta configuração**: 8% contra 2% da base (4 contra 1 das 50 imagens raras do "
             "teste), embora fique consistentemente acima da base ao longo do treino. Foram três versões documentadas "
             "(cauda radial: 4%; modo raro coerente com peso declarado reduzido por engano: 2%; peso corrigido: 8%). Com "
             "amostras raras da mesma distribuição dos aviões honestos e uma CNN pequena, 2 atacantes em 20 não "
             "sustentam a exceção; o artigo usa imagens externas e modelos maiores."))

    # 4 ---------------------------------------------------------------
    A(h1("4. Evidência: Quanto Cada Sinal Separa Maliciosos de Honestos"))
    A(h2("4.1. Protocolo"))
    A(bullet("**Cenário**: o do entregável 02 (CIFAR-10, CNN pequena, 20 clientes, Dirichlet α = 0,5, 1 época local, "
             "holdout de 100 amostras, semente 42), com 15 rounds por ataque e os 13 ataques."))
    A(bullet("**Agregação oráculo**: o modelo global é atualizado só com os honestos, para que todos os ataques sejam "
             "observados sobre a mesma trajetória; mede-se a *detectabilidade*, não o estrago."))
    A(bullet("**Métrica**: AUC-ROC calculada *dentro de cada round* (a decisão que o servidor toma), com média entre "
             "os rounds, **na direção declarada a priori**. Abaixo de 0,5, o ataque inverte o sinal."))
    A(box("Correção em relação à versão apresentada em 28/09", [
        "A tabela anterior reportava a separabilidade max(AUC, 1 − AUC), que escolhe a direção do sinal para cada ataque "
        "**depois** de ver o resultado — algo que um detector real não pode fazer. Ela superestimava os sinais: "
        "cos_median aparecia com 0,90 de média e, na direção declarada, fica em torno de 0,5."]))
    A(h2("4.2. Resultados"))
    A(image(RES / "signals" / "auc_heatmap.png"))
    A(caption("Figura 1 — AUC por sinal e ataque, na direção declarada. Azul: o sinal detecta; vermelho: o ataque "
              "inverte o sinal. As cinco colunas antes da média são os ataques recentes."))
    A(image(RES / "signals" / "auc_bilateral_heatmap.png"))
    A(caption("Figura 2 — AUC do escore bilateral |z| (desvio da mediana do round em qualquer sentido, sem direção)."))
    A(image(RES / "signals" / "auc_heatmap_v2.png", 5.2))
    A(caption("Figura 3 — Sinais v2 (formulações robustas), medidos nos 5 ataques recentes."))
    A(h2("4.3. Achados"))
    A(bullet("**Nenhum sinal isolado cobre todos os ataques.** No pior caso entre os 13, só mahalanobis fica acima de "
             "0,5 (0,52, no Mimic). Até max_cos_peer, o melhor em média (0,86), se inverte no LF (0,07). O detector "
             "precisa *combinar* famílias."))
    A(bullet("**Ataques de modelo se escondem no centro.** LIE, MinMax e MinSum deixam o malicioso *mais* alinhado à "
             "mediana que um honesto típico (cos_median 0,00 na direção declarada); o IPM faz o oposto (1,00). Lido como "
             "desvio bilateral (“diferente demais” ou “típico demais”), o cosseno sobe para média 0,84, pior caso 0,57."))
    A(bullet("**Normas devem ser lidas numa direção só.** No escore bilateral elas pioram (l2_median 0,08 no MinMax); "
             "falham no PGD e no Edge-case, que limitam a norma, e no IPM."))
    A(bullet("**A referência no servidor cobre o que a geometria não pega**: o ganho de perda no holdout separa "
             "perfeitamente PGD, LIE e Edge-case, mas se inverte no IPM (0,02)."))
    A(bullet("**Os sinais de conluio são fortes, mas frágeis**: dependem de atacantes parecidos entre si. O LF já os "
             "inverte, e um atacante adaptativo (mês 10) os evitaria. A lógica é avaliada com e sem eles."))
    A(bullet("**O label flip continua sem sinal confiável** (melhor: hist_cos_self, 0,76). Os sinais por classe no "
             "holdout chegaram a 1,00 no Edge-case, que também afeta uma classe — candidatos a medir no label flip."))
    A(bullet("**A regra muda com a fase do treino.** No backdoor por pixel, cos_mean tem AUC 0,21 nos rounds 2–7 e "
             "0,87 nos rounds 8–15. Um classificador treinado nos primeiros rounds (como o RAGNAR v1) e testado contra "
             "ataque não visto: sem normalização, AUC 0,48 e 44% de falsos positivos; com as mesmas features "
             "normalizadas no round, AUC 0,68 e 3%; com 10 sinais normalizados, AUC 0,90. A **normalização dentro do "
             "round** é requisito da lógica."))
    A(bullet("**Cada ataque recente escapa de uma família diferente.** Mimic escapa dos cossenos e do holdout (pego "
             "por norma, conluio e espectral, 0,97); LF escapa das normas e do conluio (pego pelos cossenos, 0,92–0,98); "
             "LP e DBA são invisíveis no holdout (pegos por normas e dispersão, 0,94–0,97); Edge-case escapa das normas "
             "(pego pelo holdout, 1,00)."))

    # 5 ---------------------------------------------------------------
    A(h1("5. Lógica de Detecção"))
    A(p("A lógica segue diretamente dos achados. Ela é especificada como uma regra **sem treino**, auditável, que é a "
        "versão 0 do agente."))
    A(h2("5.1. Etapas"))
    A(p("**Etapa 1 — Extração.** Para cada cliente, os sinais do conjunto ativo (tabela abaixo). Um sinal indisponível, "
        "como o histórico no round 1, não acusa ninguém."))
    A(p("**Etapa 2 — Normalização no round.** O z-score robusto independe da escala do ambiente e da fase do treino; "
        "com mediana e MAD, os atacantes não deslocam a referência enquanto forem menos da metade:"))
    A(formula("zₖ(i, t) = [ sₖ(i, t) − medianaⱼ sₖ(j, t) ] / [ 1,4826 · MADⱼ sₖ(j, t) ]"))
    A(p("**Etapa 3 — Evidência por sinal.** Cada sinal tem um modo fixo: unilateral, eₖ = max(0, dₖ · zₖ), com dₖ a "
        "direção declarada; ou bilateral, eₖ = |zₖ|."))
    A(p("**Etapa 4 — Escore e explicação.** S(i, t) = maxₖ eₖ(i, t). O sinal que atinge o máximo, com seu z, é a "
        "explicação principal; os demais com eₖ > τ entram como evidência secundária."))
    A(p("**Etapa 5 — Acúmulo entre rounds (CUSUM)**, para sinais fracos mas persistentes. Descontar a mediana de S é "
        "necessário porque o máximo de vários |z| de um honesto já passa de 1:"))
    A(formula("C(i, t) = max( 0, C(i, t − 1) + S(i, t) − medianaⱼ S(j, t) − κ )"))
    A(p("**Etapa 6 — Decisão.** O cliente é excluído se S > τ ou C > h, com no máximo ⌊(n − 1)/2⌋ exclusões por round "
        "(as de maior escore). A regra **nunca** “aceita todos” quando muitos são marcados e atua **desde o round 1** — "
        "as duas escolhas corrigem a falha do RAGNAR v1 no ruído gaussiano, em que o round 1 sem filtro destrói o "
        "modelo e, depois, todos parecem anômalos e são aceitos."))
    A(table(["Sinal", "Família", "Modo", "Justificativa"], [
        ["norm", "geometria", "unilateral ↑", "ruído e atualizações escaladas"],
        ["mahalanobis", "geometria", "unilateral ↑", "maior pior caso entre os sinais sem conluio"],
        ["cos_median", "geometria robusta", "bilateral", "IPM (oposto) e LIE/MinMax/MinSum (“típico demais”)"],
        ["fc1_cos_median", "camada crítica", "bilateral", "idem, na camada que concentra o backdoor"],
        ["holdout_loss_gain", "servidor", "unilateral ↑", "backdoor PGD e LIE"],
        ["holdout_acc", "servidor", "unilateral ↓", "estável entre ataques"],
        ["hist_cos_self", "histórico", "bilateral", "persistência (backdoor) e aleatoriedade (ruído)"],
        ["max_cos_peer, hist_max_cos_cum", "conluio", "unilateral ↑", "só na variante “+ conluio”"],
    ], [2400, 1700, 1500, 3760], size=18))
    A(caption("Tabela — Sinais e modos da regra proposta. Parâmetros fixados a priori: τ = 3, κ = 0,5, h = 6."))
    A(h2("5.2. Variante com classificador (RAGNAR evoluído)"))
    A(p("A etapa 4 pode ser substituída por um classificador treinado sobre os zₖ normalizados (e não sobre features "
        "absolutas, como no RAGNAR original), com dados que cubram **todas as fases** do treino. Contra ataque não "
        "visto, essa variante chegou a AUC 0,90 com 2–5% de falsos positivos. As etapas 5 e 6 e a explicação valem "
        "para as duas variantes; a escolha entre elas é do mês 5."))

    # 6 ---------------------------------------------------------------
    A(h1("6. Avaliação Preliminar da Lógica"))
    A(p("A regra foi aplicada aos sinais já coletados, sem novo treino; o cálculo roda em segundos "
        "(script `detection_logic.py`). A linha de base é a mesma regra usando só as 4 features do RAGNAR v1 (RAGNAR-4). Recall e falsos "
        "positivos (FP) são frações dos pares (cliente, round) excluídos. Cada célula traz RAGNAR-4 / proposta / "
        "proposta + conluio."))
    A(table(["Ataque", "AUC", "Recall", "FP"], detection_rows(), [2760, 2200, 2200, 2200], size=18))
    A(p("**Nos 8 ataques usados no desenho da regra:**"))
    A(bullet("Proposta contra RAGNAR-4: AUC 0,95 contra 0,78 e recall 0,86 contra 0,37, com 9% de falsos positivos "
             "contra 5%. O maior ganho está onde o RAGNAR-4 não detecta nada: LIE, MinMax e MinSum (recall 0 → "
             "0,87–0,93) e backdoor PGD (0,10 → 1,00)."))
    A(bullet("Sem os sinais de conluio a proposta perde pouco (recall 0,86 contra 0,90)."))
    A(bullet("O CUSUM ainda não se justificou: só o escore instantâneo dá recall 0,82 com 5% de falsos positivos; o "
             "acúmulo acrescenta 4 pontos de recall ao custo de 4 de falsos positivos. Fica como componente a validar "
             "nos experimentos longos (meses 7–9)."))
    A(p("**Nos 5 ataques recentes (fora da amostra):**"))
    A(bullet("A vantagem se mantém, com custo maior: proposta AUC 0,88, recall 0,72, 15% de falsos positivos; "
             "proposta + conluio AUC 0,93, recall 0,85; RAGNAR-4 AUC 0,70, recall 0,34. A queda em relação aos 8 "
             "ataques do desenho mostra quanto daqueles números vinha de desenhar e avaliar com os mesmos dados."))
    A(bullet("O Mimic é o caso mais difícil sem conluio (recall 0,60 com 29% de falsos positivos): o próprio cliente "
             "imitado se destaca tanto quanto os atacantes. O DBA escapa da proposta sem conluio (recall 0,38; 0,70 com "
             "conluio). LF, LP e Edge-case são detectados (recall 0,77–1,00)."))
    A(bullet("As explicações mudam de família conforme o ataque (fc1_cos_median no LF e no Mimic, hist_cos_self e "
             "mahalanobis no LP, ganho de perda no holdout no Edge-case), como se espera de uma regra que combina "
             "famílias."))
    A(box("Como ler estes números", [
        "São **evidência de viabilidade, não validação**. Nos 8 ataques do desenho, o modo de cada sinal foi escolhido "
        "com os mesmos dados; os 5 recentes não têm esse viés, mas usam a mesma semente e o mesmo cenário. A agregação "
        "é oráculo e os runs têm 15 rounds. A validação exige outra semente e o agente atuando na agregação (meses 6–9)."]))

    # 7 ---------------------------------------------------------------
    A(h1("7. Contexto: o RAGNAR v1 com o Filtro Ativo"))
    A(p("O RAGNAR v1 do laboratório (4 features, XGBoost treinado em 5 rounds simulados a partir do modelo inicial) foi "
        "avaliado com o filtro ativo, em 40 rounds, contra os 8 ataques originais:"))
    A(bullet("**Com a família do ataque excluída do treino** (ataque não visto), não protege: recall 0 nos ataques de "
             "modelo; nos backdoors exclui cerca de metade dos honestos, a acurácia cai de 72% para 57–58% e o backdoor "
             "fica mais forte (75% → 94% no ataque por pixel)."))
    A(bullet("**Com todos os ataques no treino** (braço `ragnar_pool_todos`), a detecção melhora (F1 0,38–0,55 nos "
             "ataques de modelo), mas a acurácia final fica abaixo do FedAvg sem defesa em todos os ataques, porque a "
             "maioria dos honestos é excluída nos rounds finais."))
    A(p("O quadro é coerente com os achados: features absolutas lidas numa só direção, com um classificador treinado "
        "numa fase do treino e aplicado em outra. *Nota de rastreabilidade*: o braço `ragnar_pool_todos` estava rotulado "
        "por engano como “ataque visto” até 29/09. A comparação com o RAGNAR do artigo (CoUrb-2024, 54 features, "
        "classificador pré-treinado) está em andamento em `courb_eval/`."))

    # 8 ---------------------------------------------------------------
    A(h1("8. Limitações e Próximos Passos"))
    A(bullet("**Uma semente e um cenário** (CIFAR-10, α = 0,5); label skew severo (α = 0,1) ainda não foi testado."))
    A(bullet("**Sinais v2 medidos só nos ataques recentes.** Medi-los nos 8 originais exige repetir aquela coleta "
             "(cerca de 60 minutos na CPU)."))
    A(bullet("**Edge-case fraco nesta configuração**; uma versão com imagens raras externas ao CIFAR-10 fica para o mês 4."))
    A(bullet("**O label flip precisa de uma métrica de efeito** (não altera a acurácia global); será acrescentada a "
             "acurácia da classe de origem no mês 4."))
    A(bullet("**O limiar fixo não controla os falsos positivos**: calibrar τ por predição conforme é o objeto do mês 9."))
    A(bullet("**Os ataques não são adaptativos**; os sinais de conluio são os primeiros que um atacante adaptativo "
             "contornaria (mês 10)."))
    A(p("**Encadeamento**: M4 — ataques injetáveis (13 implementados; faltam a métrica do label flip, a correção do "
        "ruído gaussiano e um Edge-case mais forte); M5 — protótipo do agente com a lógica da seção 5, nas duas "
        "variantes; M6 — agente integrado à agregação, com o filtro ativo desde o round 1."))

    # refs ------------------------------------------------------------
    A(h1("Referências Bibliográficas"))
    for ref in [
        "[Morais et al. 2024] Morais, M. et al. RAGNAR: Lightweight Defense against Poisoning Attacks in Federated LLMs and IoT. CoUrb, 2024.",
        "[Cao et al. 2021] Cao, X. et al. FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. NDSS, 2021.",
        "[Xie et al. 2019] Xie, C.; Koyejo, O.; Gupta, I. Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. UAI, 2019.",
        "[Baruch et al. 2019] Baruch, G.; Baruch, M.; Goldberg, Y. A Little Is Enough: Circumventing Defenses for Distributed Learning. NeurIPS, 2019.",
        "[Shejwalkar e Houmansadr 2021] Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. NDSS, 2021.",
        "[Karimireddy et al. 2022] Karimireddy, S. P.; He, L.; Jaggi, M. Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing. ICLR, 2022.",
        "[Xie et al. 2020] Xie, C.; Huang, K.; Chen, P.-Y.; Li, B. DBA: Distributed Backdoor Attacks against Federated Learning. ICLR, 2020.",
        "[Zhuang et al. 2024] Zhuang, H. et al. Backdoor Federated Learning by Poisoning Backdoor-Critical Layers. ICLR, 2024.",
        "[Wang et al. 2020] Wang, H. et al. Attack of the Tails: Yes, You Really Can Backdoor Federated Learning. NeurIPS, 2020.",
        "[Blanchard et al. 2017] Blanchard, P. et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS, 2017.",
        "[Fung et al. 2020] Fung, C.; Yoon, C. J. M.; Beschastnikh, I. The Limitations of Federated Learning in Sybil Settings. RAID, 2020.",
        "[Rousseeuw e Van Driessen 1999] A Fast Algorithm for the Minimum Covariance Determinant Estimator. Technometrics, 1999.",
        "[Tran et al. 2018] Tran, B.; Li, J.; Madry, A. Spectral Signatures in Backdoor Attacks. NeurIPS, 2018.",
        "[Xu et al. 2022] Xu, J. et al. SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering. ICDCS, 2022.",
        "[Page 1954] Page, E. S. Continuous Inspection Schemes. Biometrika, 1954.",
    ]:
        A(p(ref, size=20))
    return "".join(B)


# ----------------------------------------------------------------------------- montagem
def main():
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(tmp)
    doc = (tmp / "word" / "document.xml").read_text(encoding="utf8")
    head, rest = doc.split("<w:body>", 1)
    sect = rest[rest.index("<w:sectPr"):rest.index("</w:body>")]
    body = content()
    (tmp / "word" / "document.xml").write_text(f"{head}<w:body>{body}{sect}</w:body></w:document>", encoding="utf8")

    rels_path = tmp / "word" / "_rels" / "document.xml.rels"
    rels = rels_path.read_text(encoding="utf8")
    add = "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                  f'Target="media/{name}"/>' for rid, name, _ in IMAGES)
    rels_path.write_text(rels.replace("</Relationships>", add + "</Relationships>"), encoding="utf8")
    for _, name, src in IMAGES:
        shutil.copy(src, tmp / "word" / "media" / name)

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        ct = tmp / "[Content_Types].xml"
        z.write(ct, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ct:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(OUT)],
                   check=True, capture_output=True, timeout=300)
    print(OUT, "+ .pdf")


if __name__ == "__main__":
    main()
