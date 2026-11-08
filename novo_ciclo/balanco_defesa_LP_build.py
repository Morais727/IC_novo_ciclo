"""Monta a apresentação de balanço (defesa contra o LP) no modelo H.IAAC de apresentacao.pptx."""
import copy
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TPL, OUT, LOGO = sys.argv[1], sys.argv[2], sys.argv[3]
NAVY, ORANGE, GRAY, MUTED, BAR = "071F4A", "FF7519", "F3F3F3", "5A6878", "A3ADB8"
FONT = "Roboto"

prs = Presentation(TPL)
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids):
    prs.part.drop_rel(sid.get(qn("r:id")))
    sld_ids.remove(sid)

L_COVER, L_SECTION, L_TITLE, L_SUB = (prs.slide_layouts[i] for i in range(4))
num = [0]


def rgb(h):
    return RGBColor.from_string(h)


def style_run(r, size, color="000000", bold=False, italic=False):
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = rgb(color)


def bulletize(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "320040"); pPr.set("indent", "-274320")
    for tag, attrs in (("a:buClr", None), ("a:buSzPct", {"val": "80000"}), ("a:buFont", {"typeface": "Arial"}), ("a:buChar", {"char": "■"})):
        el = etree.SubElement(pPr, qn(tag), **(attrs or {}))
        if tag == "a:buClr":
            etree.SubElement(el, qn("a:srgbClr"), val=ORANGE)


def fill_tf(tf, paras, size, color="000000", bullets=False, space=8, align=PP_ALIGN.LEFT):
    """paras: lista de str ou de listas de (texto, negrito)."""
    tf.word_wrap = True
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(space)
        runs = [(para, False)] if isinstance(para, str) else para
        for text, b in runs:
            style_run(p.add_run() if True else None, size, color, b) if False else None
            r = p.add_run(); r.text = text
            style_run(r, size, color, b)
        if bullets:
            bulletize(p)


def text(s, x, y, w, h, paras, size=22, color="000000", bullets=False, space=8, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.vertical_anchor = anchor
    fill_tf(tf, paras, size, color, bullets, space, align)
    return tb


def rect(s, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = rgb(fill)
    r.line.fill.background(); r.shadow.inherit = False
    return r


def header(s, x, y, w, label, color=NAVY, size=22):
    r = rect(s, x, y, w, 0.6, color)
    tf = r.text_frame; tf.margin_left = Inches(0.15); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_tf(tf, [[(label, True)]], size, "FFFFFF")


def card(s, x, y, w, h, label, paras, color=NAVY, size=22, bullets=False):
    header(s, x, y, w, label, color)
    rect(s, x, y + 0.6, w, h - 0.6, GRAY)
    text(s, x + 0.1, y + 0.7, w - 0.2, h - 0.8, paras, size, bullets=bullets)


def message(s, y, label, body, color=NAVY, h=1.5):
    card(s, 0.8, y, 18.4, h, label, [body], color, size=22)


def table(s, x, y, w, rows, widths, size=20, hl=None, row_h=0.55):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shp.table
    tblPr = t._tbl.tblPr
    for k in ("firstRow", "bandRow"):
        tblPr.set(k, "0")
    for j, f in enumerate(widths):
        t.columns[j].width = Inches(w * f)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = rgb(NAVY)
            elif i == hl:
                c.fill.fore_color.rgb = rgb("FFE3CF")
            else:
                c.fill.fore_color.rgb = rgb(GRAY if i % 2 == 0 else "FFFFFF")
            tf = c.text_frame
            tf.text = ""
            r = tf.paragraphs[0].add_run(); r.text = val
            style_run(r, size, "FFFFFF" if i == 0 else "000000", i == 0 or i == hl)
    return shp


def slide_number(s):
    num[0] += 1
    tb = s.shapes.add_textbox(Inches(18.53), Inches(10.2), Inches(1.2), Inches(0.6))
    p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(num[0])
    style_run(r, 14, MUTED)


def clean_placeholders(s, keep):
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx not in keep:
            ph._element.getparent().remove(ph._element)


def set_ph(s, idx, value, size=None):
    tf = s.placeholders[idx].text_frame
    tf.text = value
    if size:
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.size = Pt(size)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def content(title, sub=None, note=None):
    s = prs.slides.add_slide(L_SUB if sub else L_TITLE)
    set_ph(s, 0, title)
    if sub:
        set_ph(s, 3, sub, 26)
    clean_placeholders(s, {0, 3} if sub else {0})
    slide_number(s)
    if note:
        notes(s, note)
    return s


def section(title, sub, note=None):
    s = prs.slides.add_slide(L_SECTION)
    set_ph(s, 0, title); set_ph(s, 1, sub)
    num[0] += 1
    if note:
        notes(s, note)
    return s


def footnote(s, t):
    text(s, 0.8, 9.95, 17.0, 0.5, [t], 14, MUTED)


# ------------------------------------------------------------------ capa
s = prs.slides.add_slide(L_COVER)
set_ph(s, 0, "Defesa contra o ataque de envenenamento por camadas no aprendizado federado")
tf = s.placeholders[1].text_frame
tf.text = "Balanço dos experimentos"
for line in ("Aluno: Marcos Gilliard Oliveira Morais", "Orientador: Leandro A. Villas", "H.IAAC · Outubro de 2026"):
    tf.add_paragraph().text = line
for p_ in tf.paragraphs:
    for r_ in p_.runs:
        r_.font.size = Pt(18)
s.shapes.add_picture(LOGO, Inches(0.35), Inches(0.3), Inches(3.12), Inches(1.1))
num[0] += 1
notes(s, "Objetivo: mostrar de forma direta o que foi feito, onde a proposta funciona, onde falha e o que ela tem de novo.")

# ------------------------------------------------------------------ roteiro
s = content("O que vamos mostrar")
items = [("O problema", "um ataque que se esconde em poucas camadas do modelo e passa pelas defesas atuais"),
         ("A nossa proposta", "como funciona e o que tem de novo frente aos trabalhos mais próximos"),
         ("Como avaliamos", "o que foi rodado e os critérios de sucesso, fixados antes"),
         ("Onde ela funciona", "contra o ataque LP e contra ataques que destroem o modelo"),
         ("Onde ela falha", "e o que ainda está em aberto"),
         ("Decisões", "para discutirmos")]
for i, (a, b) in enumerate(items):
    y = 2.6 + i * 1.15
    c = rect(s, 0.8, y, 0.85, 0.85, ORANGE if a == "Onde ela falha" else NAVY, MSO_SHAPE.OVAL)
    c.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_tf(c.text_frame, [[(str(i + 1), True)]], 24, "FFFFFF", align=PP_ALIGN.CENTER)
    text(s, 1.9, y + 0.12, 17.0, 0.7, [[(a + ": ", True), (b, False)]], 26)

# ------------------------------------------------------------------ seção 1
section("O problema e a proposta", "Um ataque que se esconde em poucas camadas e como detectá-lo")

s = content("O problema", "Um cliente pode esconder uma porta dos fundos na sua atualização",
            "No aprendizado federado o servidor nunca vê os dados, só as atualizações. Isso abre espaço para um cliente malicioso implantar uma porta dos fundos.")
steps = [("1. Clientes treinam", "Cada cliente treina o modelo com os próprios dados, que nunca saem dele"),
         ("2. Enviam a atualização", "Só a alteração nos pesos do modelo vai para o servidor"),
         ("3. Servidor junta", "Combina as atualizações e devolve o modelo comum; repete por várias rodadas")]
for i, (a, b) in enumerate(steps):
    x = 0.8 + i * 6.35
    card(s, x, 3.5, 5.5, 2.9, a, [b], size=24)
    if i < 2:
        rect(s, x + 5.6, 4.7, 0.65, 0.45, BAR, MSO_SHAPE.RIGHT_ARROW)
message(s, 7.0, "A ameaça", "Um cliente malicioso envia uma atualização que implanta uma porta dos fundos: o modelo continua acertando as imagens normais, mas, quando a imagem tem um gatilho escolhido pelo atacante, responde o que ele quer.", ORANGE, 2.3)

s = content("O ataque LP (1/2)", "Como o atacante prepara a atualização envenenada",
            "LP vem de Layer-wise Poisoning, envenenamento por camadas (Zhuang e colegas, ICLR 2024). A ideia central: a porta dos fundos mora em poucas camadas do modelo. "
            "O atacante descobre quais são, altera só essas e copia o resto de um treino limpo. No passo 4, metade/metade é o valor dos autores para redes pequenas (lambda = 0,5).")
prep = [("1. Treina duas versões", "Uma limpa, com os dados normais, e uma com a porta dos fundos: metade das imagens recebe o gatilho e o rótulo que o atacante quer"),
        ("2. Acha as camadas críticas", "Troca, uma por vez, cada camada do modelo envenenado pela do limpo. Se a porta dos fundos some, aquela camada é crítica"),
        ("3. Fica com o mínimo", "Escolhe o menor grupo de camadas críticas que ainda reproduz 95% da porta dos fundos; em geral, 1 a 6 das 10 camadas"),
        ("4. Monta e testa", "Envia o modelo limpo com só essas camadas misturadas ao envenenado (metade de cada). Simula a defesa; se for barrado, tira camadas até passar")]
for i, (a, b) in enumerate(prep):
    card(s, 0.8 + (i % 2) * 9.5, 3.4 + (i // 2) * 2.95, 8.9, 2.6, a, [b], ORANGE if i == 1 else NAVY, 22)
text(s, 0.8, 9.3, 18.4, 0.6, ["O nosso atacante é mais forte que o do artigo: no passo 4 ele conhece as atualizações dos clientes honestos da rodada."], 18, MUTED)
footnote(s, "Zhuang e colegas, “Backdoor Federated Learning by Poisoning Backdoor-Critical Layers”, ICLR 2024")

s = content("O ataque LP (2/2)", "Por fora a atualização parece normal; o veneno está em poucas camadas",
            "As três versões existem para testar a defesa de forma honesta: a evasiva e a com ruído foram criadas por nós para atacar exatamente o sinal que a nossa proposta usa.")
for col, (label, bad) in enumerate((("Cliente honesto", ()), ("Atacante LP", (3, 4)))):
    x = 0.9 + col * 3.3
    text(s, x, 3.3, 2.9, 0.5, [[(label, True)]], 20, align=PP_ALIGN.CENTER)
    for k in range(6):
        rect(s, x, 3.9 + k * 0.75, 2.9, 0.58, ORANGE if k in bad else BAR, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 8.4, 2.9, 0.5, ["camadas do modelo"], 16, MUTED, align=PP_ALIGN.CENTER)
text(s, 0.9, 8.9, 6.2, 1.0, [[("Laranja: ", True), ("camadas envenenadas. Cinza: igual a um treino limpo", False)]], 16, MUTED)
card(s, 7.8, 3.3, 11.4, 2.2, "Por que engana as defesas comuns", ["Quase todo o modelo enviado é legítimo. Medida no cliente inteiro, a alteração envenenada fica diluída e o atacante parece um cliente normal."], NAVY, 21)
table(s, 7.8, 5.85, 11.4, [
    ["Versão testada", "O que o atacante faz"],
    ["Original (dos autores)", "Todos os atacantes enviam exatamente a mesma atualização"],
    ["Evasiva (nossa)", "Cada atacante parte do próprio treino limpo; só as camadas críticas têm parte em comum"],
    ["Com ruído (nossa)", "Evasiva + ruído aleatório nas camadas críticas, para disfarçar a semelhança entre atacantes"]],
    [0.3, 0.7], 18, row_h=0.95)

s = content("A nossa proposta", "Detectar, camada por camada, atacantes que agem em conjunto",
            "Ideia central em uma frase. Os próximos slides mostram como funciona e o que muda em relação aos trabalhos existentes.")
r = rect(s, 0.8, 3.6, 18.4, 3.4, NAVY)
r.text_frame.margin_left = r.text_frame.margin_right = Inches(0.6); r.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
fill_tf(r.text_frame, [[("Em cada camada, procuramos clientes com alterações parecidas demais entre si ou grandes demais. O cliente suspeito sai só daquela camada; o resto da atualização dele continua contando.", False)]], 30, "FFFFFF")
card(s, 0.8, 7.4, 8.9, 2.3, "Por que por camada", ["O LP concentra o ataque em poucas camadas: olhando o cliente inteiro, a alteração fica diluída."], size=22)
card(s, 10.3, 7.4, 8.9, 2.3, "Por que em conjunto", ["Os atacantes do LP enviam a mesma alteração nas mesmas camadas, então ficam parecidos demais entre si."], ORANGE, size=22)

s = content("Como funciona", "A cada rodada e em cada camada, quatro passos")
passos = [("1. Retirar o que é comum", "Subtrai de cada atualização a mediana de todos os clientes; sobra o que é particular de cada um"),
          ("2. Medir a semelhança", "Quanto a camada do cliente se parece com a de algum outro. Atacantes coordenados enviam alterações quase iguais"),
          ("3. Medir o tamanho", "O tamanho da alteração na camada. Ruído usado para disfarçar a semelhança aumenta o tamanho"),
          ("4. Decidir e juntar", "Se o cliente foge muito do grupo nas duas medidas somadas, fica fora daquela camada; ela vira a média dos demais")]
for i, (a, b) in enumerate(passos):
    card(s, 0.8 + (i % 2) * 9.5, 3.4 + (i // 2) * 2.95, 8.9, 2.6, a, [b], size=22)
text(s, 0.8, 9.3, 18.4, 0.6, ["“Foge muito” = mais de 3 desvios típicos acima da mediana do grupo, calculados de forma resistente a valores extremos."], 18, MUTED)

s = content("Nosso diferencial", "Três contribuições, separadas do que já existe",
            "A ideia de decidir por camada não é nossa: a LASA já faz isso. O que defendemos como novo é o sinal, a primeira avaliação séria das defesas recentes contra o LP e o mapa de onde todas falham.")
rect(s, 0.8, 3.35, 18.4, 0.9, "E4E4E4")
text(s, 1.0, 3.42, 18.0, 0.8, [[("Já existe, não reivindicamos: ", True), ("excluir o cliente só da camada suspeita (LASA, 2025) · comparar clientes por camada (RFLDefense, 2026) · defesa voltada ao LP (FedSurrogate, 2026)", False)]], 18, anchor=MSO_ANCHOR.MIDDLE)
contrib = [("1. Um sinal novo", [[("O que é: ", True), ("em cada camada, a maior semelhança do cliente com outro, somada ao tamanho da alteração, comparadas com o resto do grupo", False)],
                                 [("Por que é novo: ", True), ("a LASA olha só o próprio cliente; o RFLDefense usa limiar fixo de semelhança e não mede tamanho", False)],
                                 [("Evidência: ", True), ("com α = 0,5, 12 de 12 casos; LASA e AlignIns, 3 de 12; RFLDefense, 0 de 6", False)]]),
           ("2. Uma avaliação que faltava", [[("O que é: ", True), ("6 defesas recentes frente a 3 versões do LP, com α de 0,1 a 0,5", False)],
                                             [("Por que é novo: ", True), ("só a FedSurrogate tinha sido testada contra o LP, e só com α = 0,5", False)],
                                             [("Evidência: ", True), ("458 treinos; 6 sementes, 3 reservadas só para o veredito", False)]]),
           ("3. Um mapa dos limites", [[("O que é: ", True), ("onde todas as defesas param de funcionar: α = 0,1, 4 ou mais atacantes, atacante que conhece a defesa", False)],
                                       [("Por que importa: ", True), ("diz quando confiar numa defesa; os artigos originais não mostram onde falham", False)],
                                       [("Evidência: ", True), ("com α = 0,1, nenhuma defesa testada segurou o LP", False)]])]
for i, (a, paras) in enumerate(contrib):
    x = 0.8 + i * 6.25
    header(s, x, 4.55, 5.9, a)
    rect(s, x, 5.15, 5.9, 4.65, GRAY)
    text(s, x + 0.1, 5.25, 5.7, 4.5, paras, 19, space=12)
footnote(s, "Busca bibliográfica feita em 21 e 25 de outubro de 2026")

s = content("Trabalhos mais próximos", "Usam outro sinal, e quase nenhum foi testado contra o LP")
table(s, 0.8, 3.45, 18.4, [
    ["Trabalho", "O que mede em cada camada", "O que faz com o suspeito", "Testado contra o LP?", "Contra o LP, aqui"],
    ["LASA (2025)", "Tamanho e coerência de sinais do próprio cliente; não compara clientes", "Exclui só a camada", "Não", "3 de 12 casos"],
    ["RFLDefense (2026)", "Conta quantos clientes passam de um limiar fixo de semelhança", "Reduz o peso", "Não", "0 de 6 casos"],
    ["FedSurrogate (2026)", "Agrupa clientes nas camadas críticas", "Troca as camadas pelas de outro cliente", "Sim, só α = 0,5", "Falha com α = 0,1 e perde 10 a 20 pontos"],
    ["Nossa proposta", "Semelhança máxima com outro cliente e tamanho, relativos ao grupo", "Exclui só a camada", "Sim", "12 de 12 casos"]],
    [0.15, 0.3, 0.19, 0.14, 0.22], 18, hl=4, row_h=0.85)
message(s, 8.15, "Por que o sinal combinado", "Os atacantes do LP enviam a mesma alteração nas camadas críticas e ficam parecidos demais; se usam ruído para disfarçar, a alteração fica grande demais.", h=1.65)
footnote(s, "Casos = α 0,5 e 0,3, sementes de teste") if False else None

# ------------------------------------------------------------------ seção 2
section("Como avaliamos", "O que foi rodado e como decidimos se a defesa funcionou")

s = content("Medidas usadas", "Como lemos os resultados")
table(s, 0.8, 3.6, 18.4, [
    ["Termo", "Significado"],
    ["Taxa de sucesso do ataque", "Porcentagem de imagens com gatilho que o modelo classifica como o atacante quer (quanto menor, melhor)"],
    ["Acurácia", "Porcentagem de acertos do modelo em imagens normais (quanto maior, melhor)"],
    ["α (alfa)", "Mede o quão diferentes são os dados entre os clientes (parâmetro da distribuição de Dirichlet usada para dividi-los). Quanto menor, mais diferentes. Usamos 0,5, 0,3 e 0,1"],
    ["Sem defesa", "Junção das atualizações por média simples (média federada); serve de linha de base"],
    ["Semente", "Número que fixa os sorteios do treino (divisão dos dados, pesos iniciais); cada semente é uma execução diferente"]],
    [0.26, 0.74], 20, row_h=1.0)

s = content("O que foi avaliado", "6 defesas publicadas comparadas nas mesmas condições",
            "A comparação é justa porque cada defesa roda com o mesmo ataque, a mesma semente e a mesma divisão dos dados; só muda a regra que junta as atualizações. As defesas concorrentes foram reimplementadas a partir dos artigos.")
for i, (n, d) in enumerate((("458", "treinos completos, cerca de 158 horas de placa de vídeo"), ("20", "clientes por treino, 2 deles atacantes, 40 rodadas"), ("6", "sementes: 3 para ajustar, 3 só para o veredito"))):
    x = 0.8 + i * 6.25
    rect(s, x, 3.4, 5.9, 2.4, GRAY)
    text(s, x + 0.2, 3.45, 5.5, 1.2, [[(n, True)]], 54, ORANGE)
    text(s, x + 0.2, 4.7, 5.5, 1.0, [d], 20)
table(s, 0.8, 6.25, 18.4, [
    ["O que variamos", "Valores"],
    ["Defesa", "Nossa proposta, sem defesa (média simples), LASA, AlignIns, FLAME, Multi-Krum, FedSurrogate, RFLDefense"],
    ["Ataque", "LP em 3 versões, atacante que conhece a defesa, 3 outras portas dos fundos, 3 ataques que destroem o modelo"],
    ["Dados", "Fotos coloridas (CIFAR-10) com rede pequena e com rede residual; fotos de roupas (Fashion-MNIST)"]],
    [0.18, 0.82], 20, row_h=0.8)

s = content("Critério de sucesso", "Fixado antes de rodar, sem ajustes depois de ver os resultados")
for i, (a, b) in enumerate((("Ataque contido", "Taxa de sucesso do ataque de no máximo 10% ao final do treino"),
                            ("Modelo preservado", "Acurácia no máximo 3 pontos abaixo da média federada, com e sem ataque"),
                            ("Caso válido", "Só conta se o ataque funciona sem defesa (taxa de pelo menos 20%)"))):
    card(s, 0.8 + i * 6.25, 3.6, 5.9, 3.2, a, [b], size=24)
message(s, 7.4, "Sementes", "3 sementes de desenvolvimento (42, 44, 45) para ajustar a proposta; 3 sementes de teste (46, 47, 48) usadas só para o veredito final.", h=1.6)

# ------------------------------------------------------------------ seção 3
section("Onde ela funciona", "Contra o ataque LP e contra ataques que destroem o modelo")

s = content("Visão geral", "Somando todos os treinos, é a que mais segura o ataque LP",
            "Este gráfico é um panorama. Como mistura cenários diferentes, a comparação justa, defesa contra defesa no mesmo cenário, está no próximo slide.")
cd = CategoryChartData()
cats = ["Multi-Krum", "FLAME", "Sem defesa", "LASA", "AlignIns", "Nossa proposta"]
cd.categories = cats
cd.add_series("Mediana", (74, 49, 49, 32, 22, 4))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(3.3), Inches(18.4), Inches(4.4), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Taxa de sucesso do ataque no treino típico (mediana, %; quanto menor, melhor)"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 18, MUTED)
pl = ch.plots[0]; pl.gap_width = 45
pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = '0"%"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(18); dl.font.name = FONT
ser = pl.series[0]
for k, cat in enumerate(cats):
    pt = ser.points[k]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(ORANGE if cat == "Nossa proposta" else BAR)
ca, va = ch.category_axis, ch.value_axis
ca.tick_labels.font.size = Pt(18); ca.tick_labels.font.name = FONT
va.visible = False; va.has_major_gridlines = False; va.maximum_scale = 85; va.minimum_scale = 0
message(s, 7.95, "Mensagem", "Sem defesa, metade das imagens com gatilho cai na classe do atacante; com a nossa proposta, 4%. Ela conteve o ataque em 68% dos treinos; a segunda melhor, em 43%. FLAME e Multi-Krum não ajudam ou pioram.", h=1.85)
footnote(s, "Inclui cenários fáceis e difíceis; o número de treinos varia por defesa (a nossa: 95)")

s = content("Comparação direta", "No mesmo cenário, defendeu todos os casos; as concorrentes, no máximo 1/4",
            "LASA e AlignIns (2025) nunca tinham sido testadas contra o LP nos artigos originais; este é o primeiro teste delas contra esse ataque, até onde a busca chegou.")
text(s, 0.8, 3.25, 18.4, 0.9, ["Mesmo ataque, mesmas sementes de teste, α = 0,5 (primeira linha: 0,5 e 0,3); muda só a defesa. Defendido = ataque com até 10% de sucesso, sem perder mais de 3 pontos de acurácia."], 18, MUTED)
table(s, 0.8, 4.2, 18.4, [
    ["Comparação", "Nossa proposta", "Concorrente"],
    ["Contra LASA e AlignIns (2 versões do LP, α 0,5 e 0,3, 3 sementes)", "12 de 12", "LASA 3 de 12 · AlignIns 3 de 12"],
    ["Contra RFLDefense, LP com ruído", "3 de 3", "0 de 3: segura parte do ataque (4% a 13%), mas perde 4 a 7 pontos de acurácia"],
    ["Contra RFLDefense, LP evasivo", "3 de 3", "0 de 3: ataque continua entre 67% e 78%"],
    ["LP na versão original dos autores", "3 de 3", "—"]],
    [0.38, 0.17, 0.45], 20, row_h=0.75)
message(s, 8.2, "Mensagem", "LASA e AlignIns, ambas de 2025, deixam o LP passar em 3 de cada 4 casos. A nossa segura todos e, sem ataque, tem a mesma acurácia que o treino sem defesa.", h=1.6)

MEDIA_HDR = ["α", "Sem defesa", "Nossa proposta", "LASA", "AlignIns", "RFLDefense"]
MEDIA_W = [0.08, 0.15, 0.2, 0.19, 0.19, 0.19]
MEDIA_NOTE = "Média das sementes de teste 46, 47 e 48. Célula: sucesso médio do ataque (perda de acurácia frente ao treino sem defesa, em pontos). ✓ = a média cumpre o critério: até 10% e até 3 pontos de perda. — = ainda não rodado."
s = content("Média por α: LP com ruído", "De α = 0,3 a 0,5, só a nossa proposta segura o ataque na média",
            "Médias calculadas só nas sementes de teste, porque nem toda defesa rodou nas sementes de ajuste; misturar daria vantagem a quem tem mais sementes. "
            "A média esconde a variação: a LASA com α = 0,5 vai de 1,5% a 63% entre sementes. Por isso o critério do artigo continua sendo semente por semente.")
text(s, 0.8, 3.2, 18.4, 0.9, [MEDIA_NOTE], 16, MUTED)
table(s, 0.8, 4.05, 18.4, [MEDIA_HDR,
    ["0,5", "55,9%", "3,9% (−0,6) ✓", "42,2% (−1,6)", "72,0% (−2,2)", "6,9% (−5,1)"],
    ["0,4", "35,5%", "3,0% (−0,1) ✓", "—", "—", "—"],
    ["0,3", "45,3%", "3,8% (0,0) ✓", "37,3% (−1,9)", "26,1% (−1,6)", "—"],
    ["0,2", "57,0%", "15,3% (−1,0)", "—", "—", "—"],
    ["0,1", "72,9%", "66,2% (−0,7)", "77,1% (−8,3)", "81,1% (−12,6)", "1,8% (−38,4)*"]],
    MEDIA_W, 20, row_h=0.6)
message(s, 7.95, "Mensagem", "De α = 0,3 a 0,5 a nossa fica abaixo de 4% sem perder acurácia; LASA e AlignIns ficam entre 26% e 72%. O RFLDefense segura o ataque, mas perde acurácia. Com α = 0,2 a nossa já não alcança o critério (15%).", h=1.85)
footnote(s, "* RFLDefense com α = 0,1: só 2 sementes; segura o ataque, mas a acurácia cai para 28%")

s = content("Média por α: LP evasivo", "Contra o LP evasivo, a nossa segura de α = 0,2 a 0,5",
            "Na média, a LASA cumpre o critério com α = 0,5 e a AlignIns com α = 0,3, mas por pouco. Semente por semente, a LASA passa do limite de acurácia na semente 48.")
text(s, 0.8, 3.2, 18.4, 0.9, [MEDIA_NOTE], 16, MUTED)
table(s, 0.8, 4.05, 18.4, [MEDIA_HDR,
    ["0,5", "58,9%", "2,0% (−0,6) ✓", "2,9% (−2,8) ✓", "15,7% (−3,4)", "73,2% (−5,6)"],
    ["0,4", "32,4%", "1,9% (−1,0) ✓", "—", "—", "—"],
    ["0,3", "45,1%", "2,0% (−0,1) ✓", "12,4% (−1,8)", "3,6% (−2,3) ✓", "—"],
    ["0,2", "57,0%", "3,5% (−0,7) ✓", "—", "—", "—"],
    ["0,1", "70,9%", "34,7% (−1,0)", "84,1% (−9,7)", "76,2% (−6,4)", "81,3% (−32,5)*"]],
    MEDIA_W, 20, row_h=0.6)
message(s, 7.95, "Mensagem", "A nossa é a única que cumpre o critério em todos os α de 0,2 a 0,5. Com α = 0,1 ninguém cumpre, mas a nossa corta o ataque pela metade (71% → 35%) sem custo, enquanto as outras não reduzem e perdem de 6 a 33 pontos.", h=1.85)
footnote(s, "* RFLDefense com α = 0,1: só 1 semente")

s = content("Repetição entre sementes", "O resultado se repetiu em todas as sementes de teste")
rect(s, 0.8, 3.4, 8.9, 3.9, GRAY)
text(s, 1.0, 3.45, 8.5, 1.4, [[("100%", True)]], 66, ORANGE)
text(s, 1.0, 4.9, 8.5, 2.3, [[("Nas 3 sementes de teste, com α = 0,5, a proposta conteve ", False), ("todas as versões do LP: 9 de 9 casos", True), (", com sucesso do ataque entre 1,3% e 7,7%", False)]], 22)
card(s, 10.3, 3.4, 8.9, 3.9, "Por que isso importa", ["Com uma semente só, nós mesmos teríamos errado: na semente 42, com α = 0,1, a proposta derrubou o ataque de 68% para 3%; nas outras 5 sementes, falhou."], ORANGE, 22)
message(s, 7.75, "Mensagem", "Um resultado de semente única pode ser sorte. Muitos trabalhos da área relatam uma única execução por cenário; nós separamos 3 sementes para ajustar e 3 só para o veredito, e é esse protocolo que sustenta o 100%.", h=1.95)

s = content("Um exemplo", "O ataque não muda a acurácia; só a taxa de sucesso mostra quem defendeu")
text(s, 0.8, 3.25, 18.4, 0.6, ["Um treino: fotos coloridas, 20 clientes, 2 atacantes com LP com ruído, α = 0,5, semente 46. Resultado após 40 rodadas."], 18, MUTED)
table(s, 0.8, 3.95, 18.4, [
    ["Defesa", "Acurácia", "Sucesso do ataque"],
    ["Sem ataque (referência)", "71,2%", "—"],
    ["Sem defesa", "71,3%", "50,8% (falhou)"],
    ["Nossa proposta", "70,6%", "2,1% (conteve)"],
    ["LASA", "70,3%", "1,5% (conteve)"],
    ["AlignIns", "70,3%", "76,0% (falhou)"]],
    [0.4, 0.3, 0.3], 22, hl=3, row_h=0.6)
message(s, 7.95, "Mensagem", "Todos têm acurácia de cerca de 71%, então quem olha só a acurácia não vê o ataque. Sem defesa, metade das imagens com gatilho é desviada; com a nossa, 2%. Aqui a LASA também conteve, mas não se repetiu nas outras sementes.", h=1.85)

s = content("Ataques de destruição", "O atacante troca sua atualização por uma fabricada para estragar o modelo")
atqs = [("Ruído aleatório", [[("Envia: ", True), ("números sorteados ao acaso, sem treinar nada", False)], [("Efeito: ", True), ("os valores são grandes e, na média, soterram as atualizações honestas; o modelo passa a chutar", False)]]),
        ("Inversão da média (IPM)", [[("Envia: ", True), ("a média das atualizações dos honestos com o sinal trocado", False)], [("Efeito: ", True), ("puxa o modelo no sentido contrário ao aprendizado; a cada rodada, desfaz parte do que os honestos ensinaram", False)]]),
        ("Desvios pequenos (LIE)", [[("Envia: ", True), ("a média dos honestos deslocada 1,5 desvio-padrão em cada peso", False)], [("Efeito: ", True), ("cada valor parece normal e passa pelos filtros, mas todos empurram para o mesmo lado e o erro se acumula", False)]])]
for i, (a, paras) in enumerate(atqs):
    x = 0.8 + i * 6.25
    header(s, x, 3.4, 5.9, a, ORANGE)
    rect(s, x, 4.0, 5.9, 3.5, GRAY)
    text(s, x + 0.1, 4.1, 5.7, 3.3, paras, 20, space=12)
message(s, 7.85, "Por que a nossa proposta resiste", "Na inversão e nos desvios pequenos, os 4 atacantes enviam exatamente a mesma coisa e ficam parecidos demais; o ruído aleatório tem tamanho muito maior que o normal. Os dois casos são o que o nosso sinal procura.", h=1.95)
footnote(s, "IPM: Xie e colegas, UAI 2019 · LIE (“A Little Is Enough”): Baruch e colegas, NeurIPS 2019")

s = content("Destruição: resultados", "Só a nossa proposta manteve a acurácia em todos")
text(s, 0.8, 3.25, 18.4, 0.6, ["4 atacantes de 20, α = 0,1. Medida: acurácia final (quanto maior, melhor)."], 18, MUTED)
table(s, 0.8, 3.95, 18.4, [
    ["Ataque", "Sem defesa", "Multi-Krum", "FLAME", "Nossa proposta"],
    ["Nenhum (referência)", "68,7%", "53,2%", "40,8%", "68,8%"],
    ["Ruído aleatório", "10,0%", "62,6%", "46,7%", "67,2%"],
    ["Inversão da média (IPM)", "65,4%", "36,6%", "13,1%", "66,1%"],
    ["Desvios pequenos (LIE)", "64,5%", "44,6%", "47,0%", "66,9%"]],
    [0.36, 0.16, 0.16, 0.16, 0.16], 22, row_h=0.65)
message(s, 7.6, "Mensagem", "A nossa perde no máximo 2,7 pontos. Multi-Krum e FLAME descartam clientes honestos inteiros e perdem até 55 pontos; sem defesa, o ruído destrói o modelo (10%, igual a chutar).", h=1.75)
footnote(s, "Uma semente (42); falta repetir em outras sementes")

# ------------------------------------------------------------------ seção 4
section("Onde ela falha", "Limites encontrados e o que ainda está em aberto")

s = content("Com α = 0,1", "Com α = 0,1, ela falha, como todas as outras defesas")
text(s, 0.8, 3.25, 18.4, 0.6, ["Mesmos ataques LP (com ruído e evasivo) e mesmas 3 sementes de teste, mas com α = 0,1 em vez de 0,5."], 18, MUTED)
table(s, 0.8, 3.9, 18.4, [
    ["Defesa", "Casos defendidos", "Sucesso médio: com ruído / evasivo", "Acurácia média"],
    ["Sem defesa", "—", "72,9% / 70,9%", "66%"],
    ["Nossa proposta", "0 de 6", "66,2% / 34,7%", "66%"],
    ["LASA", "0 de 6", "77,1% / 84,1%", "57% a 58%"],
    ["AlignIns", "0 de 6", "81,1% / 76,2%", "54% a 60%"],
    ["FLAME", "0 de 3", "3,5% / não rodado", "38%"],
    ["Multi-Krum", "0 de 3", "82,5% / não rodado", "53%"],
    ["RFLDefense", "0 de 3", "1,8% / 81,3%", "28% a 34%"]],
    [0.22, 0.18, 0.38, 0.22], 20, hl=2, row_h=0.5)
message(s, 8.15, "Mensagem", "Com α = 0,1 os honestos já são muito diferentes entre si e os atacantes se escondem no grupo: ninguém cumpre o critério. A nossa ainda corta o LP evasivo pela metade sem perder acurácia; quem segura o ataque (FLAME, RFLDefense) destrói o modelo.", ORANGE, h=1.65)

s = content("Fora do padrão do LP", "Também falha com mais atacantes ou outros tipos de porta dos fundos")
falhas = [("Atacante que conhece a defesa", "Ajusta o ataque para ficar abaixo do limiar de suspeita: defendidos só 2 de 6 casos"),
          ("Mais atacantes", "4 de 20: 0 de 3 casos. 6 de 20: 0 de 6, quase sem redução do ataque"),
          ("Porta dos fundos otimizada (PGD)", "Gatilho treinado com limite de tamanho da alteração: 0 de 3, sucesso entre 20% e 63%"),
          ("Gatilho de pixels e distribuído", "Cada atacante treina sozinho, sem alteração em comum: não contidos")]
for i, (a, b) in enumerate(falhas):
    card(s, 0.8 + (i % 2) * 9.5, 3.4 + (i // 2) * 2.2, 8.9, 1.95, a, [b], ORANGE, 20)
message(s, 7.95, "Mensagem", "A proposta procura o padrão do LP: poucos atacantes com a mesma alteração nas mesmas camadas. Fora desse padrão ela não tem vantagem. Não é uma defesa geral contra porta dos fundos, e o artigo deve dizer isso.", ORANGE, h=1.85)

s = content("Em aberto", "Ainda não sabemos se vale para outros dados e ataques mais novos")
card(s, 0.8, 3.4, 8.9, 3.7, "Outros dados e redes", ["Em fotos de roupas e na rede residual, o LP não funcionou nem sem defesa (sucesso abaixo de 20%). Dobrar a força do ataque não resolveu. Sem ataque que funcione, não há o que defender."], size=22)
card(s, 10.3, 3.4, 8.9, 3.7, "Ataques mais novos", ["Dois ataques de 2026 também envenenam só camadas sensíveis (LaySelFL e LSA) e relatam passar pelas defesas atuais. Não foram testados contra a nossa."], size=22)
message(s, 7.6, "Mensagem", "Hoje o resultado vale para fotos coloridas (CIFAR-10) com uma rede pequena. Generalizar exige um ataque que funcione nos outros cenários e testar os ataques de 2026.", MUTED, h=1.7)

s = content("Próximo passo", "Avaliar todas as defesas nos mesmos cenários",
            "Grade completa: 8 formas de agregar × 5 valores de α × 4 cenários (LP com ruído, LP evasivo, LP original, sem ataque) × 3 sementes de teste = 480 treinos. "
            "178 já existem e são reaproveitados; faltam 302. Nenhum parâmetro muda. Fila pronta em experiments/fase7.py.")
text(s, 0.8, 3.2, 18.4, 0.6, ["Hoje: quantos dos 4 cenários (3 versões do LP + sem ataque) cada defesa tem completos, com as 3 sementes de teste."], 16, MUTED)
table(s, 0.8, 3.85, 11.6, [
    ["Defesa", "α 0,5", "α 0,4", "α 0,3", "α 0,2", "α 0,1"],
    ["Sem defesa", "4/4", "3/4", "3/4", "3/4", "4/4"],
    ["Nossa proposta", "4/4", "3/4", "3/4", "3/4", "4/4"],
    ["LASA", "3/4", "0/4", "3/4", "0/4", "3/4"],
    ["AlignIns", "3/4", "0/4", "3/4", "0/4", "3/4"],
    ["RFLDefense", "3/4", "0/4", "0/4", "0/4", "0/4"],
    ["FLAME", "0/4", "0/4", "0/4", "0/4", "1/4"],
    ["Multi-Krum", "0/4", "0/4", "0/4", "0/4", "1/4"],
    ["FedSurrogate", "0/4", "0/4", "0/4", "0/4", "1/4"]],
    [0.3, 0.14, 0.14, 0.14, 0.14, 0.14], 18, row_h=0.62)
rect(s, 12.9, 3.85, 6.3, 5.6, GRAY)
text(s, 13.1, 3.95, 5.9, 1.2, [[("302", True)]], 60, ORANGE)
text(s, 13.1, 5.15, 5.9, 4.2, [
    [("treinos faltando de 480", True)],
    "Cerca de 32 horas com 7 treinos em paralelo nesta máquina",
    "Reaproveita os 178 que já existem, sem mudar nenhum parâmetro",
    "Depois disso, toda tabela compara defesas nos mesmos cenários e sementes"], 20, space=14)

# ------------------------------------------------------------------ resumo
s = content("Em resumo")
card(s, 0.8, 2.6, 8.9, 3.0, "Funciona", ["Contra o LP com α de 0,3 a 0,5 (o evasivo até 0,2) e 2 atacantes: sucesso médio de 2% a 4%, sem perda de acurácia, melhor que as defesas publicadas. Também resiste a ataques que destroem o modelo."], NAVY, 22)
card(s, 10.3, 2.6, 8.9, 3.0, "Não funciona", ["α = 0,1, 4 ou mais atacantes, atacante que conhece a defesa e outros tipos de porta dos fundos."], ORANGE, 22)
header(s, 0.8, 6.0, 18.4, "Para decidir com o orientador")
rect(s, 0.8, 6.6, 18.4, 3.2, GRAY)
text(s, 0.9, 6.7, 18.2, 3.0, [
    [("1. ", True), ("Apresentar o artigo como defesa específica para o LP, com os limites declarados?", False)],
    [("2. ", True), ("Insistir nos outros cenários com um ataque mais forte, ou retirá-los do artigo?", False)],
    [("3. ", True), ("O diferencial delimitado (sinal, avaliação, mapa dos limites) é suficiente para um artigo?", False)],
    [("4. ", True), ("Rodar a grade completa (302 treinos, cerca de 32 horas) antes de fechar o artigo?", False)]], 22, space=12)

prs.save(OUT)
print("slides:", len(prs.slides))
