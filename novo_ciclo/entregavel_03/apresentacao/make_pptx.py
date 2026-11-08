#!/usr/bin/env python
"""Gera apresentacao.pptx a partir do template H.IAAC (apresentação_2025.pptx).

Usa os layouts do próprio template (capa com onda e logos, seção com ilustração, conteúdo com
caixa azul-marinho) e preenche o conteúdo da apresentação do mês 3. Tabelas são nativas do
PowerPoint, montadas a partir dos resultados em fl-robustness-lab/results; figuras vêm de
apresentacao/figuras (geradas por experiments/make_slides_assets.py).

    ../fl-robustness-lab/.venv/bin/python make_pptx.py
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # novo_ciclo/ (a apresentação fica em entregavel_03/apresentacao)
TEMPLATE = ROOT / "apresentação_2025.pptx"
OUT = HERE / "apresentacao.pptx"
sys.path.insert(0, str(ROOT / "fl-robustness-lab" / "experiments"))
from make_slides_assets import ATTACK_LABEL, ATTACK_ORDER, ragnar_eval_rows  # noqa: E402

NAVY = RGBColor(0x07, 0x1F, 0x4A)
ORANGE = RGBColor(0xFF, 0x75, 0x19)
GRAY = RGBColor(0x9E, 0x9E, 0x9E)
LIGHT = RGBColor(0xF3, 0xF3, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
FONT = "Roboto"
SCALE = 1.3  # o slide do template tem 20" de largura

L_TITLE, L_SECTION, L_BODY, L_BODY_SUB = "TITLE_3_1", "SECTION_HEADER", "TITLE_AND_BODY", "TITLE_AND_BODY_3"


# ------------------------------------------------------------------ template
def open_template():
    prs = Presentation(str(TEMPLATE))
    icon = None
    for slide in prs.slides:  # guarda o ícone vetorial "Aprendizado Distribuído" do slide de áreas
        for sh in slide.shapes:
            if sh.shape_type == 6 and "DISTRIBU" in "".join(t.text for t in _texts(sh)).upper():
                icon = copy.deepcopy(sh._element)
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):  # remove os slides de exemplo, mantendo layouts e mestre
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)
    return prs, icon


def _texts(shape):
    if shape.shape_type == 6:
        for s in shape.shapes:
            yield from _texts(s)
    elif shape.has_text_frame:
        yield shape.text_frame


def layout(prs, name):
    return next(l for l in prs.slide_layouts if l.name == name)


def ph(slide, idx):
    return next((p for p in slide.placeholders if p.placeholder_format.idx == idx), None)


def drop_empty_placeholders(slide, keep=()):
    for p in list(slide.placeholders):
        if p.placeholder_format.idx not in keep and not p.text_frame.text.strip():
            p._element.getparent().remove(p._element)


def add_slide_number(slide):
    tb = slide.shapes.add_textbox(Inches(18.53), Inches(10.2), Inches(1.2), Inches(0.6))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld"), id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", type="slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr"), lang="pt-BR", sz="1000")
    fill = etree.SubElement(rpr, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr"), val="9E9E9E")
    etree.SubElement(fld, qn("a:t")).text = "‹#›"


# ------------------------------------------------------------------ texto
def rich(paragraph, text, size, color=BLACK, bold=False):
    """Escreve `text` no parágrafo; trechos entre ** ficam em negrito, entre ^^ em laranja."""
    parts = text.replace("^^", "\x00").split("**")
    for i, part in enumerate(parts):
        for j, seg in enumerate(part.split("\x00")):
            if not seg:
                continue
            r = paragraph.add_run()
            r.text = seg
            r.font.size, r.font.name = Pt(size * SCALE), FONT
            r.font.bold = bold or (i % 2 == 1)
            r.font.color.rgb = ORANGE if j % 2 == 1 else color


def textbox(slide, x, y, w, h, lines, size=20, color=BLACK, bullets=True, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """lines: str ou (nivel, str). Nível 0 com marcador laranja, nível 1 com marcador cinza."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for k, line in enumerate(lines):
        level, text = line if isinstance(line, tuple) else (0, line)
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(size * SCALE * 0.4)
        if bullets and text:
            _bullet(p, level, size)
        rich(p, text, size - 3 * level, color)
    return tb


def _bullet(p, level, size):
    pPr = p._p.get_or_add_pPr()
    indent = int(Inches(0.35 + 0.4 * level))
    pPr.set("marL", str(indent))
    pPr.set("indent", str(-int(Inches(0.3))))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr"), val="FF7519" if level == 0 else "9E9E9E")
    etree.SubElement(pPr, qn("a:buSzPct"), val="80000")
    etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
    etree.SubElement(pPr, qn("a:buChar"), char="■")


def callout(slide, x, y, w, h, title, lines, head=NAVY, size=18):
    """Caixa com cabeçalho colorido (equivalente aos blocos da versão LaTeX)."""
    top = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.6))
    top.fill.solid(); top.fill.fore_color.rgb = head; top.line.fill.background()
    tf = top.text_frame
    tf.margin_left = Inches(0.15)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    rich(tf.paragraphs[0], title, size, WHITE, bold=True)
    body = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.6), Inches(w), Inches(h - 0.6))
    body.fill.solid(); body.fill.fore_color.rgb = LIGHT; body.line.fill.background()
    textbox(slide, x + 0.1, y + 0.7, w - 0.2, h - 0.8, lines, size=size, bullets=len(lines) > 1)


def table(slide, x, y, w, rows, col_w=None, size=16, header=True, row_h=0.45, highlight=None, num_cols=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for child in list(tblPr):
        tblPr.remove(child)  # sem estilo padrão do PowerPoint
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    if col_w:
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_head = header and i == 0
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if is_head else (LIGHT if highlight is not None and i == highlight else WHITE)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j in num_cols else PP_ALIGN.LEFT
            rich(p, str(val), size, WHITE if is_head else BLACK, bold=is_head)
            _cell_border(cell, bottom=not is_head)
    return gt


def _numeric(v):
    s = str(v).replace(",", ".").replace("%", "").replace("--", "").strip()
    try:
        float(s); return True
    except ValueError:
        return s == ""


def _cell_border(cell, bottom=True):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag), w="6350" if (tag == "a:lnB" and bottom) else "0")
        if tag == "a:lnB" and bottom:
            f = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(f, qn("a:srgbClr"), val="D9D9D9")
        else:
            etree.SubElement(ln, qn("a:noFill"))


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def center_picture(slide, path, y, max_w, max_h, slide_w=20.0):
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    return picture(slide, path, (slide_w - w) / 2, y, w=w, h=h)


# ------------------------------------------------------------------ tipos de slide
def title_slide(prs, title, subtitle, author_lines, icon=True):
    s = prs.slides.add_slide(layout(prs, L_TITLE))
    tf = ph(s, 0).text_frame
    tf.text = ""
    rich(tf.paragraphs[0], title, 34, WHITE, bold=True)
    if subtitle:
        p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        rich(p, subtitle, 17, WHITE)
    tf = ph(s, 1).text_frame
    for k, line in enumerate(author_lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        rich(p, line, 14, WHITE)
    if icon:
        picture(s, HERE / "identidade" / "icone_aprendizado_distribuido.png", 0.35, 0.3, h=1.1)
    return s


def section(prs, title, subtitle):
    s = prs.slides.add_slide(layout(prs, L_SECTION))
    ph(s, 0).text = title
    ph(s, 1).text = subtitle
    return s


def content(prs, title, subtitle=None):
    s = prs.slides.add_slide(layout(prs, L_BODY_SUB if subtitle else L_BODY))
    t = ph(s, 0)
    t.text = ""
    # a caixa azul do layout tem largura fixa (~10,9"): reduz a fonte até o título caber em uma linha
    size = min(40.0, 10.0 * 72 / (0.52 * max(len(title), 1)))
    rich(t.text_frame.paragraphs[0], title, size / SCALE, WHITE, bold=True)
    if subtitle:
        ph(s, 3).text = subtitle
    drop_empty_placeholders(s, keep=(0, 3))
    add_slide_number(s)
    return s


# ------------------------------------------------------------------ dados
def ragnar_table_rows():
    df = ragnar_eval_rows()
    get = lambda a, arm, col: (df[(df.attack == a) & (df.arm == arm)][col].iloc[0]
                               if len(df[(df.attack == a) & (df.arm == arm)]) else None)
    f = lambda v, pct=True: "--" if v is None or pd.isna(v) else (f"{v:.1f}".replace(".", ",") if pct else f"{v:.2f}".replace(".", ","))
    rows = [["Ataque", "MA FedAvg", "MA RAGNAR v1", "BA FedAvg", "BA RAGNAR v1", "Recall", "Falsos pos."]]
    for a in ATTACK_ORDER:
        if len(df[df.attack == a]):
            rows.append([ATTACK_LABEL[a], f(get(a, "fedavg", "ma")), f(get(a, "ragnar_unseen", "ma")),
                         f(get(a, "fedavg", "ba")), f(get(a, "ragnar_unseen", "ba")),
                         f(get(a, "ragnar_unseen", "recall"), False), f(get(a, "ragnar_unseen", "fpr"), False)])
    clean = df[df.attack == "clean"]
    return rows, (clean.ma.iloc[0] if len(clean) else None)


def cross_phase_rows():
    base = ROOT / "fl-robustness-lab" / "results"
    src = next((p for p in (base / "signals_v2" / "cross_phase_classifier.csv",
                            base / "signals" / "cross_phase_classifier.csv") if p.exists()), None)
    cp = pd.read_csv(src)
    cp = cp.rename(columns={"norm": "normalizacao"}) if "norm" in cp.columns else cp
    m = cp[cp.protocolo == "nao_visto"].groupby(["sinais", "normalizacao"], sort=False)[["auc", "recall", "fpr"]].mean()
    lab = {"raw": "nenhuma", "robust_z": "z robusto", "rank": "rank"}
    rows, prev = [["Sinais", "Normalização", "AUC", "Recall", "Falsos pos."]], None
    for (s, n), r in m.iterrows():
        c = lambda v: f"{v:.2f}".replace(".", ",")
        rows.append([s if s != prev else "", lab.get(n, n), c(r.auc), c(r.recall), c(r.fpr)])
        prev = s
    return rows


# ------------------------------------------------------------------ apresentação
def build():
    import build_slides  # o conteúdo dos slides e as notas ficam em build_slides.py
    build_slides.build()


if __name__ == "__main__":
    build()
