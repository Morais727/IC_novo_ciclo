#!/usr/bin/env python3
"""Monta apresentacao.html (deck autônomo) a partir de deck.json + slides/*.html — cópia local do artefato
https://claude.ai/artifact/Va8osHTBwuyrcbFB4Avwdc. Setas/espaço navegam; N mostra as notas; F tela cheia;
Ctrl+P imprime um slide por página. Uso: python3 build.py [--single ID] (um slide só, para gerar imagens)."""
import json, re, sys
from pathlib import Path
H = Path(__file__).resolve().parent
d = json.load(open(H / "deck.json"))
order = d["order"]
if "--single" in sys.argv:
    order = [sys.argv[sys.argv.index("--single") + 1]]
fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in d["faces"].values())
ICONS = {"CheckCircle": "✔", "Warning": "⚠"}
def fix(html):
    html = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>',
                  lambda m: f'<div style="{m.group(1)};background:none;display:flex;align-items:center;justify-content:center;font-size:44px;color:#B8B2A6;flex:none">→</div>', html)
    html = re.sub(r'<x-icon name="(\w+)" style="([^"]*)"></x-icon>',
                  lambda m: f'<span style="{m.group(2)};display:inline-flex;align-items:center;justify-content:center;font-size:34px;flex:none">{ICONS.get(m.group(1), "•")}</span>', html)
    return html
secs = "\n".join(fix((H / "slides" / f"{s}.html").read_text()) for s in order)
page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{d['title']}</title>{fonts}<style>
*{{box-sizing:border-box;margin:0}} html,body{{background:#0d1526;height:100%;overflow:hidden}}
#stage{{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform-origin:center}}
section{{position:absolute;inset:0;width:1920px;height:1080px;display:none;overflow:hidden}}
section.on{{display:flex!important}}
table{{border-collapse:collapse}} th,td{{padding:12px 16px;border-bottom:1px solid #E2DDD2;vertical-align:top}}
th{{color:#14213D;border-bottom:2px solid #14213D}} ol,ul{{padding-left:1.2em}}
aside{{display:none}} body.notes section.on aside{{display:block;position:absolute;left:0;right:0;bottom:0;background:rgba(13,21,38,.94);color:#F7F5F0;font:26px/1.45 'IBM Plex Sans',sans-serif;padding:28px 48px;z-index:9}}
#n{{position:fixed;right:16px;bottom:10px;color:#8090b0;font:14px sans-serif}}
@media print{{@page{{size:1920px 1080px;margin:0}} html,body{{overflow:visible;height:auto;background:none}} #stage{{position:static;transform:none!important}}
section{{position:relative;display:flex!important;page-break-after:always;break-after:page}} aside,#n{{display:none!important}}}}
</style></head><body><div id="stage">{secs}</div><div id="n"></div><script>
const S=[...document.querySelectorAll('section')];let i=Math.max(0,Math.min(S.length-1,(+location.hash.slice(1)||1)-1));
function fit(){{const k=Math.min(innerWidth/1920,innerHeight/1080);document.getElementById('stage').style.transform=`translate(-50%,-50%) scale(${{k}})`}}
function go(j){{i=Math.max(0,Math.min(S.length-1,j));S.forEach((s,t)=>s.classList.toggle('on',t===i));document.getElementById('n').textContent=(i+1)+' / '+S.length;history.replaceState(null,'','#'+(i+1))}}
addEventListener('resize',fit);addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key))go(i+1);else if(['ArrowLeft','PageUp'].includes(e.key))go(i-1);
else if(e.key==='n'||e.key==='N')document.body.classList.toggle('notes');else if(e.key==='f'||e.key==='F')document.documentElement.requestFullscreen?.();else if(e.key==='Home')go(0);else if(e.key==='End')go(S.length-1)}});
addEventListener('click',e=>go(e.clientX>innerWidth/3?i+1:i-1));fit();go(i);
</script></body></html>"""
out = H / ("apresentacao.html" if "--single" not in sys.argv else f"_tmp_{order[0]}.html")
page = page.replace("<div id=\"n\"></div>", "") if "--single" in sys.argv else page
out.write_text(page); print(out, len(order), "slides")
