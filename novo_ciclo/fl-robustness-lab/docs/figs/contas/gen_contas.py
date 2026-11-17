"""Gera a seção 'As contas da defesa, passo a passo' (H.IAAC) a partir de toy.json (verificado com layer_guard.py)."""
import json, re
import numpy as np

SP = "/tmp/claude-10434/-home-marcos-morais/0456b4d0-2ccc-4ae8-adfa-de654dd9322d/scratchpad"
P = f"{SP}/deck_h/project"
d = json.load(open(f"{SP}/contas/toy.json"))
D = np.array(d["D"]); W = np.array(d["W"]); MED = np.array(d["med"]); TA = d["tens"]["A"]; TB = d["tens"]["B"]; TC = d["tens"]["C"]
AGG = {k: np.array(v) for k, v in d["agg"].items()}; HON = np.array(d["honest"])
ROB = "'Roboto', Arial, sans-serif"
NAVY, ORG, GRN, RUST, GRY = "#071F4A", "#FF7519", "#1F7A52", "#A9481A", "#6E6E6E"
ATK = "background:#FFF1E6"
TH = f"color:#FFFFFF;background:{NAVY};padding:9px 12px;text-align:center"
MAL = {4, 5}


def f(x, nd=1):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−").replace(".", ",")


def td(x, extra="", last=False):
    b = "" if last else "border-bottom:1px solid #D9D9D9;"
    return f'<td style="{b}padding:7px 12px;text-align:center;{extra}">{x}</td>'


def cl(i):
    return f"cliente {i + 1}" + (" <span style='color:#C25400'>(atacante)</span>" if i in MAL else "")


def slide(sid, title, eyebrow, body, footer, aside, gap=24, wide=False, size=42):
    bw, hw = (1600, 1445) if wide else (1147, 992)
    return f'''<section id="{sid}" data-transition="fade" style="background:#FFFFFF;color:#222222;font-family:{ROB};padding:272px 128px 160px;display:flex;flex-direction:column;gap:{gap}px">
<div style="position:absolute;left:1378px;top:0px;width:542px;height:31px;background:#FF7519"></div>
<div style="position:absolute;left:0px;top:1049px;width:542px;height:31px;background:#FF7519"></div>
<div style="position:absolute;left:0px;top:76px;width:{bw}px;height:138px;background:{NAVY}"></div>
<div style="position:absolute;left:115px;top:76px;width:{hw}px;height:138px;display:flex;align-items:center"><h2 style="font-family:{ROB};font-size:{size}px;font-weight:700;line-height:1.12;color:#FFFFFF">{title}</h2></div>
<p style="position:absolute;left:115px;top:228px;width:1650px;font-family:{ROB};font-size:24px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#9E9E9E">{eyebrow}</p>
<p style="position:absolute;left:1760px;top:1036px;width:120px;text-align:right;font-family:{ROB};font-size:22px;color:#9E9E9E">0</p>
{body}
<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:22px;color:{GRY}">{footer}</p>
<aside>{aside}</aside>
</section>
'''


def box(html, color=NAVY, fs=25):
    return f'<div style="flex:1;border-left:6px solid {color};padding:4px 0 4px 22px"><p style="font-size:{fs}px;line-height:1.42">{html}</p></div>'


def formula(html, fs=30):
    return f'<div style="background:#FFFFFF;border:3px solid {NAVY};border-radius:4px;padding:16px 26px"><p style="font-size:{fs}px;line-height:1.4;color:{NAVY}">{html}</p></div>'


def table(head, rows, fs=23, widths=None):
    h = "".join(f'<th style="{TH};{("width:" + widths[k] + ";") if widths else ""}{"text-align:left;" if k == 0 else ""}">{x}</th>' for k, x in enumerate(head))
    return f'<table style="width:100%;font-size:{fs}px;color:#222222;border-collapse:collapse"><tr>{h}</tr>{"".join(rows)}</table>'


S = {}
FOOT = "Exemplo ilustrativo (números pequenos, escala arbitrária), calculado e conferido com o código da defesa: src/defenses/layer_guard.py."

# ------------------------------------------------------------------ divisor
S["div-contas"] = f'''<section id="div-contas" data-transition="fade" data-section="As contas da defesa, passo a passo" style="background:#FFFFFF;color:#071F4A;font-family:{ROB};padding:128px">
<img src="/_blob/435b108df3f4346ca16ccc5155a84374" alt="Ilustração do H.IAAC: um celular com um chip cerebral, em roxo, sobre fundo branco, com uma linha laranja abaixo." style="position:absolute;left:0px;top:0px;width:1920px;height:1080px;object-fit:cover">
<h1 style="position:absolute;left:160px;top:690px;width:1600px;text-align:center;font-family:{ROB};font-size:88px;font-weight:700;line-height:1.1;color:#071F4A">As contas da defesa, passo a passo</h1>
<p style="position:absolute;left:160px;top:866px;width:1600px;text-align:center;font-family:{ROB};font-size:40px;font-weight:700;color:#9E9E9E">Um exemplo pequeno, feito à mão e conferido no código</p>
<aside>Nesta seção abrimos a caixa: cada conta que a defesa faz numa rodada, na ordem em que o código faz, com um exemplo pequeno o bastante para conferir à mão. Os números do exemplo foram calculados pelo próprio código da defesa, então tudo que aparece aqui é exatamente o que o servidor faria.</aside>
</section>
'''

# ------------------------------------------------------------------ 1. dados
rows = []
for i in range(6):
    st = ATK if i in MAL else ""
    last = i == 5
    cells = "".join(td(f(D[i, k]), st + ("border-left:3px solid #071F4A;" if k in (3, 6) else ""), last) for k in range(9))
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>{cells}{td(int(W[i]), st, last)}{td(f(W[i] / W.sum(), 1), st, last)}</tr>')
head2 = f'<tr><th style="{TH}"></th><th colspan="3" style="{TH}">tensor A</th><th colspan="3" style="{TH};border-left:3px solid #FFFFFF">tensor B</th><th colspan="3" style="{TH};border-left:3px solid #FFFFFF">tensor C</th><th colspan="2" style="{TH};border-left:3px solid #FFFFFF">peso</th></tr>'
t = table(["cliente"] + ["1", "2", "3"] * 3 + ["n<sub>i</sub>", "p<sub>i</sub>"], rows, fs=23).replace("<tr><th", head2 + "<tr><th", 1)
body = t + f'''<div style="display:flex;gap:26px">
{box("<b>Δ<sub>i</sub></b> = a atualização que o cliente <i>i</i> manda: modelo treinado por ele menos o modelo global da rodada. Aqui, 9 números em 3 tensores.")}
{box("<b>n<sub>i</sub></b> = quantos exemplos o cliente tem; <b>p<sub>i</sub> = n<sub>i</sub> / Σn</b> é o peso dele na média (como no FedAvg).", GRN)}
{box("Os clientes <b>5 e 6</b> são os atacantes e mandam quase a mesma coisa. <b>O servidor não sabe disso.</b>", ORG)}
</div>'''
S["contas-dados"] = slide("contas-dados", "O exemplo: 6 clientes, 3 tensores de 3 números", "As contas · o ponto de partida", body,
    "Exemplo ilustrativo calculado pelo código da defesa, com T = 2. No real: 20 clientes, 62 tensores (ResNet18) e 11 milhões de números por cliente; as contas são as mesmas.",
    "Este é o ponto de partida. Cada linha é o que um cliente mandou na rodada: 9 números, separados em 3 tensores, que fazem o papel das camadas do modelo. As duas últimas colunas são o peso de cada um na média, proporcional à quantidade de dados, como no FedAvg. Os clientes 5 e 6 são os atacantes e mandam atualizações quase iguais, o que é típico de um backdoor coordenado. O servidor não sabe quem é quem: tudo o que ele tem é esta tabela.")

# ------------------------------------------------------------------ 2. mediana
col = D[:, 0]; srt = sorted(col)
rows = []
for i in range(6):
    st = ATK if i in MAL else ""; last = i == 5
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>' +
                "".join(td(f(D[i, k]), st, last) for k in range(3)) +
                "".join(td(f"<b>{f(TA['C'][i][k])}</b>", st + ("border-left:3px solid #071F4A;" if k == 0 else ""), last) for k in range(3)) + "</tr>")
rows.insert(0, f'<tr><td style="border-bottom:2px solid {NAVY};padding:7px 12px;color:{NAVY}"><b>mediana</b></td>' + "".join(td(f"<b>{f(MED[k])}</b>", f"color:{NAVY};border-bottom:2px solid {NAVY};") for k in range(3)) + "".join(td("—", f"color:{GRY};border-bottom:2px solid {NAVY};" + ("border-left:3px solid #071F4A;" if k == 0 else "")) for k in range(3)) + "</tr>")
t = table(["tensor A", "Δ[1]", "Δ[2]", "Δ[3]", "C[1]", "C[2]", "C[3]"], rows, fs=23)
srt_s = " · ".join(f(x) for x in srt)
body = f'''{formula("C<sub>i</sub> = Δ<sub>i</sub> − mediana<sub>j</sub>(Δ<sub>j</sub>), coordenada por coordenada, com os 6 clientes")}
<div style="display:flex;gap:34px">
<div style="width:900px">{t}</div>
<div style="flex:1;display:flex;flex-direction:column;gap:18px">
<div style="background:#F3F3F3;box-shadow:0 2px 4px rgba(0,0,0,0.22);border-radius:4px;padding:20px 24px"><p style="font-size:24px;line-height:1.45"><b>Conta da coordenada 1:</b> ordena os 6 valores<br>{srt_s}<br>Com número par, o código pega o <b>menor dos dois do meio</b>: mediana = <b>{f(MED[0])}</b>.<br>Cliente 5: C = 1,2 − 0,2 = <b>1,0</b>.</p></div>
{box("Os atacantes são minoria, então a mediana fica do lado dos honestos. Ao descontá-la, sobra o que é <b>próprio</b> de cada cliente: nos atacantes, o backdoor.", ORG, 24)}
</div></div>'''
S["contas-mediana"] = slide("contas-mediana", "Passo 1: descontar a mediana de cada coordenada", "As contas · centralizar", body,
    FOOT + " A mediana usa todos os clientes, inclusive os atacantes.",
    "O primeiro passo tira de cada atualização o que é comum a todos. Para cada um dos números, coordenada por coordenada, calculamos a mediana entre os clientes e subtraímos. Veja a coordenada 1: ordenados, os valores são menos 1,3, 0, 0,2, 0,3, 1,2 e 1,2. Como são 6 valores, há dois no meio, 0,2 e 0,3; o código usa o menor, 0,2. O cliente 5 mandou 1,2, então fica com 1,0. Note como os atacantes ficam com números grandes e do mesmo sinal depois de centralizar: é o backdoor aparecendo. Isso funciona porque os atacantes são minoria e não conseguem mover a mediana.")

# ------------------------------------------------------------------ 3. cosseno
cm = np.array(TA["cm"]); s = np.array(TA["s"])
rows = []
for i in range(6):
    st = ATK if i in MAL else ""; last = i == 5
    j_best = max((j for j in range(6) if j != i), key=lambda j: cm[i, j])
    cells = "".join(td("—" if j == i else (f"<b style='color:{RUST}'>{f(cm[i, j], 2)}</b>" if j == j_best else f(cm[i, j], 2)), st + ("color:#9E9E9E;" if j == i else ""), last) for j in range(6))
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>{cells}{td(f"<b>{f(s[i], 3)}</b>", st + "border-left:3px solid #071F4A;", last)}</tr>')
t = table(["tensor A", "c1", "c2", "c3", "c4", "c5", "c6", "s<sub>i</sub>"], rows, fs=22)
body = f'''{formula("cos(a, b) = (a · b) / (‖a‖ · ‖b‖) &nbsp;&nbsp;&nbsp;&nbsp; s<sub>i</sub> = maior cos(C<sub>i</sub>, C<sub>j</sub>) entre os outros clientes j", 28)}
<div style="display:flex;gap:34px">
<div style="width:1000px">{t}</div>
<div style="flex:1;display:flex;flex-direction:column;gap:18px">
<div style="background:#F3F3F3;box-shadow:0 2px 4px rgba(0,0,0,0.22);border-radius:4px;padding:20px 24px"><p style="font-size:23px;line-height:1.45"><b>Conta de cos(C<sub>5</sub>, C<sub>6</sub>):</b><br>a · b = 1,0·1,0 + 2,2·2,4 + 2,2·2,6 = 12,00<br>‖C<sub>5</sub>‖·‖C<sub>6</sub>‖ = 3,268 · 3,677 = 12,02<br>cos = 12,00 / 12,02 = <b>0,999</b></p></div>
{box("1 = mesma direção · 0 = sem relação · −1 = direção oposta. Em vermelho, o mais parecido de cada linha: é o <b>s<sub>i</sub></b>.", NAVY, 23)}
</div></div>'''
S["contas-cosseno"] = slide("contas-cosseno", "Passo 2a: quão parecido cada cliente é com o mais parecido", "As contas · semelhança, por tensor", body,
    FOOT + " Calculado separadamente em cada tensor, com as atualizações já centralizadas.",
    "Agora, dentro de cada tensor, comparamos cada cliente com todos os outros pelo cosseno, que mede se dois vetores apontam para a mesma direção, sem ligar para o tamanho. A tabela é a matriz de cossenos do tensor A. Para cada cliente, guardamos só o maior valor da linha, o cliente mais parecido com ele: é o s. Os atacantes 5 e 6 têm cosseno 0,999 entre si, praticamente idênticos, porque mandam o mesmo backdoor. Os honestos, com dados diferentes, ficam bem abaixo. A conta do quadro à direita mostra o cosseno entre 5 e 6 feito à mão.", wide=True)

# ------------------------------------------------------------------ 4. norma
nr = np.array(TA["nr"]); C = np.array(TA["C"])
rows = []
for i in range(6):
    st = ATK if i in MAL else ""; last = i == 5
    sq = " + ".join(f"{f(abs(x))}²" for x in C[i])
    bar = f'<div style="height:22px;width:{int(nr[i] / 3.7 * 360)}px;background:{ORG if i in MAL else NAVY}"></div>'
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>{td("(" + "; ".join(f(x) for x in C[i]) + ")", st, last)}{td("√(" + sq + ")", st + "text-align:left;", last)}{td(f"<b>{f(nr[i], 3)}</b>", st, last)}<td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{bar}</td></tr>')
t = table(["tensor A", "C<sub>i</sub>", "conta", "‖C<sub>i</sub>‖", ""], rows, fs=23, widths=["22%", "18%", "28%", "10%", "22%"])
body = f'''{formula("‖C<sub>i</sub>‖ = √(soma dos quadrados dos números de C<sub>i</sub> naquele tensor)")}
{t}
<div style="display:flex;gap:26px">
{box("O cosseno não vê tamanho. A norma vê: um atacante que <b>aumenta</b> a atualização (escala, ruído) aparece aqui mesmo quando não se parece com ninguém.", NAVY, 24)}
{box("No tensor A, os atacantes têm as maiores normas: <b>3,27 e 3,68</b>, contra 0,9 a 1,6 dos honestos.", ORG, 24)}
</div>'''
S["contas-norma"] = slide("contas-norma", "Passo 2b: o tamanho de cada atualização centralizada", "As contas · tamanho, por tensor", body, FOOT,
    "A segunda medida é o tamanho, a norma: a raiz da soma dos quadrados dos números do cliente naquele tensor, depois de centralizar. Ela completa o cosseno, que ignora o tamanho. Um atacante que multiplica a atualização, ou que mistura ruído para não se parecer com ninguém, escapa do cosseno mas não escapa da norma. No exemplo, os dois atacantes têm as maiores normas do tensor A.")

# ------------------------------------------------------------------ 5. z robusto
def ztab(vals, z, lab, med, nd):
    rows = []
    for i in range(6):
        st = ATK if i in MAL else ""; last = i == 5
        rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:6px 10px;{st}">cliente {i + 1}</td>{td(f(vals[i], nd), st, last)}{td(f(abs(vals[i] - med), nd), st, last)}{td(f"<b>{f(z[i], 2)}</b>", st, last)}</tr>')
    return table(["", lab, f"|{lab} − {f(med, nd)}|", "z"], rows, fs=22)

smed, sm = float(np.median(s)), float(np.median(np.abs(s - np.median(s))))
nmed, nm = float(np.median(nr)), float(np.median(np.abs(nr - np.median(nr))))
card = lambda title, med, mad, tab, note: f'''<div style="flex:1;background:#F3F3F3;box-shadow:0 2px 4px rgba(0,0,0,0.22);border-radius:4px"><p style="font-size:25px;font-weight:700;color:#FFFFFF;padding:10px 20px;background:{NAVY}">{title}</p>
<div style="padding:14px 18px;display:flex;flex-direction:column;gap:10px"><p style="font-size:22px;line-height:1.4">mediana = <b>{med}</b> · MAD = <b>{mad[0]}</b> · 1,4826 × MAD = <b>{mad[1]}</b></p>{tab}<p style="font-size:21px;line-height:1.35;color:{GRY}">{note}</p></div></div>'''
body = f'''{formula("z = (x − mediana) / (1,4826 × MAD) &nbsp;&nbsp;&nbsp; MAD = mediana de |x − mediana| &nbsp;&nbsp;&nbsp; (entre os 6 clientes, no mesmo tensor)", 27)}
<div style="display:flex;gap:28px">
{card("z_sim · a partir de s", f(smed, 3) + " (média de 0,176 e 0,623)", (f(sm, 3), f(1.4826 * sm, 3)), ztab(s, TA["zs"], "s", smed, 3), "Cliente 5: (0,999 − 0,400) / 0,400 = 1,50")}
{card("z_norm · a partir de ‖C‖", f(nmed, 3), (f(nm, 3), f(1.4826 * nm, 3)), ztab(nr, TA["zn"], "‖C‖", nmed, 3), "Cliente 5: (3,268 − 1,483) / 0,636 = 2,80")}
</div>'''
S["contas-z"] = slide("contas-z", "Passo 3a: a régua de cada tensor, o z robusto", "As contas · padronizar", body,
    FOOT + " O fator 1,4826 torna o MAD comparável ao desvio-padrão; o código soma 10⁻¹² ao denominador para nunca dividir por zero.",
    "Cosseno e norma têm escalas diferentes em cada tensor e em cada momento do treino. Para comparar, transformamos cada medida num z robusto: quantos desvios típicos o cliente está acima do cliente típico. Usamos mediana e MAD em vez de média e desvio-padrão porque elas não se deixam arrastar pelos próprios atacantes. O MAD é a mediana das distâncias até a mediana; multiplicado por 1,4826, ele vale o mesmo que o desvio-padrão quando os dados são normais. Repare que aqui a mediana de s é a média dos dois do meio, 0,176 e 0,623: nesta etapa o código usa a mediana usual. As duas contas de baixo mostram o cliente 5.", wide=True, gap=20)

# ------------------------------------------------------------------ 6. nota
zs, zn, sc = np.array(TA["zs"]), np.array(TA["zn"]), np.array(TA["sc"])
rows = []
for i in range(6):
    st = ATK if i in MAL else ""; last = i == 5; acc = sc[i] > 3
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>{td(f(zs[i], 2), st, last)}{td(f(zn[i], 2), st, last)}{td(f(max(zs[i], 0), 2), st, last)}{td(f(max(zn[i], 0), 2), st, last)}{td(f"<b>{f(sc[i], 2)}</b>", st, last)}{td(f"<b style=color:{RUST if acc else GRN}>{'acusa' if acc else 'não'}</b>", st, last)}</tr>')
t = table(["tensor A", "z_sim", "z_norm", "max(z_sim, 0)", "max(z_norm, 0)", "nota", "nota &gt; 3?"], rows, fs=23)
body = f'''{formula("nota<sub>i</sub> = max(z_sim, 0) + max(z_norm, 0) &nbsp;&nbsp;&nbsp;&nbsp; o tensor <b>acusa</b> o cliente se nota &gt; 3")}
{t}
<div style="display:flex;gap:26px">
{box("<b>max(·, 0)</b>: só conta o que é suspeito para cima. Ser menos parecido ou menor que o típico não soma nem desconta.", NAVY, 24)}
{box("<b>Cliente 5:</b> nenhuma medida sozinha passa de 3 (1,50 e 2,80), mas a soma, <b>4,30</b>, passa. Por isso somamos as duas.", ORG, 24)}
</div>'''
S["contas-nota"] = slide("contas-nota", "Passo 3b: a nota de suspeita e a acusação", "As contas · acusar, por tensor", body,
    FOOT + " O limiar 3 é o mesmo em todos os tensores, modelos e variantes.",
    "Juntamos as duas réguas numa nota. Primeiro zeramos os valores negativos: ser menos parecido ou menor que o típico não é suspeito. Depois somamos. Se a nota passa de 3, o tensor acusa o cliente. No tensor A, os dois atacantes são acusados. O cliente 5 mostra por que a soma importa: o z de semelhança dele é 1,50 e o de tamanho 2,80, nenhum passa de 3 sozinho, mas juntos dão 4,30. Os honestos ficam entre 0 e 0,56.")

# ------------------------------------------------------------------ 7. contagem
scs = [np.array(TA["sc"]), np.array(TB["sc"]), np.array(TC["sc"])]
cnt = sum((x > 3).astype(int) for x in scs)
rows = []
for i in range(6):
    st = ATK if i in MAL else ""; last = i == 5
    cells = "".join(td(f"<b style='color:#FFFFFF'>{f(x[i], 2)}</b>" if x[i] > 3 else f(x[i], 2), (f"background:{RUST};" if x[i] > 3 else st), last) for x in scs)
    conf = cnt[i] >= 2
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px;{st}">{cl(i)}</td>{cells}{td(f"<b>{cnt[i]}</b>", st + "border-left:3px solid #071F4A;", last)}{td(f"<b style=color:{RUST if conf else GRN}>{'confirmado' if conf else 'não'}</b>", st, last)}</tr>')
t = table(["", "nota no tensor A", "nota no tensor B", "nota no tensor C", "acusações", "≥ T = 2?"], rows, fs=23)
body = f'''{formula("acusações<sub>i</sub> = nº de tensores com nota &gt; 3 &nbsp;&nbsp;&nbsp;&nbsp; <b>confirmado</b> se acusações<sub>i</sub> ≥ T")}
{t}
<div style="display:flex;gap:22px">
{box("<b>Cliente 2:</b> acusado só pelo tensor B (norma alta, 3,03). Uma acusação isolada <b>não confirma</b>: é o honesto acusado por acaso.", NAVY, 23)}
{box("<b>Tensor C:</b> os atacantes passam com nota 0. É o <b>vazamento</b>: decidindo só tensor a tensor, o veneno de C entraria.", ORG, 23)}
{box("No real: <b>T = 8</b> de 62 tensores na ResNet18 e T = 2 na SmallCNN, fixados antes de rodar.", GRN, 23)}
</div>'''
S["contas-contagem"] = slide("contas-contagem", "Passo 4: contar quantos tensores acusam cada cliente", "As contas · a corroboração", body, FOOT,
    "Repetimos os passos 2 e 3 em cada tensor e contamos as acusações de cada cliente. Em vermelho, as notas acima de 3. Os atacantes são acusados pelos tensores A e B, duas vezes, e com T igual a 2 ficam confirmados. O cliente 2, honesto, foi acusado uma vez, no tensor B, porque a atualização dele ali é grande; com uma só acusação ele não é confirmado. E repare no tensor C: lá os atacantes não chamam atenção nenhuma. É o mesmo vazamento que vimos no LaySelFL, onde o veneno passa pelos tensores convolucionais. A contagem resolve isso: quem é confirmado sai de todos os tensores, inclusive do C.")

# ------------------------------------------------------------------ 8. quem entra
def cset(xs):
    return ", ".join(str(x) for x in xs)
rule = [("sem defesa (FedAvg)", "todos entram", "1 a 6", "1 a 6", "1 a 6", ""),
        ("sn · só por tensor", "sai quem <b>aquele tensor</b> acusou", "1, 2, 3, 4", "1, 3, 4", "<b style='color:#A9481A'>1 a 6</b>", ""),
        ("corr · corroboração", "sai quem o tensor acusou <b>ou</b> quem foi confirmado", "1, 2, 3, 4", "1, 3, 4", "<b style='color:#1F7A52'>1, 2, 3, 4</b>", ""),
        ("corr-soft · substituição", "como a corr, mas quem sai é <b>trocado</b> pela mediana de quem entra", "1–4 · <i>5, 6 trocados</i>", "1, 3, 4 · <i>2, 5, 6 trocados</i>", "1–4 · <i>5, 6 trocados</i>", ""),
        ("client · purga por cliente", "qualquer acusação tira o cliente de <b>tudo</b>", "1, 3, 4", "1, 3, 4", "1, 3, 4", "")]
rows = []
for k, (v, r, a, b, c, _) in enumerate(rule):
    last = k == len(rule) - 1
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:9px 12px"><b>{v}</b></td><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:9px 12px">{r}</td>{td(a, "", last)}{td(b, "", last)}{td(c, "", last)}</tr>')
t = table(["variante", "regra", "tensor A", "tensor B", "tensor C"], rows, fs=23, widths=["21%", "33%", "15%", "16%", "15%"])
body = f'''{t}
<div style="display:flex;gap:26px">
{box("<b>sn no tensor C:</b> ninguém foi acusado ali, então os atacantes <b>entram</b>. A corr os tira porque já foram confirmados por A e B.", ORG, 24)}
{box("<b>client:</b> o cliente 2, honesto, sai de tudo por uma acusação isolada. Contém mais, mas descarta dados bons.", NAVY, 24)}
{box("<b>Se ninguém sobrar</b> num tensor: entram os não confirmados; se nem eles, todos.", GRY, 24)}
</div>'''
S["contas-quem"] = slide("contas-quem", "Passo 5a: quem entra na média de cada tensor", "As contas · a decisão de cada variante", body,
    FOOT + " Os passos 1 a 4 são idênticos em todas as variantes; só esta decisão muda.",
    "Com as acusações e a contagem na mão, cada variante decide quem entra na média de cada tensor. Na sn, a decisão é tensor a tensor: no tensor C ninguém foi acusado, então os atacantes entram, e é por isso que a sn deixa o backdoor passar. A corr soma a regra da contagem: confirmado sai de todos os tensores. A corr-soft decide igual à corr, mas em vez de tirar, troca a atualização pela mediana de quem entrou. A client tira de tudo qualquer cliente acusado uma vez, o que inclui o honesto 2.", wide=True)

# ------------------------------------------------------------------ 9. média ponderada (A1)
a1 = D[:, 0]
body = f'''{formula("agregado = Σ p<sub>i</sub> · Δ<sub>i</sub> / Σ p<sub>i</sub> &nbsp;(só quem entra) &nbsp;&nbsp;&nbsp;&nbsp; corr-soft: Σ p<sub>i</sub> · x<sub>i</sub>, com x<sub>i</sub> = Δ<sub>i</sub> ou a mediana de quem entra", 26)}
<p style="font-size:24px;line-height:1.4">Coordenada 1 do tensor A. Valores: <b>0,3 · 0,2 · −1,3 · 0,0 · 1,2 · 1,2</b> &nbsp;·&nbsp; pesos n<sub>i</sub>: 400 · 200 · 200 · 400 · 400 · 400</p>
{table(["variante", "conta", "resultado"], [
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>sem defesa</b></td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">(400·0,3 + 200·0,2 + 200·(−1,3) + 400·0,0 + 400·1,2 + 400·1,2) / 2000 = 860 / 2000</td>{td("<b style=color:#A9481A>0,43</b>")}</tr>',
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>sn e corr</b> (sem 5 e 6)</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">(120 + 40 − 260 + 0) / 1200 = −100 / 1200</td>{td("<b style=color:#1F7A52>−0,083</b>")}</tr>',
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>client</b> (sem 2, 5 e 6)</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">(120 − 260 + 0) / 1000 = −140 / 1000</td>{td("<b style=color:#1F7A52>−0,14</b>")}</tr>',
    f'<tr><td style="padding:8px 12px"><b>corr-soft</b> (5 e 6 trocados)</td><td style="padding:8px 12px">mediana de quem entra (0,3 · 0,2 · −1,3 · 0,0) = <b>0,0</b> → (120 + 40 − 260 + 0 + 400·0,0 + 400·0,0) / 2000</td>{td("<b style=color:#1F7A52>−0,05</b>", "", True)}</tr>'], fs=23, widths=["22%", "64%", "14%"])}
<div style="display:flex;gap:26px">
{box("<b>Referência:</b> a média só dos 4 honestos dá <b>−0,083</b>. Sem defesa, os atacantes puxam o número para <b>+0,43</b>.", NAVY, 24)}
{box("A corr-soft mantém o peso dos atacantes (400 + 400), mas com o valor do consenso: a média não encolhe e o veneno não entra.", GRN, 24)}
</div>'''
S["contas-media"] = slide("contas-media", "Passo 5b: a média ponderada, número por número", "As contas · agregar", body,
    FOOT + " O novo modelo global é o global anterior somado ao agregado. A mediana de 4 valores, aqui, é o menor dos dois do meio.",
    "Agora a conta final, na primeira coordenada do tensor A. Sem defesa, é a média ponderada de todos: os dois atacantes, com 1,2 cada e peso alto, puxam o resultado para 0,43, quando os honestos sozinhos dariam menos 0,083. A sn e a corr tiram os atacantes desse tensor e chegam exatamente na média dos honestos. A client tira também o honesto 2 e chega a menos 0,14. A corr-soft não tira ninguém: troca o valor dos atacantes pela mediana dos que entraram, que aqui é 0, e mantém os pesos. O resultado, menos 0,05, fica perto dos honestos sem diminuir o número de votos.", wide=True, gap=18)

# ------------------------------------------------------------------ 10. tensor C (vazamento)
c1 = D[:, 6]
body = f'''<p style="font-size:24px;line-height:1.4">Coordenada 1 do tensor C, onde <b>nenhum tensor acusou os atacantes</b>. Valores: <b>1,8 · −0,2 · 1,2 · 0,7 · −0,2 · −0,1</b> &nbsp;·&nbsp; pesos: 400 · 200 · 200 · 400 · 400 · 400</p>
{table(["variante", "quem entra em C", "conta", "resultado"], [
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>sem defesa</b></td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">1 a 6</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">(720 − 40 + 240 + 280 − 80 − 40) / 2000</td>{td("<b style=color:#A9481A>0,54</b>")}</tr>',
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>sn</b></td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b style=color:#A9481A>1 a 6</b>: ninguém acusado em C</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">a mesma conta do FedAvg</td>{td("<b style=color:#A9481A>0,54</b>")}</tr>',
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>corr</b></td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">1 a 4: 5 e 6 confirmados</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">(720 − 40 + 240 + 280) / 1200</td>{td("<b style=color:#1F7A52>1,00</b>")}</tr>',
    f'<tr><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px"><b>corr-soft</b></td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">1 a 4; 5 e 6 trocados por 0,7</td><td style="border-bottom:1px solid #D9D9D9;padding:8px 12px">mediana (1,8 · −0,2 · 1,2 · 0,7) = 0,7 → (720 − 40 + 240 + 280 + 280 + 280) / 2000</td>{td("<b style=color:#1F7A52>0,88</b>")}</tr>',
    f'<tr><td style="padding:8px 12px"><b>client</b></td><td style="padding:8px 12px">1, 3, 4</td><td style="padding:8px 12px">(720 + 240 + 280) / 1000</td>{td("<b>1,24</b>", "", True)}</tr>'], fs=23, widths=["15%", "27%", "45%", "13%"])}
<div style="display:flex;gap:26px">
{box("Média só dos honestos: <b>1,00</b>. A sn fica no <b>0,54</b> do FedAvg: os atacantes entram em C sem ser vistos.", ORG, 24)}
{box("Com a contagem, quem foi confirmado em A e B sai também de C. <b>É isso que fecha o vazamento</b> nos tensores convolucionais.", GRN, 24)}
</div>'''
S["contas-vazamento"] = slide("contas-vazamento", "Por que contar: o tensor onde o atacante não aparece", "As contas · o efeito da corroboração", body, FOOT,
    "Este slide mostra, em número, por que a corroboração é o coração da defesa. No tensor C os atacantes não chamam atenção, então a sn, que decide tensor a tensor, deixa os dois entrarem: o resultado é igual ao do FedAvg. A corr usa a evidência dos tensores A e B: os atacantes já estão confirmados e saem de C também, e a média volta exatamente para a dos honestos. É o mesmo mecanismo que, no LaySelFL real, leva o sucesso do backdoor de 62% na sn para 11% na corr, 5,5% na corr-soft e 4,3% na client.", wide=True, gap=20)

# ------------------------------------------------------------------ 11. resultado completo
names = [("sem defesa", "fedavg"), ("sn", "sn"), ("corr", "corr"), ("corr-soft", "corr_soft"), ("client", "client")]
rows = [f'<tr><td style="border-bottom:2px solid {NAVY};padding:7px 12px;color:{NAVY}"><b>só os honestos (referência)</b></td>' + "".join(td(f"<b>{f(HON[k], 2)}</b>", f"color:{NAVY};border-bottom:2px solid {NAVY};" + ("border-left:3px solid #071F4A;" if k in (3, 6) else "")) for k in range(9)) + "</tr>"]
for k, (lab, key) in enumerate(names):
    last = k == len(names) - 1
    v = AGG[key]
    rows.append(f'<tr><td style="{"" if last else "border-bottom:1px solid #D9D9D9;"}padding:7px 12px"><b>{lab}</b></td>' + "".join(td(f(v[c], 2), ("border-left:3px solid #071F4A;" if c in (3, 6) else "") + (f"color:{RUST};font-weight:700;" if abs(v[c] - HON[c]) > 0.3 else ""), last) for c in range(9)) + "</tr>")
head2 = f'<tr><th style="{TH}"></th><th colspan="3" style="{TH}">tensor A</th><th colspan="3" style="{TH};border-left:3px solid #FFFFFF">tensor B</th><th colspan="3" style="{TH};border-left:3px solid #FFFFFF">tensor C</th></tr>'
t = table(["agregado"] + ["1", "2", "3"] * 3, rows, fs=23).replace("<tr><th", head2 + "<tr><th", 1)
body = f'''{t}
<div style="display:flex;gap:26px">
{box("Em vermelho: mais de 0,3 longe da média dos honestos. Sem defesa, os atacantes deslocam <b>os três tensores</b>.", RUST, 24)}
{box("A <b>sn</b> limpa A e B mas deixa o tensor C igual ao FedAvg. A <b>corr</b> acerta A e C em cheio.", ORG, 24)}
{box("A <b>client</b> e a <b>corr-soft</b> ficam perto, com desvios menores: o preço de descartar ou de trocar pelo consenso.", NAVY, 24)}
</div>'''
S["contas-resultado"] = slide("contas-resultado", "O resultado: os 9 números agregados por cada variante", "As contas · o efeito no modelo", body,
    FOOT + " Novo modelo global = modelo global + agregado. No treino real isso se repete a cada rodada.",
    "Aqui está o agregado completo de cada variante, os 9 números, comparado com a média que teríamos se só os honestos participassem. Sem defesa, os três tensores saem deslocados na direção dos atacantes. A sn conserta A e B, onde houve acusação, mas não o C. A corr chega exatamente na referência em A e C; no B, ela tira o honesto 2 por ter sido acusado ali, e por isso difere um pouco. A corr-soft e a client ficam perto, cada uma com o seu custo. Esse número somado ao modelo global é o novo modelo da rodada.", wide=True)

# ------------------------------------------------------------------ 12. tm e clip
body = f'''<p style="font-size:24px;line-height:1.4">As duas variantes da Fase 16 fazem <b>a mesma troca da corr-soft</b> e depois uma conta a mais. Coordenada 1 do tensor A depois da troca: <b>0,3 · 0,2 · −1,3 · 0,0 · 0,0 · 0,0</b></p>
<div style="display:flex;gap:28px">
<div style="flex:1;background:#F3F3F3;box-shadow:0 2px 4px rgba(0,0,0,0.22);border-radius:4px"><p style="font-size:26px;font-weight:700;color:#FFFFFF;padding:12px 22px;background:{ORG}">+ média aparada (β = 0,2)</p>
<div style="padding:16px 22px;display:flex;flex-direction:column;gap:10px">
<p style="font-size:24px;line-height:1.45">1. Ordena: −1,3 · 0,0 · 0,0 · 0,0 · 0,2 · 0,3</p>
<p style="font-size:24px;line-height:1.45">2. Corta k = ⌊0,2 × 6⌋ = <b>1</b> de cada ponta</p>
<p style="font-size:24px;line-height:1.45">3. Média simples, <b>sem pesos</b>: (0,0 + 0,0 + 0,0 + 0,2) / 4 = <b>0,05</b></p>
<p style="font-size:22px;line-height:1.4;color:{GRY}">No real: 20 clientes, corta 4 de cada ponta.</p>
</div></div>
<div style="flex:1;background:#F3F3F3;box-shadow:0 2px 4px rgba(0,0,0,0.22);border-radius:4px"><p style="font-size:26px;font-weight:700;color:#FFFFFF;padding:12px 22px;background:{ORG}">+ corte de norma</p>
<div style="padding:16px 22px;display:flex;flex-direction:column;gap:10px">
<p style="font-size:24px;line-height:1.45">1. Normas no tensor A: 1,99 · 1,72 · 1,78 · 0,92 · 1,28 · 1,28</p>
<p style="font-size:24px;line-height:1.45">2. Teto = mediana das normas de quem entra (1 a 4) = <b>1,72</b></p>
<p style="font-size:24px;line-height:1.45">3. Fator = min(1, teto / norma): 0,86 · 1 · 0,96 · 1 · 1 · 1</p>
<p style="font-size:24px;line-height:1.45">4. Média ponderada: (400·0,3·0,86 + 200·0,2 + 200·(−1,3)·0,96) / 2000 = <b>−0,054</b></p>
</div></div>
</div>
{box("A ideia era limitar quem passa <b>abaixo</b> de T. Não funcionou contra o atacante adaptativo: 58 a 60% de sucesso do backdoor com o corte de norma e 76 a 83% com a média aparada, por isso a recomendada segue sendo a corr-soft.", RUST, 24)}'''
S["contas-extras"] = slide("contas-extras", "As variantes da Fase 16: média aparada e corte de norma", "As contas · as respostas ao adaptativo", body, FOOT,
    "As duas tentativas de resposta ao atacante adaptativo partem da corr-soft e acrescentam uma conta. A média aparada ordena os valores de cada coordenada, corta os extremos, um de cada ponta no exemplo e quatro no real, e tira a média simples. O corte de norma limita o tamanho de cada cliente, no tensor, à mediana dos tamanhos de quem foi aceito, e depois faz a média ponderada. A intenção era que um atacante que fica abaixo do limiar de contagem tivesse influência limitada. Nos testes, nenhuma das duas segurou o atacante adaptativo.", wide=True, gap=20)

# ------------------------------------------------------------------ 13. resumo
steps = [
    ("1", "Centralizar", "C<sub>i</sub> = Δ<sub>i</sub> − mediana<sub>j</sub>(Δ<sub>j</sub>), por coordenada, com todos os clientes"),
    ("2", "Medir, em cada tensor g", "s<sub>i</sub> = max<sub>j≠i</sub> cos(C<sub>i</sub>[g], C<sub>j</sub>[g]) &nbsp;e&nbsp; ‖C<sub>i</sub>[g]‖"),
    ("3", "Padronizar", "z = (x − mediana) / (1,4826 · MAD + 10⁻¹²), entre os clientes da rodada"),
    ("4", "Acusar", "nota = max(z_sim, 0) + max(z_norm, 0); o tensor acusa se nota &gt; 3"),
    ("5", "Contar", "acusações<sub>i</sub> = nº de tensores que acusam; confirmado se ≥ T (8 na ResNet18, 2 na SmallCNN)"),
    ("6", "Agregar", "média ponderada por p<sub>i</sub> de quem entra (sn, corr, client) ou com os excluídos trocados pela mediana (corr-soft)"),
    ("7", "Atualizar", "novo modelo global = modelo global + agregado"),
]
items = "".join(f'<div style="display:flex;gap:22px;align-items:baseline;border-bottom:1px solid #D9D9D9;padding:8px 0"><p style="font-size:34px;font-weight:700;color:{ORG};width:44px">{n}</p><p style="font-size:27px;font-weight:700;color:{NAVY};width:360px">{t}</p><p style="font-size:26px;line-height:1.35;flex:1">{x}</p></div>' for n, t, x in steps)
body = f'''<div style="display:flex;flex-direction:column">{items}</div>
<p style="font-size:22px;line-height:1.4;color:{GRY}">Detalhes do código: a mediana do passo 1 e a da troca da corr-soft pegam o menor dos dois valores do meio (torch.median); a do z robusto usa a média dos dois (np.median). Se num tensor ninguém sobrar, entram os não confirmados; se nem eles, todos.</p>'''
S["contas-resumo"] = slide("contas-resumo", "Todas as contas numa página", "As contas · resumo", body,
    "src/defenses/layer_guard.py: layer_guard_sn, layer_guard_corr, layer_guard_corr_soft, layer_guard_client e as variantes _tm e _clip.",
    "Este é o resumo para consulta. Sete contas por rodada, todas só com as atualizações daquela rodada, sem dados limpos e sem histórico. Os passos 1 a 5 são iguais em todas as variantes; o passo 6 é onde elas diferem. O rodapé registra dois detalhes de implementação que importam para quem for reproduzir: qual mediana é usada em cada lugar e o que acontece se um tensor ficar sem ninguém.", gap=14)

ORDER = ["div-contas", "contas-dados", "contas-mediana", "contas-cosseno", "contas-norma", "contas-z", "contas-nota", "contas-contagem",
         "contas-quem", "contas-media", "contas-vazamento", "contas-resultado", "contas-extras", "contas-resumo"]
for k in ORDER:
    open(f"{P}/slides/{k}.html", "w").write(S[k])
idx = json.load(open(f"{P}/deck.json")); o = idx["order"]
o[:] = [x for x in o if x not in ORDER]
at = o.index("tsweep") + 1
o[at:at] = ORDER
idx["sections"]["contas"] = {"description": "As contas da defesa, passo a passo, num exemplo conferido no código", "start": "div-contas"}
json.dump(idx, open(f"{P}/deck.json", "w"), ensure_ascii=False, indent=2)
NUM = re.compile(r'(top:1036px;width:120px;text-align:right;[^"]*">)\d+(</p>)')
ch = []
for n, sid in enumerate(o, 1):
    if n < at + 1:
        continue
    fp = f"{P}/slides/{sid}.html"; s0 = open(fp).read(); s2 = NUM.sub(lambda m: f"{m.group(1)}{n}{m.group(2)}", s0)
    if s2 != s0 or sid in ORDER:
        open(fp, "w").write(s2); ch.append(sid)
print(len(o), at + 1, json.dumps({f"project/slides/{c}.html": f"project/slides/{c}.html" for c in ch}))
