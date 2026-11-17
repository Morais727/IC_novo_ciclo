"""GIF do pipeline da defesa com os escores REAIS da rodada 150 (cenário fiel, s42, agregador-sombra)."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

D = np.load("round151.npz")
zs, zn, mal = D["zs"], D["zn"], set(D["mal"].tolist())
score = np.maximum(zs, 0) + np.maximum(zn, 0)          # 20 x 62
flag = score > 3
count = flag.sum(1)
T = 8
conf = count >= T
N, G = score.shape
NAVY, ORG, GRN, GRY, RED, TEAL = "#071F4A", "#FF7519", "#1F7A52", "#D9D9D9", "#A9481A", "#2A9D8F"
plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
frames, durs = [], []


def canvas(step, title, caption):
    fig = plt.figure(figsize=(16, 9), dpi=90)
    fig.patch.set_facecolor("white")
    fig.add_artist(Rectangle((0.0, 0.885), 0.62, 0.075, transform=fig.transFigure, color=NAVY))
    fig.add_artist(Rectangle((0.72, 0.985), 0.28, 0.015, transform=fig.transFigure, color=ORG))
    fig.add_artist(Rectangle((0.0, 0.0), 0.28, 0.015, transform=fig.transFigure, color=ORG))
    fig.text(0.03, 0.922, title, color="white", fontsize=24, fontweight="bold", va="center")
    if step:
        fig.text(0.66, 0.922, step, color="#9E9E9E", fontsize=18, fontweight="bold", va="center")
    fig.text(0.03, 0.03, caption, color="#222222", fontsize=17, va="bottom", wrap=True)
    return fig


def grab(fig, ms):
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    buf.seek(0); frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)); durs.append(ms)


def heat(ax, mat, cmap, vmin, vmax, cols=G, rowcolors=None):
    m = np.full_like(mat, np.nan, dtype=float); m[:, :cols] = mat[:, :cols]
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(N)); ax.set_yticklabels([f"cliente {i}" for i in range(N)], fontsize=10)
    ax.set_xticks([0, 15, 30, 45, 61]); ax.set_xticklabels(["tensor 1", "16", "31", "46", "62"], fontsize=11)
    ax.set_xlabel("os 62 tensores do modelo (ResNet18-GN)", fontsize=13)
    for s in ax.spines.values(): s.set_visible(False)
    if rowcolors:
        for i, c in rowcolors.items(): ax.get_yticklabels()[i].set_color(c)


cm_score = LinearSegmentedColormap.from_list("s", ["#FFFFFF", "#C9D3E6", NAVY])
cm_flag = ListedColormap(["#EFEFEF", ORG])

# 0) clientes enviam
fig = canvas("", "Uma rodada real da defesa, passo a passo",
             "Rodada 150 do cenário fiel do LaySelFL: 20 clientes enviam suas atualizações. O servidor não sabe quem é atacante.")
ax = fig.add_axes([0.05, 0.2, 0.9, 0.6]); ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 10)
for i in range(N):
    ax.add_patch(FancyBboxPatch((i + 0.1, 6.6), 0.8, 1.6, boxstyle="round,pad=0.05", color="#9E9E9E"))
    ax.text(i + 0.5, 7.4, str(i), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.annotate("", xy=(10, 2.6), xytext=(i + 0.5, 6.5), arrowprops=dict(arrowstyle="->", color="#BDBDBD", lw=1))
ax.add_patch(FancyBboxPatch((7.5, 0.8), 5, 1.8, boxstyle="round,pad=0.1", color=NAVY))
ax.text(10, 1.7, "servidor", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
grab(fig, 2600)

# 1) centralizar
fig = canvas("passo 1 de 4", "Centralizar", "Cada atualização menos a mediana, coordenada a coordenada, de todas: sobra só o que é próprio de cada cliente.")
ax = fig.add_axes([0.05, 0.18, 0.9, 0.62]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.text(5, 4.3, r"$C_i \;=\; \Delta_i \;-\; \mathrm{mediana}_j\,(\Delta_j)$", ha="center", fontsize=40, color=NAVY)
ax.text(5, 2.4, "o que todos aprendem em comum é descontado;\no backdoor dos atacantes, que só eles têm, fica em evidência", ha="center", fontsize=20, color="#444444")
grab(fig, 3000)

# 2) medir cada tensor (revelado em colunas)
for cols in (10, 21, 32, 43, 53, 62):
    fig = canvas("passo 2 de 4", "Medir cada tensor",
                 "Em cada tensor e cliente: semelhança com o cliente mais parecido (z_sim) + tamanho (z_norm), em z robusto. Mais escuro = mais suspeito.")
    ax = fig.add_axes([0.1, 0.19, 0.86, 0.65]); heat(ax, np.clip(score, 0, 10), cm_score, 0, 10, cols)
    grab(fig, 450 if cols < 62 else 2400)

# 3) acusar
fig = canvas("passo 3 de 4", "Acusar: escore acima de 3",
             "Cada tensor 'acusa' o cliente se max(z_sim, 0) + max(z_norm, 0) > 3. Laranja = acusação. Repare nas linhas cheias.")
ax = fig.add_axes([0.1, 0.19, 0.86, 0.65]); heat(ax, flag.astype(float), cm_flag, 0, 1)
grab(fig, 3000)

# 4) contar
order = list(range(N))
for frac in (0.35, 0.7, 1.0):
    fig = canvas("passo 4 de 4", "Contar as acusações de cada cliente",
                 f"Corroboração: quantos dos 62 tensores acusaram cada cliente. Linha tracejada: limiar T = {T}.")
    ax = fig.add_axes([0.1, 0.19, 0.5, 0.65]); heat(ax, flag.astype(float), cm_flag, 0, 1)
    bx = fig.add_axes([0.66, 0.19, 0.3, 0.65])
    bx.barh(range(N), count * frac, color=[NAVY] * N, height=0.7); bx.set_ylim(N - 0.5, -0.5)
    bx.axvline(T, color=RED, ls="--", lw=2.5); bx.set_xlim(0, 30); bx.set_yticks([])
    bx.set_xlabel("tensores que acusam", fontsize=13)
    for s in ("top", "right"): bx.spines[s].set_visible(False)
    if frac == 1.0:
        for i, c in enumerate(count): bx.text(c + 0.5, i, str(int(c)), va="center", fontsize=11)
    grab(fig, 600 if frac < 1 else 2600)

# 5) confirmados + verdade
fig = canvas("decisão", "Confirmados: 8 ou mais acusações",
             "Os 6 atacantes reais (laranja) são acusados por 16 a 25 tensores e são confirmados; 1 honesto (cliente 11, com 8) é falso alarme.")
ax = fig.add_axes([0.1, 0.19, 0.5, 0.65]); heat(ax, flag.astype(float), cm_flag, 0, 1,
                                               rowcolors={i: ORG for i in mal})
bx = fig.add_axes([0.66, 0.19, 0.3, 0.65])
bx.barh(range(N), count, color=[(ORG if i in mal else ("#9E9E9E" if not conf[i] else NAVY)) for i in range(N)], height=0.7,
        edgecolor=[(RED if conf[i] else "none") for i in range(N)], linewidth=2.5)
bx.set_ylim(N - 0.5, -0.5); bx.axvline(T, color=RED, ls="--", lw=2.5); bx.set_xlim(0, 30); bx.set_yticks([])
bx.set_xlabel("tensores que acusam", fontsize=13)
for s in ("top", "right"): bx.spines[s].set_visible(False)
for i, c in enumerate(count):
    bx.text(c + 0.5, i, ("atacante" if i in mal else ("falso alarme" if conf[i] else "")), va="center", fontsize=11,
            color=(ORG if i in mal else RED), fontweight="bold")
grab(fig, 3600)

# 6) agregar (corr-soft): substitui pelo consenso
rep = flag.astype(float).copy()
rep[conf, :] = 2                      # confirmados: todas as camadas trocadas
rep[(~conf[:, None]) & flag] = 2      # acusações isoladas: só aquele tensor
cm_rep = ListedColormap(["#FFFFFF", ORG, TEAL])
fig = canvas("agregação", "Agregar: trocar pelo consenso (corr-soft)",
             "Verde-azulado = trocado pela mediana dos aceitos. Confirmados: as 62 camadas. Acusação isolada: só aquele tensor. Todos mantêm o peso.")
ax = fig.add_axes([0.1, 0.19, 0.86, 0.65]); heat(ax, rep, cm_rep, 0, 2, rowcolors={i: ORG for i in mal})

grab(fig, 3600)

# 7) resultado
fig = canvas("", "Resultado no treino completo",
             "Cenário fiel do LaySelFL (ResNet18-GN, CIFAR-10), média de 4 sementes. Sem dados limpos no servidor e sem treino prévio.")
ax = fig.add_axes([0.05, 0.16, 0.9, 0.66]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(FancyBboxPatch((0.6, 1.2), 3.8, 3.6, boxstyle="round,pad=0.1", color="#F3F3F3"))
ax.add_patch(FancyBboxPatch((5.6, 1.2), 3.8, 3.6, boxstyle="round,pad=0.1", color="#F3F3F3"))
ax.text(2.5, 4.1, "sem defesa", ha="center", fontsize=24, color="#555555")
ax.text(2.5, 2.6, "77,7%", ha="center", fontsize=64, color=RED, fontweight="bold")
ax.text(2.5, 1.6, "de sucesso do backdoor", ha="center", fontsize=18, color="#555555")
ax.text(7.5, 4.1, "com a corroboração (corr-soft)", ha="center", fontsize=22, color="#555555")
ax.text(7.5, 2.6, "5,5%", ha="center", fontsize=64, color=GRN, fontweight="bold")
ax.text(7.5, 1.6, "acurácia: −1,9 pontos sem ataque", ha="center", fontsize=18, color="#555555")
ax.annotate("", xy=(5.4, 3.0), xytext=(4.6, 3.0), arrowprops=dict(arrowstyle="-|>", color=ORG, lw=4, mutation_scale=30))
grab(fig, 4500)

frames[0].save("pipeline_defesa.gif", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
print(len(frames), "quadros", sum(durs) / 1000, "s")
