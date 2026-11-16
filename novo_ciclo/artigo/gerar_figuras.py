#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera as figuras do artigo (M11) a partir dos resultados REAIS e já concluídos do
laboratório (fl-robustness-lab/results/). Rótulos em português.

Saídas: figuras/*.pdf (vetorial, para o LaTeX) + figuras/*.png (prévia) e
        dados/*.csv (números usados em cada figura, para reprodutibilidade).

Uso (na raiz do laboratório ou aqui — o caminho é resolvido automaticamente):
    .venv/bin/python gerar_figuras.py
"""
import json, glob, os, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
FIG = HERE / "figuras"; FIG.mkdir(exist_ok=True)
DAT = HERE / "dados";   DAT.mkdir(exist_ok=True)
# o laboratório fica em ../fl-robustness-lab relativo a este arquivo
LAB_CANDIDATES = [HERE.parent / "fl-robustness-lab", Path.cwd(), Path.cwd() / "fl-robustness-lab"]
LAB = next((p for p in LAB_CANDIDATES if (p / "results").is_dir()), None)
if LAB is None:
    raise SystemExit("Não encontrei results/ do laboratório. Rode a partir de novo_ciclo/artigo/.")
RES = LAB / "results"
print("[lab]", LAB)

# ----------------------------------------------------------------------------- estilo
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
# paleta segura para daltônicos (Okabe-Ito)
AZUL, LARANJA, VERDE, VERMELHO, ROXO, CINZA = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#8a8a8a"

def salvar(fig, nome):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{nome}.{ext}")
    plt.close(fig)
    print("  ->", FIG / f"{nome}.pdf")

def dump_csv(nome, header, rows):
    with open(DAT / f"{nome}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ============================================================================= FIG 1
# ASR final por agregação (4 sementes), com barras de erro. Números auditados (painel).
def fig_rigor():
    # (nome, média, desvio, contém)  — ASR final do backdoor (%), sementes 42-45
    dados = [
        ("Sem defesa",            77.7, 10.4, "ref"),
        ("Nossa — expurgo",        4.3,  1.3, "ok"),
        ("Nossa — corr-soft",      5.5,  2.0, "ok"),
        ("Nossa — corr",          10.5,  6.3, "ok"),
        ("Nossa — só por tensor", 61.8, 20.1, "no"),
        ("LASA",                  21.7,  7.6, "no"),
        ("AlignIns",              73.4, 11.7, "no"),
        ("Multi-Krum",            76.0, 10.2, "no"),
        ("FedSurrogate",          70.6,  8.9, "no"),
        ("FLAME",                 75.3,  4.6, "no"),
        ("RFLDefense",            88.4,  2.8, "no"),
    ]
    dump_csv("rigor_asr", ["agregacao", "asr_media", "asr_desvio", "contem"], dados)
    dados = dados[::-1]  # barh de baixo p/ cima
    nomes = [d[0] for d in dados]; med = [d[1] for d in dados]; dev = [d[2] for d in dados]
    cor = {"ref": CINZA, "ok": VERDE, "no": VERMELHO}
    cores = [cor[d[3]] for d in dados]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(nomes))
    ax.barh(y, med, xerr=dev, color=cores, edgecolor="white", capsize=3, error_kw=dict(lw=1, ecolor="#444"))
    ax.axvline(20, ls="--", lw=1.2, color="#222")
    ax.text(20.8, 0.1, "limiar de contenção (20%)", rotation=90, va="bottom", ha="left", fontsize=8, color="#222")
    ax.set_yticks(y); ax.set_yticklabels(nomes)
    ax.set_xlabel("ASR final do backdoor (%)  —  média $\\pm$ desvio, 4 sementes")
    ax.set_xlim(0, 100)
    for yi, m in zip(y, med):
        ax.text(min(m + dev[yi] + 1.5, 94), yi, f"{m:.1f}", va="center", ha="left", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=VERDE, label="nossa família (contém)"),
                       Patch(color=VERMELHO, label="falha"),
                       Patch(color=CINZA, label="sem defesa")],
              loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8)
    ax.set_title("Contenção do LaySelFL (CVPR 2026) — cenário fiel")
    salvar(fig, "fig_rigor_asr")

# ============================================================================= FIG 2 e 3
# A partir do agregador-sombra (lg_shadow): contagem de corroboração por cliente/rodada.
def carregar_contagens(seeds=(42, 43, 44, 45), z_thr=3.0, janela=(80, 200)):
    """Retorna (counts_mal, counts_ben): listas de contagens por (rodada,cliente) na janela do ataque."""
    cmal, cben = [], []
    for s in seeds:
        f = RES / f"fase8e/raw/laysel_paper__lg_shadow__a0.9_s{s}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        mal = set(d["malicious_ids"]); n = d["num_clients"]
        for r, round_diag in enumerate(d["diag_log"]):
            if not (janela[0] <= r <= janela[1]):
                continue
            count = np.zeros(n)
            for g, sig in round_diag.items():
                zs = np.maximum(np.array(sig["z_sim"]), 0.0)
                zn = np.maximum(np.array(sig["z_norm"]), 0.0)
                count += (zs + zn) > z_thr
            for i in range(n):
                (cmal if i in mal else cben).append(count[i])
    return np.array(cmal), np.array(cben)

def fig_curva_T(cmal, cben):
    Ts = np.arange(1, 31)
    P, R, F = [], [], []
    for T in Ts:
        tp = np.sum(cmal >= T); fn = np.sum(cmal < T); fp = np.sum(cben >= T)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        P.append(prec); R.append(rec); F.append(f1)
    P, R, F = map(np.array, (P, R, F))
    dump_csv("curva_T", ["T", "precisao", "revocacao", "f1"], list(zip(Ts, P.round(3), R.round(3), F.round(3))))
    Tbest = int(Ts[np.argmax(F)])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(Ts, R, color=AZUL, marker="o", ms=3, label="revocação")
    ax.plot(Ts, P, color=LARANJA, marker="s", ms=3, label="precisão")
    ax.plot(Ts, F, color=VERDE, lw=2.2, marker="^", ms=3, label="$F_1$")
    for T in (8, Tbest):
        ax.axvline(T, ls=":", lw=1, color="#888")
    ax.annotate(f"$T{{=}}8$: $F_1$={F[7]:.2f}", (8, F[7]), textcoords="offset points",
                xytext=(6, -14), fontsize=8, color=VERDE)
    ax.annotate(f"pico $T{{=}}{Tbest}$: $F_1$={F.max():.2f}", (Tbest, F.max()),
                textcoords="offset points", xytext=(6, 8), fontsize=8, color=VERDE)
    ax.set_xlabel("limiar de corroboração $T$ (nº de tensores que acusam)")
    ax.set_ylabel("detecção (maliciosos × honestos)")
    ax.set_ylim(0, 1.05); ax.set_xlim(1, 30)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("Curva de detecção pelo limiar $T$")
    salvar(fig, "fig_curva_T")
    return Tbest, F[7], F.max()

def fig_corroboracao(cmal, cben):
    dump_csv("corroboracao_dist", ["grupo", "mediana", "p10", "p90", "n"],
             [["malicioso", np.median(cmal), np.percentile(cmal, 10), np.percentile(cmal, 90), len(cmal)],
              ["honesto",   np.median(cben), np.percentile(cben, 10), np.percentile(cben, 90), len(cben)]])
    bins = np.arange(0, max(cmal.max(), cben.max()) + 2) - 0.5
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.hist(cben, bins=bins, density=True, color=AZUL, alpha=0.75, label=f"honestos (mediana {np.median(cben):.0f})")
    ax.hist(cmal, bins=bins, density=True, color=VERMELHO, alpha=0.65, label=f"maliciosos (mediana {np.median(cmal):.0f})")
    ax.axvline(8, ls="--", lw=1.2, color="#222"); ax.text(8.3, ax.get_ylim()[1]*0.9, "$T{=}8$", fontsize=8)
    ax.set_xlabel("tensores que acusam o cliente por rodada")
    ax.set_ylabel("densidade")
    ax.legend(fontsize=8)
    ax.set_title("Corroboração entre camadas separa os grupos")
    ax.set_xlim(-0.5, min(bins[-1], 40))
    salvar(fig, "fig_corroboracao")

# ============================================================================= FIG 4
# Sobrecusto de agregação (% do tempo da rodada) — da tabela OVERHEAD.md (GPU).
def fig_overhead():
    # modelo -> {defesa: pct}  (agregação em GPU, % do tempo da rodada)
    modelos = ["ResNet18-GN\nCIFAR-10", "VGG19-GN\nCIFAR-10", "ResNet18-GN\nFMNIST"]
    series = {  # ordem: nossa, FLAME, Multi-Krum, AlignIns, RFLDefense, LASA
        "Nossa proposta": [4.2, 6.5, 3.6],
        "FLAME":          [1.9, 2.8, 1.6],
        "Multi-Krum":     [0.9, 1.8, 1.0],
        "AlignIns":       [27.6, 49.4, 26.6],
        "RFLDefense":     [36.4, 49.5, 36.3],
        "LASA":           [51.9, 93.2, 50.1],
    }
    dump_csv("overhead", ["defesa"] + modelos, [[k] + v for k, v in series.items()])
    cores = [VERDE, AZUL, "#56B4E9", LARANJA, ROXO, VERMELHO]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(modelos)); w = 0.13
    for i, (k, v) in enumerate(series.items()):
        ax.bar(x + (i - 2.5) * w, v, w, color=cores[i], label=k, edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(modelos)
    ax.set_ylabel("sobrecusto de agregação\n(% do tempo da rodada)")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.set_title("Custo por rodada — nossa proposta vs. defesas robustas (GPU)")
    salvar(fig, "fig_overhead")

# ============================================================================= FIG 5
# Generalidade: ASR sem defesa (sombra) vs. nossa melhor variante, por arquitetura/gatilho.
def fig_generalidade():
    def asr(p):
        d = json.load(open(RES / p)); return d["history"][-1]["backdoor_asr"]
    linhas = [  # (rótulo, json_sem_defesa(sombra), json_nossa_melhor, legenda_variante)
        ("ResNet18\nLaySelFL", None, None, "corr"),  # usa número auditado da tabela de rigor
        ("VGG19\nLaySelFL", "fase8f/raw/laysel_vgg19gn_cifar10__lg_shadow__a0.9_s42.json",
                            "fase8f/raw/laysel_vgg19gn_cifar10__layer_guard_corr_T2__a0.9_s42.json", "corr $T{=}2$"),
        ("EfficientNet\nLaySelFL", "fase8f/raw/laysel_efficientnetgn_cifar10__lg_shadow__a0.9_s42.json",
                            "fase8f/raw/laysel_efficientnetgn_cifar10__layer_guard_corr_soft_T2__a0.9_s42.json", "corr-soft $T{=}2$"),
        ("ResNet18\nA3FL CIFAR", "fase8f/raw/laysel_a3fl_resnet18gn_cifar10__lg_shadow__a0.9_s42.json",
                            "fase8f/raw/laysel_a3fl_resnet18gn_cifar10__layer_guard_corr__a0.9_s42.json", "corr"),
        ("ResNet18\nA3FL FMNIST", "fase8f/raw/laysel_a3fl_resnet18gn_fmnist__lg_shadow__a0.9_s42.json",
                            "fase8f/raw/laysel_a3fl_resnet18gn_fmnist__layer_guard_corr__a0.9_s42.json", "corr"),
    ]
    rot, sem, nossa, leg = [], [], [], []
    for r, js, jn, lg in linhas:
        rot.append(r); leg.append(lg)
        if js is None:
            sem.append(82.3); nossa.append(10.5)  # ResNet18 LaySelFL: números auditados (sombra/portão e corr 4 sementes)
        else:
            sem.append(asr(js)); nossa.append(asr(jn))
    dump_csv("generalidade", ["caso", "asr_sem_defesa", "asr_nossa", "variante"],
             list(zip(rot, np.round(sem, 1), np.round(nossa, 1), leg)))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(rot)); w = 0.38
    ax.bar(x - w/2, sem, w, color=VERMELHO, label="sem defesa", edgecolor="white")
    ax.bar(x + w/2, nossa, w, color=VERDE, label="nossa proposta", edgecolor="white")
    ax.axhline(20, ls="--", lw=1.1, color="#222")
    ax.text(len(rot)-0.5, 22, "limiar 20%", fontsize=8, ha="right")
    for xi, (a, b, l) in enumerate(zip(sem, nossa, leg)):
        ax.text(xi - w/2, a + 1.5, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + w/2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8, color=VERDE)
    ax.set_xticks(x); ax.set_xticklabels(rot, fontsize=8.5)
    ax.set_ylabel("ASR final do backdoor (%)")
    ax.set_ylim(0, 105); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Generalidade: arquiteturas e gatilho adaptativo (semente 42)")
    salvar(fig, "fig_generalidade")

# ============================================================================= main
if __name__ == "__main__":
    print("[1/5] rigor"); fig_rigor()
    print("[2-3/5] sombra: contagens de corroboração")
    cmal, cben = carregar_contagens()
    print(f"     maliciosos: n={len(cmal)} mediana={np.median(cmal):.0f} p10={np.percentile(cmal,10):.0f}")
    print(f"     honestos:   n={len(cben)} mediana={np.median(cben):.0f} p90={np.percentile(cben,90):.0f}")
    Tbest, f1_8, f1_max = fig_curva_T(cmal, cben)
    fig_corroboracao(cmal, cben)
    print(f"     F1(T=8)={f1_8:.3f}  pico F1(T={Tbest})={f1_max:.3f}")
    print("[4/5] overhead"); fig_overhead()
    print("[5/5] generalidade"); fig_generalidade()
    print("OK — figuras em", FIG)
