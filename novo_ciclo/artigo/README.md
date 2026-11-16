# Artigo (M11) — Contenção de Backdoors com Seleção de Camadas em FL

Projeto LaTeX **em português**, pronto para o Overleaf. Todas as figuras são geradas a
partir de resultados **reais e já concluídos** do laboratório (`../fl-robustness-lab/results/`).

## Estrutura
```
artigo/
├── main.tex              # principal: preâmbulo + \input das seções (compilar ESTE)
├── secoes/               # uma seção por arquivo
│   ├── 00_resumo.tex     06_adaptativo.tex
│   ├── 01_introducao.tex 07_limitacoes.tex
│   ├── 02_relacionados.tex  08_conclusao.tex
│   ├── 03_ameaca.tex     referencias.tex
│   ├── 04_metodo.tex
│   └── 05_experimentos.tex
├── figuras/              # .pdf (vetorial, usado pelo LaTeX) + .png (prévia)
│   ├── fig_rigor_asr     fig_curva_T       fig_corroboracao
│   └── fig_overhead      fig_generalidade
├── dados/               # .csv com os números de cada figura (reprodutibilidade)
└── gerar_figuras.py     # regenera figuras/ e dados/ a partir dos results/ do lab
```

## Compilar no Overleaf
1. Suba a pasta inteira (zip) em *New Project → Upload Project*.
2. Defina **`main.tex`** como *Main document* (Menu → Main document).
3. Compilar (pdfLaTeX). O `\graphicspath{{figuras/}}` já aponta para as figuras.

## Regenerar as figuras (opcional)
Na raiz `novo_ciclo/artigo/`:
```
../fl-robustness-lab/.venv/bin/python gerar_figuras.py
```
Lê os JSONs de `results/` e reescreve `figuras/*.pdf|png` e `dados/*.csv`.

## Procedência das figuras
| Figura | O que mostra | Fonte real |
|---|---|---|
| `fig_rigor_asr` | ASR por agregação, 4 sementes | tabela de rigor (painel, auditada) |
| `fig_curva_T` | precisão/revocação/F1 vs T | `fase8e/.../lg_shadow` (sementes 42–45) |
| `fig_corroboracao` | contagem de tensores que acusam: malicioso×honesto | idem (sombra) |
| `fig_overhead` | custo de agregação por rodada | `fase8/OVERHEAD.md` |
| `fig_generalidade` | ASR sem defesa × nossa, por arquitetura/A3FL | `fase8f/raw/*` (semente 42) |

## Pendências (marcadas com `\todo{}` no texto)
- Trabalhos relacionados: citações/anos completos.
- Fig. do adversário adaptativo (fase8g, rodando nas GPUs 3–5).
- Fig. da Fase 9A (contagem de corroboração 1 vs 2 atacantes; entra com GPU livre).
- Trocar `documentclass` pelo template da conferência-alvo.
