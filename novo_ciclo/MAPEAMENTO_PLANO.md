# Mapeamento do plano de 12 meses — estado em 10/11/2026

Entregáveis mensais do Plano de Bolsa (H.IAAC, Meta 3). Legenda: ✅ entregue · 🟩 rascunho pronto · 🟨 substância pronta, falta consolidar/escrever · 🟧 falta implementar/rodar.

| Mês | Entregável do plano | Estado | O que já existe | O que falta (adiantável sem GPU em *itálico*) |
|---|---|---|---|---|
| M1–M3 | Infra, ataques da revisão, especificação dos sinais | ✅ | Relatórios 1–3 | — |
| M4 · out | Conjunto de ataques injetáveis (LP, LaySelFL, A3FL) | ✅ | Relatório 4 (.docx) | *atualizar a tabela de rigor para 4 sementes* |
| **M5 · nov** | **Protótipo do agente (perfil, concentração, sonda)** | **🟩** | **Relatório 5 (.docx) montado** | revisão do texto |
| **M6 · dez** | **Integração à agregação + resposta cirúrgica + parcial** | **🟩** | **Relatório 6 parcial (.docx) montado** | revisão; *figuras* |
| M7 · jan | Envenenamento de dados (label flip, pixel, DBA); H3 | 🟩 | **Relatório 7 (.docx) montado**; H3 (sonda com gatilho aleatório) refutada e documentada | revisão |
| M8 · fev | Envenenamento de modelo (LP, LF, IPM, LIE, MinMax) | 🟩 | **Relatório 8 (.docx) montado**, inclui validação no repo do LP (20% zera, 10% não) | revisão |
| M9 · mar | Falsos positivos e impacto; limiar por predição conforme | 🟧 | α 0,1 medido; curva de detecção por T | implementar **calibração do limiar por predição conforme** (dev + validação) |
| M10 · abr | Ataque que conhece a defesa (H4) | 🟨 | **adversário adaptativo implementado (fase8g)** | rodar (na fila) + *analisar/escrever* |
| M11 · mai | Artigo submetido | 🟩 | **rascunho LaTeX iniciado** (`artigo/main.tex`): abstract, método e resultados com a tabela de rigor | completar related work; inserir resultados fase8g/9A; figuras; template da conferência |
| M12 · jun | Relatório final + repositório reprodutível | 🟧 | repo do laboratório + resultados | *consolidar relatório final; empacotar reprodutibilidade* |

## Leitura
- **Entregas formais adiantadas nesta sessão:** M5 (rascunho completo) e M6 (parcial completo). O M4 convém reemitir com a tabela de 4 sementes.
- **Já adiantado fora de ordem:** o **M10 (H4, atacante que conhece a defesa)** está implementado 5 meses antes (`src/attacks/recent.py::_evade_corr` + `experiments/fase8g_adaptive.py`), aguardando só a GPU.
- **Maior bloco não-GPU pendente:** o **artigo (M11)** — todo o núcleo experimental já existe; é escrita.
- **Único item que exige implementação nova (não-GPU em boa parte):** a **calibração por predição conforme (M9)**.

## Próximos passos não-GPU sugeridos (ordem)
1. Revisar os rascunhos M5 e M6.
2. Reemitir o Relatório 4 com a tabela de rigor de 4 sementes.
3. Começar o **artigo (M11)**: esqueleto + método + resultados (tudo com dados prontos).
4. Prototipar a **calibração por predição conforme (M9)** — dev puro, validação leve.
