# Originalidade da defesa e trabalhos correlatos

**Data:** 04/11/2026 · **Autor da análise:** levantamento assistido (busca web em fontes revisadas) + código do laboratório.
**Objetivo:** garantir a **unicidade** da nossa defesa antes de escrever o artigo — mapear o que já existe, o que
é parecido e qual é exatamente a nossa contribuição defensável. **Não** copiar; posicionar com honestidade.

PDFs dos trabalhos citados estão em `~/IC/artigos/` (nomes entre parênteses em cada item). O único que não pôde ser
baixado automaticamente (proteção anti-bot da Springer) é o **RFLDefense** — baixar pelo navegador; já é concorrente
implementada no laboratório.

---

## 1. A nossa defesa (o que é, com precisão)

Família em `src/defenses/layer_guard.py`. Todas operam sobre os deltas (update − modelo global) de cada cliente,
**por tensor de parâmetro** `g` (não sobre o vetor inteiro nem só a última camada).

Passo comum a cada rodada:
1. **Centragem:** `C_i = delta_i − mediana_coordenada_j(delta_j)` (remove o "consenso" benigno).
2. **Dois sinais por tensor** `g`:
   - `z_sim(i,g)` = z robusto (mediana/MAD) de `max_{j≠i} cos(C_i[g], C_j[g])` — detecta **conluio** (atacantes
     fazem a *mesma* mudança nas *mesmas* camadas → cosseno alto entre si).
   - `z_norm(i,g)` = z robusto de `||C_i[g]||` — detecta **inflação de norma**.
   - **Escore** `s(i,g) = max(z_sim,0) + max(z_norm,0)`; marca o par (cliente i, tensor g) se `s > z_thr` (= 3).
3. **Resposta** (é aqui que as variantes diferem):

| variante (registro) | resposta | papel |
|---|---|---|
| `layer_guard_sn` ("nossa proposta" original) | agregação **cirúrgica por tensor**: média ponderada só dos **não-marcados naquele tensor**. Um cliente marcado só em `fc1` continua contribuindo no resto. | defende o LSA 24/24; **falha** no LaySelFL |
| `layer_guard` / `layerguard` | sn + **corte de norma por tensor** (reescala aceitos à mediana das normas) | ablação |
| `layer_guard_client` | **ablação:** marca em qualquer tensor → remove o **cliente inteiro** (como FLAME/Multi-Krum) | contém LaySelFL, mas custa acurácia |
| **`layer_guard_corr`** (NOVO) | **corroboração entre camadas:** conta em quantos tensores o cliente é marcado; se `≥ T` (=8), é **confirmado** e removido de **todas** as camadas; os não-confirmados seguem o corte cirúrgico | contém LaySelFL **mantendo a acurácia** (resultado-alvo) |
| **`layer_guard_corr_soft`** (NOVO) | como `corr`, mas **substitui** a contribuição do confirmado pela **mediana dos benignos** no tensor (mantém o peso de todos → massa preservada) | conter sem descartar dados do cliente |
| `lg_shadow` | diagnóstico: agrega como FedAvg e só **registra** `z_sim/z_norm/cosseno/norma` por rodada e tensor | — |

**Fundamento empírico da corroboração** (medido no `lg_shadow`, cenário LaySelFL): o atacante é acusado por uma
**mediana de ~22 tensores/rodada** (p10 = 13); o benigno, por **~1** (p90 = 6). Com `T = 8`: recall ≈ 0,95 e
**precisão ≈ 0,85** (vs. ~0,4 do flag por-tensor isolado) — ou seja, dá para purgar o cliente inteiro derrubando
~1 benigno/rodada em vez de ~7.

---

## 2. Por que a `sn` falha no LaySelFL (o problema que a `corr` resolve)

Diagnóstico completo em `fl-robustness-lab-estado.md` (memória) e nos logs da Fase 8E. Resumo:
- A `sn` **detecta o cliente** (recall 1,0 em todas as rodadas) nos tensores de **GroupNorm/atalho** (z 4–6), mas o
  backdoor viaja nos tensores **conv**, onde ~**59% das contribuições maliciosas passam abaixo do limiar**.
- Resultado: `acceptance.malicious_params ≈ 0,70` — 70% da **massa de parâmetros** maliciosos entra na agregação.
- O LaySelFL é desenhado exatamente para isso ("Eliminate Distance Differences…"): âncora β=20 (norma malic. ≤
  benigna) + alternância P=2 com dados próprios (deltas malic. **dissimilares entre si**) → derruba `z_norm` **e**
  `z_sim` nos conv. Não há mudança súbita na rod. 150; é **acúmulo** (vaza desde a rod. 80) + **aumento de furtividade**
  conforme o modelo converge.
- **Chave:** a `sn` decide **tensor-a-tensor** e joga fora a evidência cruzada. A `corr` recupera essa evidência
  (o atacante se denuncia em ~22 tensores) e purga o cliente inteiro — fechando o vazamento nos conv.

---

## 3. Trabalhos correlatos (o que já existe)

Busca em arXiv/USENIX/NDSS/IEEE/ACM/Springer (04/11/2026). Classificados por proximidade.

### 3.1 Mais próximos (risco de sobreposição — ler na íntegra antes de submeter)

- **FLD — Federated Layer Detection** (`FLD_Federated_Layer_Detection_arxiv2303.00302.pdf`).
  Primeira defesa *fine-grained* **por camada**: atribui um score de *outlier* a cada camada (baseado em densidade,
  não distância) e usa **MAD** (desvio absoluto mediano) para decidir se o **modelo inteiro** é anômalo e descartá-lo.
  **Diferença nossa:** (a) sinal = z robusto de **cosseno de deltas centrados + norma** por **tensor**; (b) resposta
  **cirúrgica por tensor** (mantém o cliente nas camadas limpas), não descarte do modelo; (c) **corroboração por
  contagem** de tensores para gatilhar a purga. ⚠️ **Não reivindicar "primeira defesa por camada"** — é da FLD.

- **Representative-Attention Defense** (`RepresentativeAttention_Defense_FL_arxiv2505.10297.pdf`).
  Usa **mediana + IQR** (estatística robusta) de um escore **de cosseno** + **MAD** contra inflação de norma — isto é
  **muito próximo** do nosso `z_sim + z_norm` com estatística robusta. ⚠️ Maior risco de sobreposição no *score*.
  **Diferença nossa:** granularidade **por tensor** + **corroboração entre camadas** + foco no cenário de **ataque
  camada-seletivo** (LaySelFL/LP), que é o que quebra as defesas de score agregado.

### 3.2 Correlatos diretos (comparar/baseline)

- **RFLDefense** (Springer/Cybersecurity 2026, DOI 10.1186/s42400-026-00679-z — *baixar manual*).
  *Layer-wise*: similaridade local–global + **consistência de sinal por camada** + agregação ponderada robusta; não
  precisa de dados limpos. **É uma das nossas concorrentes** (e perde para o LaySelFL na Fase 8E). Diferença: usa sinal
  de consistência/peso; a nossa usa cosseno par-a-par + norma em z robusto **com voto entre camadas**.
- **CosDefense / Client Filtering** (`CosDefense_ClientFiltering_arxiv2304.00160.pdf`).
  Cosseno da **última camada** entre cada update e o global; limiar por **média**. Uma camada só; a nossa usa todos os
  tensores, par-a-par, em z **robusto**, com resposta cirúrgica.
- **FoolsGold** (`FoolsGold_SybilsFL_arxiv1808.04866.pdf`).
  Cosseno de **históricos** de gradiente para detectar *sybils* (updates parecidos) + ajuste de learning rate. Nosso
  sinal de similaridade é conceitualmente parente, mas é de **deltas centrados pela mediana, por tensor, em uma rodada**,
  e a decisão é **purga por corroboração**, não reponderação de LR. (Custo O(n²) como o nosso passo de cosseno.)
- **FL-PLAS** (`FL-PLAS_PartialLayerAggregation_arxiv2505.12019.pdf`).
  **Estrutural**: mantém a camada classificadora **local** (não agrega), para o backdoor não se propagar. Não há
  detecção de cliente; mecanismo ortogonal ao nosso.

### 3.3 Contexto

- **Survey de ataques/defesas de envenenamento em FL** (`Survey_FL_Poisoning_Attacks_Defenses_arxiv2306.03397.pdf`):
  confirma que *clustering por cosseno* é abordagem comum para separar maliciosos; situa as famílias.
- **LP / Backdoor-Critical Layers** (Zhuang et al., ICLR 2024; arXiv 2308.04466 — já referenciado no lab como ataque):
  mostra que atacar poucas camadas críticas evade defesas de similaridade. É o **ataque** que motiva a granularidade
  por camada da defesa.

---

## 4. Veredito de originalidade

**Os ingredientes já existem na literatura** — detecção por camada (FLD, RFLDefense), z robusto de cosseno+norma
(Representative-Attention), cosseno para conluio (FoolsGold), cosseno de última camada (CosDefense). **Portanto não se
pode reivindicar "primeira defesa por/por camada".**

**A contribuição defensável é a COMBINAÇÃO**, que não apareceu em nenhum trabalho encontrado:
1. z robusto de **(cosseno de deltas centrados pela mediana) + (norma)** **por tensor**;
2. **agregação cirúrgica por tensor** (não descarta o cliente inteiro por padrão);
3. **corroboração entre camadas** — purga o cliente de todas as camadas **somente** quando `≥ T` tensores o acusam
   (voto), fundamentada na separação medida (atacante ~22 × benigno ~1 tensores);
4. **robustez a ataques camada-seletivos de 2026 (LaySelFL, LP)** que derrubam a classe inteira de defesas SOTA — e
   com **overhead baixo** (≤ 6,1% sobre o FedAvg).

**Ações obrigatórias antes de submeter (no roteiro do painel):**
- [ ] Ler **FLD (2303.00302)** e **Representative-Attention (2505.10297)** na íntegra e escrever um comparativo
      ponto-a-ponto (mecanismo, granularidade, decisão, custo, ataques avaliados).
- [ ] Rodar FLD e CosDefense como baselines adicionais no cenário LaySelFL (hoje temos LASA/AlignIns/RFLDefense/
      FLAME/Multi-Krum/FedSurrogate).
- [ ] Seção de *related work* honesta deixando claro: ingredientes conhecidos → contribuição = combinação + robustez
      ao ataque camada-seletivo.

---

## 5. Evidência experimental de apoio (resumo; detalhes no painel)

- **LSA (ICC 2026):** `sn` defende **24/24** em 5 α; melhor concorrente (LASA) 8/24.
- **LaySelFL (CVPR 2026, cenário do artigo, ResNet18-GN/CIFAR):** ataque pega (portão ASR 82%). `sn` **não contém**
  (ASR final 76,7%). Head-to-head das 8 SOTA: só a **LASA** fecha < limiar (13,5% final; pico 27,8% durante o ataque);
  as demais 71–87%. **`corr`/`client` (parciais rod. 150–160) seguram** onde a `sn` rompeu: `corr` BA ~7% com MA ~74%
  (≈ sn); `client` ~3–6% com MA ~66–70% (custo de acurácia).
- **Generalidade:** o ataque também pega em FMNIST (95,8%) e VGG19-GN (parcial 27%); portões A3FL em curso.
- **Custo:** agregação da nossa proposta ≤ 6,1% do tempo da rodada sobre o FedAvg (vs. LASA até 93%).

---

## 5-bis. Comparativo PONTO-A-PONTO (lido na íntegra — 04/11/2026)

Lidos integralmente via `gs txtwrite`: **FLD** (Alg. 1–3) e **FeRA/Representative-Attention** (seção de método).

### FLD — Federated Layer Detection (o mais próximo; **atenção**)
Mecanismo exato (Algoritmos 2 e 3 do paper):
- **Layer Scoring:** para cada camada `j`, calcula o **COF** (Connectivity-based Outlier Factor, detector de outlier
  por **densidade**) do parâmetro daquela camada de cada cliente, entre os clientes → score `s^j_i`.
- **Anomaly Detection:** por camada, `Me = mediana_i(s^j_i)`, `MAD = mediana_i(|s^j_i − Me|)`; marca a camada `j` do
  cliente `i` se `s^j_i ≥ Me + µ·MAD` ou `≤ Me − µ·MAD`, com **µ = 3**; incrementa `Outlierflag_i`.
- **Decisão:** se `Outlierflag_i < total/2` (menos de **metade** das camadas marcadas) → benigno; senão → **descarta o
  modelo inteiro**. Agrega (FedAvg) só os benignos. Sem conhecer o nº de atacantes; sem dados limpos no servidor.

| dimensão | **Nossa (`sn` + `corr`/`corr_soft`)** | **FLD** |
|---|---|---|
| substrato | deltas de parâmetro (update − global) | parâmetros do modelo enviado |
| granularidade | **por tensor** | por camada |
| sinal por unidade | z robusto de **cosseno par-a-par de deltas centrados pela mediana** + z robusto de **norma** | **COF** (outlier por densidade) do tensor entre clientes |
| limiar | z robusto, fator **3** (igual) | MAD, `µ = 3` (igual) |
| agregação de camadas → cliente | `corr`: **conta** tensores marcados; purga se `≥ T` (=8) | **conta** camadas marcadas; descarta se `≥ total/2` |
| resposta p/ NÃO-confirmado | **cirúrgica por tensor** (mantém o cliente nas camadas limpas) — *exclusivo nosso* | n/a (decisão é binária: modelo entra ou sai) |
| variante "soft" | `corr_soft`: substitui pelo **consenso (mediana benigna)** — *exclusivo nosso* | n/a |
| nº de atacantes | não precisa | não precisa |
| dados limpos no servidor | **não precisa** | não precisa |
| ataques avaliados | **LaySelFL/LP (2024–2026, camada-seletivos)** + LSA | DBA, "A Little Is Enough" (≤ 2020–2023) |

**Conclusão honesta:** a **corroboração por contagem de camadas + limiar MAD(µ=3) + descarte do modelo** é,
essencialmente, o mecanismo de *Anomaly Detection* da FLD. **Não reivindicar isso como novo.** O que permanece
nosso: (1) o **sinal por tensor** (cosseno par-a-par de deltas **centrados pela mediana** + norma — a FLD usa COF de
densidade, não similaridade/norma); (2) a **agregação cirúrgica por tensor** (`sn`) — a FLD é binária por modelo;
(3) a variante **`corr_soft`** (substituição por consenso); (4) a avaliação contra **ataques camada-seletivos de
2026 (LaySelFL)** que a FLD não considerou. **Posicionamento:** citar a FLD com destaque; enquadrar a `corr` como
"porta de corroboração estilo-FLD aplicada ao nosso sinal por tensor", e situar a contribuição central na **resposta
cirúrgica (`sn`) + `corr_soft` + robustez ao LaySelFL + custo baixo**. (Opção: adotar o limiar relativo "fração de
camadas" da FLD, em vez do `T` absoluto, para evitar um ponto de ataque por hiperparâmetro.)

### FeRA / Representative-Attention (menos próxima do que a busca sugeria)
- **Substrato:** **representações/ativações** dos clientes, extraídas com um **root dataset limpo no servidor** (≠ nosso,
  que usa só os deltas e a mediana coordenada entre clientes — **sem dados limpos**).
- **Sinais:** "atenção" espectral/espacial (variância de representação), **alinhamento direcional** (cosseno com a
  trajetória global), **similaridade mútua** (cosseno par-a-par de representações) e **inflação de norma** (norma
  espectral). Dois filtros: *consistência* e *norm-inflation*.
- **Decisão:** **pesos contínuos** (não aceita/rejeita binário), para evitar ataques perto do limiar. Nível de modelo
  (não por camada/tensor).

| dimensão | **Nossa** | **FeRA** |
|---|---|---|
| substrato | deltas de parâmetro, por tensor | **representações/ativações** (nível de modelo) |
| dados limpos no servidor | **não precisa** | **precisa** (root dataset) |
| granularidade | por tensor + corroboração entre camadas | por modelo |
| similaridade | cosseno par-a-par de deltas centrados, por tensor | cosseno de representações (mútua + direcional) |
| norma | z robusto de norma do delta por tensor | inflação de norma espectral |
| decisão | cirúrgica por tensor / purga por corroboração | reponderação contínua |

**Conclusão:** compartilha o par conceitual *(similaridade + norma)* com estatística robusta, mas o **substrato
(ativações + root dataset)** e a **granularidade (modelo, não tensor)** são fundamentalmente diferentes. Baixo risco de
sobreposição; citar como defesa de *representação* correlata, destacando que a nossa **não usa dados limpos** e age
**por tensor**.

### Veredito atualizado
A peça que eu antes listei como novidade — "corroboração entre camadas por contagem" — **está substancialmente
antecipada pela FLD**. A originalidade sólida e defensável concentra-se em: **(a) agregação cirúrgica por tensor
(`sn`)**, **(b) o sinal por tensor = cosseno par-a-par de deltas centrados + norma**, **(c) `corr_soft`
(substituição por consenso)**, **(d) robustez demonstrada contra o LaySelFL/LP** (ataques camada-seletivos de 2026
que derrubam a classe SOTA inteira), **(e) sem dados limpos no servidor e com overhead ≤ 6,1%**. A `corr` deve ser
apresentada honestamente como "gate de corroboração estilo-FLD sobre o nosso sinal", não como ideia inédita.

---

## 6. Referências (arquivos em `~/IC/artigos/`)

| # | Trabalho | Local | Arquivo |
|---|---|---|---|
| 1 | FLD — Federated Layer Detection | arXiv 2303.00302 | `FLD_Federated_Layer_Detection_arxiv2303.00302.pdf` |
| 2 | Representative-Attention Defense (FL backdoor) | arXiv 2505.10297 | `RepresentativeAttention_Defense_FL_arxiv2505.10297.pdf` |
| 3 | CosDefense / Secure FL via Client Filtering | arXiv 2304.00160 | `CosDefense_ClientFiltering_arxiv2304.00160.pdf` |
| 4 | FL-PLAS — Partial Layer Aggregation | arXiv 2505.12019 | `FL-PLAS_PartialLayerAggregation_arxiv2505.12019.pdf` |
| 5 | FoolsGold — limitations of FL in sybil settings | arXiv 1808.04866 (RAID 2020) | `FoolsGold_SybilsFL_arxiv1808.04866.pdf` |
| 6 | Survey — FL poisoning attacks & defenses | arXiv 2306.03397 | `Survey_FL_Poisoning_Attacks_Defenses_arxiv2306.03397.pdf` |
| 7 | RFLDefense | Springer/Cybersecurity 2026, DOI 10.1186/s42400-026-00679-z | *(baixar manual — anti-bot)* |
| 8 | LaySelFL (ataque alvo) | Li et al., CVPR 2026 | `LaySelFL_Li_CVPR2026_Eliminate_Distance_Differences.pdf` |
| 9 | LP / Backdoor-Critical Layers (ataque) | Zhuang et al., ICLR 2024 / arXiv 2308.04466 | *(já no acervo: `2689_Backdoor_Federated_Learni.pdf`)* |
