"""Comparadores mais próximos da corroboração entre camadas (prioridade 1 do roteiro de 12/11/2026).

FLD — Federated Layer Detection (Lin, Wu, Liao e Zhou, arXiv 2303.00302, 2023), reimplementada a partir dos
Algoritmos 1–3 do artigo (sem código público no acervo):
  1. Layer Scoring (Alg. 2): para cada camada j, o escore de outlier COF (Connectivity-based Outlier Factor,
     Tang et al. 2002) de cada modelo local w_i^j, comparando os n modelos da rodada;
  2. Anomaly Detection (Alg. 3): por camada, Me = mediana dos escores e MAD = mediana(|S − Me|); a camada do
     cliente i é anômala se S >= Me + µ·MAD (flag1) ou S <= Me − µ·MAD (flag2), µ = 3 (padrão do artigo);
     Outlierflag_i soma as camadas anômalas;
  3. benigno se Outlierflag_i < total/2; o global é a média simples dos benignos (Alg. 1, linha 13).
Escolhas que o artigo não fixa, definidas ANTES de rodar:
  - "camada" = cada tensor de parâmetros treináveis (pesos, vieses, GroupNorm), a mesma granularidade da
    nossa defesa; total = nº de tensores;
  - COF no estilo do pyod (method='fast', n_neighbors padrão 20, limitado a n − 1): o caminho SBN é aproximado
    pela ordem de distância; distância euclidiana entre os modelos locais da camada, que é igual à distância
    entre os deltas (o global é comum a todos);
  - se todos forem marcados, agrega todos (nunca devolve agregado vazio).

CosDefense (Yan et al., ICLR 2023 Workshop on Backdoor Attacks and Defenses, arXiv 2304.00160), fiel ao
Algoritmo 1 do apêndice C:
  cs_i = |cos(θ_t,L , g_i,L)|, L = peso (não viés) da última camada do modelo; normalização mín-máx;
  limiar = média dos cs normalizados; benigno se cs_i < limiar; agrega os benignos com FedAvg (média
  ponderada pelo nº de amostras; o artigo diz que vale com qualquer agregador).
"""
import numpy as np
import torch

from src.defenses.lasa_alignins import _shapes


def _cof(dist: np.ndarray, k: int) -> np.ndarray:
    """COF (versão 'fast' do pyod) a partir da matriz de distâncias n×n."""
    n = dist.shape[0]
    k = max(1, min(k, n - 1))
    ac = np.zeros(n); nbr = np.zeros((n, k), dtype=int)
    w = np.array([2.0 * (k + 1 - (h + 1)) / ((k + 1) * k) for h in range(k)])
    for i in range(n):
        path = np.argsort(dist[i], kind="stable")
        path = np.concatenate([[i], path[path != i]])          # o próprio ponto primeiro
        nbr[i] = path[1:k + 1]
        cost = np.array([dist[path[j + 1]][path[:j + 1]].min() for j in range(k)])
        ac[i] = (w * cost).sum()
    cof = np.zeros(n)
    for i in range(n):
        den = ac[nbr[i]].sum()
        cof[i] = ac[i] * k / den if den > 0 else 0.0
    return np.nan_to_num(cof)


def fld(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    mu, kn = params.get("mu", 3.0), params.get("n_neighbors", 20)
    n = deltas.shape[0]
    shapes = _shapes(ctx)
    flags = np.zeros(n, dtype=int)
    for _, s, _ in shapes:
        x = deltas[:, s].double()
        d = torch.cdist(x, x).cpu().numpy()
        sc = _cof(d, kn)
        me = np.median(sc); mad = np.median(np.abs(sc - me))
        flags += ((sc >= me + mu * mad) | (sc <= me - mu * mad)).astype(int)
    total = len(shapes)
    benign = [i for i in range(n) if flags[i] < total / 2]
    if not benign:
        benign = list(range(n))
    agg = deltas[benign].mean(dim=0)
    excluded = [i for i in range(n) if i not in benign]
    return agg, {"excluded_indices": excluded, "selected_indices": benign,
                 "outlier_flags": flags.tolist(), "total_layers": total}


def cosdefense(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    shapes = _shapes(ctx)
    last = [s for _, s, dim in shapes if dim >= 2][-1]          # peso da última camada (classificador)
    gL = ctx["global_vector"].to(deltas.dtype)[last]
    uL = deltas[:, last]
    cs = (uL @ gL).abs() / (uL.norm(dim=1) * gL.norm()).clamp_min(1e-12)
    cs = cs.double().cpu().numpy()
    rng = cs.max() - cs.min()
    csn = (cs - cs.min()) / rng if rng > 0 else np.zeros(n)
    thr = csn.mean()
    benign = [i for i in range(n) if csn[i] < thr]
    if not benign:                                              # todos iguais: ninguém acima da média
        benign = list(range(n))
    w = weights[benign].float(); w = w / w.sum()
    agg = (deltas[benign] * w.unsqueeze(1).to(deltas.dtype)).sum(dim=0)
    excluded = [i for i in range(n) if i not in benign]
    return agg, {"excluded_indices": excluded, "selected_indices": benign, "cos_scores": cs.tolist()}
