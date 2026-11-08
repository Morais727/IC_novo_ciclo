"""Regras de Agregação Robusta (AGRs) clássicas: FedAvg, Median, Trimmed Mean, Krum/Multi-Krum.

Contrato comum de todas as funções de defesa neste pacote:

    aggregate(deltas, weights, ctx, params) -> (agg_delta: Tensor[D], info: dict)

  deltas: Tensor [N, D] com o delta (w_local - w_global) de cada cliente do round.
  weights: Tensor [N] com o número de amostras locais de cada cliente (para médias ponderadas).
  ctx: dict com estado/contexto compartilhado (global_vector, server_holdout, model_ctor,
       device, state persistente entre rounds, etc).
  params: dict de hiperparâmetros da defesa (ex: trim_ratio, num_malicious assumido).
  info: dict com metadados úteis para métricas de detecção (ex: "excluded_indices").
"""
import torch


def fedavg(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    w = weights.float()
    w = w / w.sum()
    agg = (deltas * w.unsqueeze(1)).sum(dim=0)
    return agg, {"excluded_indices": []}


def median(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    agg = deltas.median(dim=0).values
    return agg, {"excluded_indices": []}


def trimmed_mean(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    f = params.get("num_malicious_assumed", max(1, n // 5))
    f = min(f, (n - 1) // 2)
    sorted_deltas, _ = torch.sort(deltas, dim=0)
    trimmed = sorted_deltas[f: n - f] if f > 0 else sorted_deltas
    agg = trimmed.mean(dim=0)
    return agg, {"excluded_indices": [], "trim_f": f}


def _krum_scores(deltas: torch.Tensor, f: int) -> torch.Tensor:
    n = deltas.shape[0]
    dist_sq = torch.cdist(deltas, deltas, p=2) ** 2  # [N,N]
    k = max(1, n - f - 2)
    scores = torch.zeros(n)
    for i in range(n):
        d = dist_sq[i].clone()
        d[i] = float("inf")
        nearest, _ = torch.topk(d, k=min(k, n - 1), largest=False)
        scores[i] = nearest.sum()
    return scores


def krum(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    f = params.get("num_malicious_assumed", max(1, n // 5))
    f = min(f, (n - 3) // 2) if n > 3 else 0
    scores = _krum_scores(deltas, f)
    selected = int(torch.argmin(scores))
    excluded = [i for i in range(n) if i != selected]
    return deltas[selected].clone(), {"excluded_indices": excluded, "selected_indices": [selected]}


def multi_krum(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    f = params.get("num_malicious_assumed", max(1, n // 5))
    f = min(f, (n - 3) // 2) if n > 3 else 0
    m = params.get("m", max(1, n - f))
    scores = _krum_scores(deltas, f)
    m = min(m, n)
    selected = torch.topk(scores, k=m, largest=False).indices.tolist()
    excluded = [i for i in range(n) if i not in selected]
    agg = deltas[selected].mean(dim=0)
    return agg, {"excluded_indices": excluded, "selected_indices": selected}



def rlr(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    """Robust Learning Rate (Ozdayi et al., AAAI 2021): em cada coordenada, se |soma dos sinais| < theta o passo
    é invertido (taxa -1). theta padrão = 20% dos clientes (o artigo usa 8 de 40 no CIFAR-10)."""
    n = deltas.shape[0]
    theta = params.get("theta", max(1, round(0.2 * n)))
    lr = torch.where(torch.sign(deltas).sum(0).abs() >= theta, 1.0, -1.0)
    agg, _ = fedavg(deltas, weights, ctx, params)
    return lr * agg, {"excluded_indices": [], "rlr_flipped": float((lr < 0).float().mean())}

AGGREGATORS = {
    "fedavg": fedavg,
    "rlr": rlr,
    "median": median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
}
