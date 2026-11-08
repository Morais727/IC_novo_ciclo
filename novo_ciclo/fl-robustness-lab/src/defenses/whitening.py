"""Indicador multi-métrica com branqueamento (whitening).

Combina L1, L2 e similaridade de cosseno (relativas à média dos deltas do
round) em um vetor de features 3-D por cliente; aplica branqueamento
(normalização pela inversa da matriz de covariância -> distância de
Mahalanobis) para gerar um escore de divergência adaptativo; clientes com
escore acima do quantil (1 - fração_suspeita) são excluídos da agregação.
"""
import torch


def _features(deltas: torch.Tensor) -> torch.Tensor:
    ref = deltas.mean(dim=0, keepdim=True)
    l1 = (deltas - ref).abs().sum(dim=1)
    l2 = (deltas - ref).norm(dim=1)
    cos = torch.nn.functional.cosine_similarity(deltas, ref.expand_as(deltas), dim=1)
    return torch.stack([l1, l2, cos], dim=1)  # [N,3]


def multi_metric_whitening(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    suspect_fraction = params.get("suspect_fraction", 0.25)

    feats = _features(deltas)
    mean = feats.mean(dim=0, keepdim=True)
    centered = feats - mean
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov = cov + torch.eye(3) * 1e-4  # regularização para invertibilidade
    cov_inv = torch.linalg.inv(cov)

    mahalanobis = torch.einsum("ni,ij,nj->n", centered, cov_inv, centered).clamp_min(0)
    scores = mahalanobis.sqrt()

    k_exclude = max(0, int(round(n * suspect_fraction)))
    if k_exclude > 0:
        excluded = torch.topk(scores, k=k_exclude, largest=True).indices.tolist()
    else:
        excluded = []
    selected = [i for i in range(n) if i not in excluded]
    if not selected:
        selected = list(range(n))
    agg = deltas[selected].mean(dim=0)
    return agg, {"excluded_indices": excluded, "selected_indices": selected, "scores": scores.tolist()}
