"""FoolsGold (Fung et al., 2018): penaliza clientes cujas atualizações são
historicamente muito correlacionadas entre si (indício de colusão de
sybils/backdoor). Requer estado persistente entre rounds (histórico
acumulado de atualizações por cliente), mantido em ctx["state"]["foolsgold_history"].
"""
import torch


def foolsgold(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    kappa = params.get("kappa", 1.0)
    eps = 1e-6

    state = ctx.setdefault("state", {})
    hist = state.setdefault("foolsgold_history", torch.zeros_like(deltas))
    if hist.shape != deltas.shape:
        hist = torch.zeros_like(deltas)
    hist = hist + deltas
    state["foolsgold_history"] = hist

    norm = hist.norm(dim=1, keepdim=True).clamp_min(eps)
    normalized = hist / norm
    cs = normalized @ normalized.T  # [N,N] similaridade de cosseno
    cs.fill_diagonal_(0.0)

    max_sim, max_idx = cs.max(dim=1)

    # Pardoning: se j é mais "central" (maior max_sim) que i, reduz a penalidade de i por j
    cs_adjusted = cs.clone()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if max_sim[j] > max_sim[i] and max_sim[j] > eps:
                cs_adjusted[i, j] *= max_sim[i] / max_sim[j]

    wv = 1.0 - cs_adjusted.max(dim=1).values
    wv = wv.clamp(0.0, 1.0)
    if wv.max() > eps:
        wv = wv / wv.max()
    wv = wv.clamp(eps, 1 - eps)
    logit = kappa * (torch.log(wv / (1 - wv)) + 0.5)
    alpha = logit.clamp(0.0, 1.0)

    if alpha.sum() <= eps:
        alpha = torch.ones(n) / n
    else:
        alpha = alpha / alpha.sum()

    agg = (deltas * alpha.unsqueeze(1)).sum(dim=0)
    excluded = [i for i in range(n) if alpha[i] < (0.1 / n)]
    return agg, {"excluded_indices": excluded, "weights": alpha.tolist()}
