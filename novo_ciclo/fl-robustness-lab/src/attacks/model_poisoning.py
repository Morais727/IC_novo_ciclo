"""Ataques de envenenamento de modelo "onisciente" (o adversário observa as
atualizações honestas do round corrente e otimiza sua própria atualização
em função delas — modelagem padrão em Blades/BOBA/Shejwalkar&Houmansadr).

Todas as funções recebem `benign_stack` (Tensor [n_benign, D]) e devolvem um
único vetor malicioso (Tensor [D]) replicado para todos os clientes maliciosos
colidentes no round.
"""
import torch


def gaussian_noise(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """Ruído N(0, std²). Com `norm_mult` (28/10/2026, relatório do mês 4), a norma do ruído é calibrada:
    `norm_mult` × mediana das normas dos honestos do round (std fixo derruba o modelo ou some, conforme o modelo)."""
    d = benign_stack.shape[1]
    if "norm_mult" in params:
        v = torch.randn(d)
        return v * params["norm_mult"] * benign_stack.norm(dim=1).median() / v.norm()
    return torch.randn(d) * params.get("std", 1.0)


def sign_flip(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    scale = params.get("scale", 1.0)
    return -scale * benign_stack.mean(0)


def ipm(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """Inner Product Manipulation (Xie et al.): escala negativamente a média honesta
    de forma que o produto interno com a direção real fique negativo."""
    epsilon = params.get("epsilon", 1.0)
    return -epsilon * benign_stack.mean(0)


def lie(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """A Little Is Enough (Baruch et al.): desvia da média por z desvios-padrão,
    mantendo-se dentro da variância natural dos clientes honestos."""
    z = params.get("z", 1.5)
    mean = benign_stack.mean(0)
    std = benign_stack.std(0)
    return mean + z * std


def _pairwise_max_sq_dist(stack: torch.Tensor) -> float:
    diffs = torch.cdist(stack, stack, p=2)
    return float((diffs ** 2).max())


def _pairwise_max_sum_sq_dist(stack: torch.Tensor) -> float:
    diffs = torch.cdist(stack, stack, p=2) ** 2
    return float(diffs.sum(dim=1).max())


def minmax(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """MinMax (Shejwalkar & Houmansadr): perturbação escalada via busca binária
    para que a distância máxima até qualquer benigno não ultrapasse a distância
    máxima observada entre pares de clientes honestos."""
    perturbation = params.get("perturbation", "std")
    mean = benign_stack.mean(0)
    p = -benign_stack.std(0) if perturbation == "std" else -torch.sign(mean)
    max_benign_dist_sq = _pairwise_max_sq_dist(benign_stack)

    lo, hi = 0.0, 50.0
    for _ in range(25):
        gamma = (lo + hi) / 2
        malicious = mean + gamma * p
        d_sq = float(((benign_stack - malicious) ** 2).sum(dim=1).max())
        if d_sq <= max_benign_dist_sq:
            lo = gamma
        else:
            hi = gamma
    return mean + lo * p


def minsum(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """MinSum: análogo ao MinMax, mas limita a soma das distâncias quadráticas
    até todos os benignos pela maior soma observada entre um honesto e os demais."""
    perturbation = params.get("perturbation", "std")
    mean = benign_stack.mean(0)
    p = -benign_stack.std(0) if perturbation == "std" else -torch.sign(mean)
    max_benign_sum_sq = _pairwise_max_sum_sq_dist(benign_stack)

    lo, hi = 0.0, 50.0
    for _ in range(25):
        gamma = (lo + hi) / 2
        malicious = mean + gamma * p
        sum_sq = float(((benign_stack - malicious) ** 2).sum())
        if sum_sq <= max_benign_sum_sq:
            lo = gamma
        else:
            hi = gamma
    return mean + lo * p


def byzmean(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """ByzMean (Xu et al., ICDCS 2022; usado por AlignIns e LASA), construído sobre o LIE: metade dos atacantes envia o
    vetor do LIE (v) e a outra metade envia u, escolhido para que a MÉDIA de todos os clientes seja exatamente v.
    Devolve um vetor por atacante [m, d]."""
    m = int(params.get("_n_mal", 1)); n = int(params.get("_n_total", m + benign_stack.shape[0]))
    v = lie(benign_stack, params)
    m1 = max(m // 2, 1); m2 = m - m1
    if m2 == 0:
        return v.unsqueeze(0).repeat(m, 1)
    u = (n * v - m1 * v - benign_stack.sum(0)) / m2
    return torch.stack([v] * m1 + [u] * m2)


MODEL_POISONING_ATTACKS = {
    "gaussian_noise": gaussian_noise,
    "sign_flip": sign_flip,
    "ipm": ipm,
    "lie": lie,
    "minmax": minmax,
    "minsum": minsum,
    "byzmean": byzmean,
}
