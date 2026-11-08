"""BOBA (Bao et al., 2024) — reimplementação simplificada.

Fidelidade ao artigo original: **parcial**, documentada aqui e no documento
de cenário. A ideia central é preservada (subespaço honesto via SVD robusto
+ simplex de distribuição de rótulos do servidor), mas a estimação dos
vértices do simplex usa deltas de referência por classe calculados a
partir do modelo global (em vez do procedimento exato de mínimos
quadrados restrito ao simplex descrito no artigo).

Etapa 1 (subespaço honesto): remove os clientes com norma L2 mais extrema
(trim) e ajusta uma SVD truncada de posto c-1 (c = nº de classes) sobre os
deltas restantes -> base U do subespaço honesto. Projeta TODOS os deltas
(inclusive os removidos do fit) nesse subespaço.

Etapa 2 (simplex de rótulos): usa `server_holdout` (amostras limpas
estratificadas por classe) para calcular um delta de referência r_k por
classe k; projeta {r_k} no mesmo subespaço, formando os vértices do simplex.
Para cada cliente, resolve as coordenadas baricêntricas (estimativa da
distribuição de rótulos implícita) via mínimos quadrados; clientes cuja
coordenada mínima é anomalamente negativa (abaixo de -tau) são descartados.

Os clientes enviam deltas de treino local (várias etapas de SGD, ≈ -lr·Σg),
não gradientes. Por isso r_k é o delta de um "cliente virtual" que só tem
dados da classe k: treino local a partir do modelo global, com o mesmo
otimizador/lr e o mesmo nº de passos de um cliente típico (`ctx["ref_steps"]`),
amostrando lotes do holdout da classe k. Assim os vértices ficam com o mesmo
sinal e escala dos deltas dos clientes (um gradiente único teria sinal oposto
e magnitude ~lr·passos vezes diferente, espelhando o simplex).
"""
import torch
import torch.nn.functional as F

from src.federated.utils import flatten, load_vector


def _class_reference_deltas(model_ctor, global_vector, server_holdout, num_classes, device,
                            steps: int, batch_size: int, lr: float, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    global_dev = global_vector.to(device)
    refs = []
    for k in range(num_classes):
        idx = [i for i in range(len(server_holdout)) if server_holdout[i][1] == k]
        if not idx:
            refs.append(torch.zeros_like(global_vector))
            continue
        xs = torch.stack([server_holdout[i][0] for i in idx]).to(device)
        ys = torch.full((len(idx),), k, dtype=torch.long, device=device)

        model = model_ctor().to(device)
        load_vector(model, global_dev)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model.train()
        for _ in range(steps):
            b = torch.randint(len(idx), (min(batch_size, len(idx)),), generator=gen).to(device)
            opt.zero_grad()
            F.cross_entropy(model(xs[b]), ys[b]).backward()
            opt.step()
        refs.append((flatten(model) - global_dev).detach().cpu())
    return torch.stack(refs)  # [c, D]


def boba(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n, d = deltas.shape
    num_classes = ctx["num_classes"]
    trim_ratio = params.get("trim_ratio", 0.2)
    tau = params.get("tau", 0.15)

    norms = deltas.norm(dim=1)
    order = torch.argsort(norms)
    k_trim = int(n * trim_ratio / 2)
    keep_for_fit = order[k_trim: n - k_trim] if k_trim > 0 else order
    if len(keep_for_fit) < num_classes:
        keep_for_fit = order

    fit_matrix = deltas[keep_for_fit]
    center = fit_matrix.mean(dim=0, keepdim=True)
    q = max(1, num_classes - 1)
    try:
        U, S, V = torch.svd_lowrank(fit_matrix - center, q=q)
        basis = V  # [D, q]
    except RuntimeError:
        basis = torch.eye(d)[:, :q]

    proj_all = (deltas - center) @ basis  # [N, q]

    state = ctx.setdefault("state", {})
    cache_key = "boba_ref_deltas"
    if cache_key not in state or state.get("boba_ref_round") != ctx.get("round_idx"):
        state[cache_key] = _class_reference_deltas(
            ctx["model_ctor"], ctx["global_vector"], ctx["server_holdout"], num_classes, ctx["device"],
            steps=params.get("ref_steps", ctx.get("ref_steps", 1)),
            batch_size=ctx.get("batch_size", 32), lr=ctx.get("lr", 0.01),
            seed=ctx.get("round_idx", 0),
        )
        state["boba_ref_round"] = ctx.get("round_idx")
    ref_deltas = state[cache_key]  # [c, D]
    proj_ref = (ref_deltas - center) @ basis  # [c, q]

    # resolve coordenadas baricêntricas por mínimos quadrados: proj_i ~= sum_k lambda_k * proj_ref_k, sum lambda_k = 1
    A = proj_ref.T  # [q, c]
    ones_row = torch.ones(1, num_classes)
    A_aug = torch.cat([A, ones_row], dim=0)  # [q+1, c]

    excluded = []
    keep = []
    for i in range(n):
        b = torch.cat([proj_all[i], torch.tensor([1.0])])
        lam, *_ = torch.linalg.lstsq(A_aug, b.unsqueeze(1))
        lam = lam.squeeze(1)
        if lam.min().item() < -tau:
            excluded.append(i)
        else:
            keep.append(i)

    if not keep:
        keep = list(range(n))
        excluded = []

    w = weights[keep].float()
    w = w / w.sum()
    agg = (deltas[keep] * w.unsqueeze(1)).sum(dim=0)
    return agg, {"excluded_indices": excluded, "selected_indices": keep}
