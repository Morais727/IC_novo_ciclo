"""Resposta cirúrgica por tensor (item 4 do plano de 13/10), ligada ao sinal do item 2.

Por round (desenho FIXADO ANTES de rodar o ciclo fechado):
  1. C_i = delta_i - mediana_coord_j(delta_j);
  2. para cada tensor g: s_g(i) = max_{j != i} cos(C_i[g], C_j[g]); z_g(i) = z robusto de s_g entre clientes;
  3. o par (cliente i, tensor g) é marcado se z_g(i) > z_thr (3, o mesmo limiar do item 2);
  4. agregação por tensor: média ponderada (pesos do FedAvg) só dos clientes NÃO marcados naquele tensor.
     Um cliente marcado só em fc1.weight continua contribuindo com todo o resto do modelo.

Diferente do FLAME/Multi-Krum, que descartam o cliente inteiro. `excluded_indices` lista os clientes com
algum tensor marcado (para comparar com as outras defesas); `kept_fraction` dá, por cliente, a fração dos
parâmetros que entrou na agregação.

Variante layer_guard_sn (15/10, depois da H4 refutada; desenho fixado antes de rodar): por tensor, também o
z robusto da norma de C_i[g]; marca se max(z_sim, 0) + max(z_norm, 0) > z_thr. Ruído em L* derruba a
similaridade mas infla a norma, então a soma fica alta nos dois casos.

LayerGuard final (16/10; desenho fixado antes de rodar): layer_guard_sn + corte de norma POR TENSOR — em cada
tensor, a contribuição de cada cliente aceito é reescalada para no máximo a mediana das normas dos aceitos
naquele tensor (o FLAME corta o modelo inteiro; aqui o corte é por tensor). Marca o que é compartilhado e
construído (LP, ataques de modelo idênticos) e limita o que é treinado em separado e cresce em norma.
`clip_only` (ablação): só o corte por tensor, sem marcar ninguém.
"""
import numpy as np
import torch


def _max_other_cos(x: torch.Tensor) -> torch.Tensor:
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c = u @ u.T
    c.fill_diagonal_(-2.0)
    return c.max(dim=1).values


def layer_guard(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, z_thr = deltas.shape[0], params.get("z_thr", 3.0)
    centered = deltas - deltas.median(dim=0).values
    w = weights.to(deltas.device).float()
    agg = torch.zeros(deltas.shape[1], device=deltas.device, dtype=deltas.dtype)
    kept = np.zeros(n)
    flagged_any = set()
    flags = {}   # Fase 5: tensor -> clientes marcados (só leitura; usado pelo atacante adaptativo simulado)
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    for name, s in groups.items():
        z = rz(_max_other_cos(centered[:, s]).cpu().numpy())
        if params.get("norm"):
            z = np.maximum(z, 0) + np.maximum(rz(centered[:, s].norm(dim=1).cpu().numpy()), 0)
        keep = z <= z_thr
        if params.get("clip_only"):
            keep[:] = True
        if not keep.any():
            keep[:] = True
        x = deltas[:, s]
        if params.get("clip"):
            nrm = x.norm(dim=1)
            cap = nrm[torch.tensor(keep, device=deltas.device)].median()
            x = x * (cap / nrm.clamp_min(1e-12)).clamp(max=1.0).unsqueeze(1)
        flagged_any.update(np.where(~keep)[0].tolist())
        flags[name] = np.where(~keep)[0].tolist()
        m = torch.tensor(keep, device=deltas.device, dtype=w.dtype) * w
        agg[s] = (x * (m / m.sum()).unsqueeze(1)).sum(dim=0)
        kept += keep * (s.stop - s.start)
    kept /= deltas.shape[1]
    excluded = sorted(flagged_any)
    return agg, {"excluded_indices": excluded,
                 "selected_indices": [i for i in range(n) if i not in flagged_any],
                 "kept_fraction": kept.tolist(), "flags": flags}


def layer_guard_sn(deltas, weights, ctx, params):
    return layer_guard(deltas, weights, ctx, {**params, "norm": True})


def layerguard(deltas, weights, ctx, params):
    return layer_guard(deltas, weights, ctx, {**params, "norm": True, "clip": True})


def layerguard_cliponly(deltas, weights, ctx, params):
    return layer_guard(deltas, weights, ctx, {**params, "clip": True, "clip_only": True})


def lg_shadow(deltas, weights, ctx, params):
    """Diagnóstico (16/10): agrega como o FedAvg e só REGISTRA, por round, tensor e cliente, os escores do
    layer_guard_sn (z_sim, z_norm, cosseno máximo e norma brutos). Sai em resultado['diag_log']."""
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    centered = deltas - deltas.median(dim=0).values
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    rec = {}
    for name, s in groups.items():
        cs = _max_other_cos(centered[:, s]).cpu().numpy()
        nr = centered[:, s].norm(dim=1).cpu().numpy()
        x = centered[:, s]
        u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        prev = ctx["state"].get("_shadow_prev", {}).get(name)
        rec[name] = {"cos": np.round(cs, 4).tolist(), "norm": np.round(nr, 6).tolist(),
                     "z_sim": np.round(rz(cs), 3).tolist(), "z_norm": np.round(rz(nr), 3).tolist(),
                     # 16/10 (2ª versão): matriz de cossenos completa e autoconsistência com o round anterior
                     "cmat": np.round((u @ u.T).cpu().numpy(), 3).tolist(),
                     "self": np.round((u * prev).sum(1).cpu().numpy(), 3).tolist() if prev is not None else None}
        ctx["state"].setdefault("_shadow_prev", {})[name] = u
    ctx["state"].setdefault("diag_log", []).append(rec)
    w = weights.to(deltas.device).float()
    return (deltas * (w / w.sum()).unsqueeze(1)).sum(dim=0), {"excluded_indices": []}
