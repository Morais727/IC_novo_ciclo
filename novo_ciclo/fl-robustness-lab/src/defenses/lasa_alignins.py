"""Fase 5 (21/10/2026): duas defesas recentes do mesmo grupo (Xu, Zhang e Hu, Univ. de Nevada), reimplementadas a
partir do código oficial para servir de linha de base. Nenhuma das duas foi avaliada contra o LP pelos autores.

LASA (WACV 2025, arXiv 2409.01435; github.com/JiiahaoXU/LASA, algorithms/defense/lasa.py) — a defesa publicada mais
parecida com a nossa proposta: decide por camada e por cliente.
  1. corta a norma de cada atualização (modelo inteiro) na mediana das normas;
  2. esparsifica: mantém os 70% maiores |valores| entre todos os pesos de conv/linear (limiar global; vieses e
     normalizações não são mascarados);
  3. por tensor: MZ = |x − mediana| / desvio padrão da norma da atualização esparsa e da "pureza de sinal"
     0,5·(1 + Σsinal/Σ|sinal|·(1 − esparsidade)); o cliente fica no tensor se as duas MZ < 1 (lambda_n = lambda_s);
  4. cada tensor = média simples das atualizações (cortadas e esparsificadas, como no código oficial, em que a
     máscara é aplicada no mesmo objeto) dos clientes que ficaram.
AlignIns (CVPR 2025, arXiv 2503.07978; github.com/JiiahaoXU/AlignIns, src/aggregation.py::agg_alignins) — por cliente.
  TDA = cos(atualização, modelo global); MPSA = fração dos 30% maiores |valores| cujo sinal concorda com o sinal
  majoritário; mantém os clientes com as duas MZ < 1; corta todos na mediana das normas dos mantidos; média ponderada
  pelos dados dos mantidos.
"""
import numpy as np
import torch


def _shapes(ctx):
    sh = ctx["state"].get("_shapes")
    if sh is None:
        sh, pos = [], 0
        for name, p in ctx["model_ctor"]().named_parameters():
            sh.append((name, slice(pos, pos + p.numel()), p.dim()))
            pos += p.numel()
        ctx["state"]["_shapes"] = sh
    return sh


def _mz(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return np.abs(v - np.median(v)) / (sd if sd > 0 else 1e-12)


def lasa(deltas, weights, ctx, params):
    sp, lam_n, lam_s = params.get("sparsity", 0.3), params.get("lambda_n", 1.0), params.get("lambda_s", 1.0)
    shapes = _shapes(ctx)
    n = deltas.shape[0]
    nrm = deltas.norm(dim=1, keepdim=True)
    x = deltas / nrm.clamp_min(1e-12) * nrm.clamp(max=nrm.median().item())
    wmask = torch.zeros(deltas.shape[1], dtype=torch.bool)
    for _, s, dim in shapes:
        if dim in (2, 4):
            wmask[s] = True
    x = x.clone()
    for i in range(n):
        a = x[i, wmask].abs()
        k = int(a.numel() * (1 - sp))
        thr = torch.topk(a, k, sorted=True).values[-1]
        xi = x[i, wmask]
        x[i, wmask] = xi * (xi.abs() > thr)
    agg = torch.zeros(deltas.shape[1], dtype=deltas.dtype)
    flagged, kept = set(), np.zeros(n)
    for name, s, _ in shapes:
        g = x[:, s]
        mzn = _mz(g.norm(dim=1).numpy())
        sg = torch.sign(g)
        pur = 0.5 * (1 + sg.sum(1) / sg.abs().sum(1).clamp_min(1e-12) * (1 - sp))
        mzs = _mz(pur.numpy())
        keep = (mzn < lam_n) & (mzs < lam_s)
        if not keep.any():
            keep[:] = True
        flagged.update(np.where(~keep)[0].tolist())
        agg[s] = g[torch.tensor(keep)].mean(dim=0)
        kept += keep * (s.stop - s.start)
    kept /= deltas.shape[1]
    return agg, {"excluded_indices": sorted(flagged), "kept_fraction": kept.tolist()}


def alignins(deltas, weights, ctx, params):
    sp, lam_s, lam_c = params.get("sparsity", 0.3), params.get("lambda_s", 1.0), params.get("lambda_c", 1.0)
    gv = ctx["global_vector"].to(deltas.dtype)
    major = torch.sign(torch.sign(deltas).sum(0))
    k = int(deltas.shape[1] * sp)
    mpsa, tda = [], []
    for u in deltas:
        idx = torch.topk(u.abs(), k).indices
        mpsa.append((torch.sign(u[idx]) == major[idx]).float().mean().item())
        tda.append(torch.nn.functional.cosine_similarity(u, gv, dim=0, eps=1e-6).item())
    keep = (_mz(mpsa) < lam_s) & (_mz(tda) < lam_c)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return torch.zeros_like(deltas[0]), {"excluded_indices": list(range(len(deltas)))}
    nrm = deltas.norm(dim=1, keepdim=True)
    cap = nrm[idx].median().item()
    x = deltas / nrm.clamp_min(1e-12) * nrm.clamp(max=cap)
    w = weights.float()[idx]
    return (x[idx] * (w / w.sum()).unsqueeze(1)).sum(0), {"excluded_indices": np.where(~keep)[0].tolist()}
