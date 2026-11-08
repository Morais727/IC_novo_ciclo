"""FedSurrogate (arXiv 2605.11122, mai/2026) — reimplementação a partir do texto do artigo; o repositório citado
(github.com/fabacha/FedSurrogate-Backdoor-Defense-in-FL) não existe mais (verificado em 16/10/2026).

Por round:
  1. LCA: por tensor l, d_l = média das distâncias de cosseno entre os deltas de todos os pares de clientes
     (Eq. 1); d~_l = d_l / mediana_l(d_l); L* = os k = 5 tensores de maior d~ (o artigo usa top-k na prática).
  2. HDBSCAN sobre a matriz de distâncias de cosseno de v_i = delta_i[L*]; o maior cluster é o conjunto
     confiável T_c, o resto (inclusive ruído) é suspeito S.
  3. Filtro de alinhamento nos tensores fixos L_r (CNN: fc1, fc2; em outras redes, as camadas lineares): s^_j = cos(w_j - w*, g*), com w* e g*
     médias ponderadas pelo nº de amostras de TODOS os clientes (Eq. 3), normalizado min-max no round;
     média corrida por cliente (Eq. 4). Rebaixa de T_c quem passa q3 + 1,5 IQR (calculado em T_c);
     resgata de S quem tem s <= eps = min(zeta, mediana_S(s)) (Eq. 5), zeta = 0,4.
  4. Substituição: cada cliente marcado F = S \\ R tem L* trocado pelo do doador confiável mais próximo
     (distância do passo 2) (Eq. 6).
  5. Agregação: média com pesos 1 (T_c), 0,7 (resgatados), 0,3 (substitutos) (Eqs. 7-8).

Escolhas onde o artigo é omisso (declaradas antes de rodar): "camada" = tensor (peso e viés separados, como
o código do LP); HDBSCAN(min_cluster_size = n//2 + 1, min_samples = 1, allow_single_cluster) como no FLAME
de referência; se não houver cluster, todos são confiáveis; quartis com interpolação linear do numpy;
o sentido dos limiares segue o texto do algoritmo literalmente (rebaixa escore alto, resgata escore baixo),
embora o texto também diga que o escore é "alto para benignos" — contradição registrada.
"""
import hdbscan
import numpy as np
import torch


def _cos_dist(x: torch.Tensor) -> np.ndarray:
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (1 - u @ u.T).clamp_min(0).double().cpu().numpy()


def fedsurrogate(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    st = ctx["state"]
    groups = st.get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = st["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, k = deltas.shape[0], params.get("k", 5)
    iu = np.triu_indices(n, 1)

    # 1. LCA
    d = {g: _cos_dist(deltas[:, s])[iu].mean() for g, s in groups.items()}
    med = np.median(list(d.values()))
    crit = sorted(groups, key=lambda g: -d[g] / med)[:k]
    idx = torch.cat([torch.arange(groups[g].start, groups[g].stop) for g in crit]).to(deltas.device)

    # 2. HDBSCAN no espaço de L*
    D = _cos_dist(deltas[:, idx])
    labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=n // 2 + 1, min_samples=1,
                             allow_single_cluster=True).fit(D).labels_
    if labels.max() < 0:
        trusted = set(range(n))
    else:
        big = np.bincount(labels[labels >= 0]).argmax()
        trusted = {i for i in range(n) if labels[i] == big}

    # 3. alinhamento em L_r, média corrida, IQR e resgate
    lr = params.get("align_layers") or st.get("fs_align")
    if lr is None:   # L_r = os tensores das camadas lineares (classificador): na SmallCNN, fc1 e fc2, como no artigo
        m = ctx["model_ctor"]()
        lr = st["fs_align"] = [f"{n}.{p}" for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)
                               for p, _ in mod.named_parameters(recurse=False) if f"{n}.{p}" in groups]
    ridx = torch.cat([torch.arange(groups[g].start, groups[g].stop) for g in lr]).to(deltas.device)
    gv = ctx["global_vector"].to(deltas.device)[ridx]
    g = deltas[:, ridx]
    wvec = gv + g
    om = weights.to(deltas.device).float()
    om = om / om.sum()
    w_star, g_star = (om.unsqueeze(1) * wvec).sum(0), (om.unsqueeze(1) * g).sum(0)
    dev = wvec - w_star
    sh = (dev @ g_star / (dev.norm(dim=1) * g_star.norm()).clamp_min(1e-12)).cpu().numpy()
    sh = (sh - sh.min()) / max(sh.max() - sh.min(), 1e-12)
    run_s, cnt = st.setdefault("fs_score", np.zeros(n)), st.setdefault("fs_count", np.zeros(n))
    cnt += 1
    run_s[:] = (cnt - 1) / cnt * run_s + sh / cnt
    tl = sorted(trusted)
    if len(tl) >= 4:
        q1, q3 = np.percentile(run_s[tl], [25, 75])
        trusted -= {i for i in tl if run_s[i] > q3 + 1.5 * (q3 - q1)}
    suspects = [i for i in range(n) if i not in trusted]
    rescued = set()
    if suspects:
        eps = min(params.get("zeta", 0.4), float(np.median(run_s[suspects])))
        rescued = {i for i in suspects if run_s[i] <= eps}
    flagged = [i for i in suspects if i not in rescued]

    # 4. substituição de L* pelo doador confiável mais próximo
    pool = sorted(trusted | rescued) or list(range(n))
    out = deltas.clone()
    for f in flagged:
        donor = min(pool, key=lambda j: D[f, j])
        out[f, idx] = deltas[donor, idx]

    # 5. agregação
    lam = torch.tensor([1.0 if i in trusted else 0.7 if i in rescued else 0.3 for i in range(n)],
                       device=deltas.device, dtype=deltas.dtype)
    agg = (out * (lam / lam.sum()).unsqueeze(1)).sum(0)
    kept = np.ones(n)
    for f in flagged:
        kept[f] = 1 - len(idx) / deltas.shape[1]
    return agg, {"excluded_indices": flagged, "selected_indices": sorted(trusted | rescued),
                 "kept_fraction": kept.tolist(), "critical_layers": crit}
