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


def layer_guard_client(deltas, weights, ctx, params):
    """Ablação (04/11): MESMO detector do layer_guard_sn (z_sim + z_norm por tensor, limiar 3),
    mas marca o CLIENTE INTEIRO — se é marcado em QUALQUER tensor, sai de TODA a agregação (como
    FLAME/Multi-Krum descartam o modelo inteiro, em vez do corte por tensor). Como o detector tem
    recall 1,0 contra o LaySelFL, testa se o bloqueio por cliente contém o backdoor (ao custo de
    acurácia, pois a precisão é ~0,4: também derruba os benignos falsos-positivos)."""
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, z_thr = deltas.shape[0], params.get("z_thr", 3.0)
    centered = deltas - deltas.median(dim=0).values
    w = weights.to(deltas.device).float()
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    flagged = set()
    for name, s in groups.items():
        z = np.maximum(rz(_max_other_cos(centered[:, s]).cpu().numpy()), 0.0) \
            + np.maximum(rz(centered[:, s].norm(dim=1).cpu().numpy()), 0.0)
        flagged.update(np.where(z > z_thr)[0].tolist())
    keep = np.array([i not in flagged for i in range(n)], dtype=bool)
    if not keep.any():
        keep[:] = True
    m = torch.tensor(keep, device=deltas.device, dtype=w.dtype) * w
    agg = (deltas * (m / m.sum()).unsqueeze(1)).sum(dim=0)
    excluded = sorted(np.where(~keep)[0].tolist())
    return agg, {"excluded_indices": excluded,
                 "selected_indices": np.where(keep)[0].tolist(),
                 "kept_fraction": keep.astype(float).tolist()}


def layer_guard_corr(deltas, weights, ctx, params):
    """Corroboração entre camadas (04/11) — resposta ao conv-leak do LaySelFL.

    Diagnóstico: o layer_guard_sn marca por tensor; o atacante é marcado nos tensores de GroupNorm/atalho
    (recall 1,0 como cliente), mas o backdoor viaja nos conv, onde ~59% das contribuições maliciosas passam
    abaixo do limiar. Decidir tensor-a-tensor joga fora a evidência cruzada.

    Observação-chave (medida no lg_shadow): o atacante é acusado por ~22 tensores/rodada (mediana; p10=13),
    o benigno por ~1 (p90=6). Logo, exigir que >= T tensores corroborem antes de purgar o cliente de TODAS
    as camadas separa bem (T=8: recall ~0,95, precisão ~0,85, ~1 falso-positivo/rodada — vs precisão ~0,4 do
    flag por-tensor). Mecanismo:
      1. z_sim + z_norm por tensor (idêntico ao layer_guard_sn), limiar z_thr;
      2. count_i = nº de tensores que acusam o cliente i; confirmado se count_i >= T (confirm_tensors);
      3. agregação por tensor: remove quem é (marcado NAQUELE tensor) OU (confirmado) — o confirmado sai de
         todas as camadas, fechando o vazamento nos conv; o não-confirmado segue o corte cirúrgico por tensor
         (preserva acurácia, não purga benigno marcado só em 1–2 tensores por acaso).
    `excluded_indices` = confirmados (o veredito de cliente do mecanismo)."""
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, z_thr = deltas.shape[0], params.get("z_thr", 3.0)
    T = params.get("confirm_tensors", 8)
    centered = deltas - deltas.median(dim=0).values
    w = weights.to(deltas.device).float()
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    # 1ª passada: flag por tensor + contagem por cliente
    flag = {}
    count = np.zeros(n)
    for name, s in groups.items():
        z = np.maximum(rz(_max_other_cos(centered[:, s]).cpu().numpy()), 0.0) \
            + np.maximum(rz(centered[:, s].norm(dim=1).cpu().numpy()), 0.0)
        f = z > z_thr
        flag[name] = f
        count += f
    confirmed = count >= T
    # 2ª passada: agrega removendo (marcado no tensor) OU (confirmado)
    agg = torch.zeros(deltas.shape[1], device=deltas.device, dtype=deltas.dtype)
    kept = np.zeros(n)
    for name, s in groups.items():
        keep = (~flag[name]) & (~confirmed)
        if not keep.any():                       # fallback: ao menos os não-confirmados; se nenhum, todos
            keep = ~confirmed if (~confirmed).any() else np.ones(n, dtype=bool)
        m = torch.tensor(keep, device=deltas.device, dtype=w.dtype) * w
        agg[s] = (deltas[:, s] * (m / m.sum()).unsqueeze(1)).sum(dim=0)
        kept += keep * (s.stop - s.start)
    kept /= deltas.shape[1]
    excluded = sorted(np.where(confirmed)[0].tolist())
    return agg, {"excluded_indices": excluded,
                 "selected_indices": [i for i in range(n) if not confirmed[i]],
                 "kept_fraction": kept.tolist(),
                 "confirm_count": count.tolist()}


def layer_guard_corr_soft(deltas, weights, ctx, params):
    """Variante SOFT da corroboração (04/11) — conter sem descartar os dados do cliente.

    Mesma detecção da layer_guard_corr (z_sim + z_norm por tensor, limiar z_thr; cliente CONFIRMADO se >= T
    tensores o acusam). A diferença: em vez de REMOVER o confirmado da agregação, SUBSTITUI a contribuição dele
    — e de qualquer par (cliente, tensor) marcado — pela mediana coordenada dos benignos aceitos NAQUELE tensor.
    Todos os clientes mantêm o peso do FedAvg, então a 'massa' da agregação é preservada (não encolhe o nº efetivo
    de votos como a purga por cliente), buscando a contenção da corr com acurácia melhor. `excluded_indices` =
    confirmados (o veredito de cliente, para a métrica de detecção)."""
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, z_thr = deltas.shape[0], params.get("z_thr", 3.0)
    T = params.get("confirm_tensors", 8)
    centered = deltas - deltas.median(dim=0).values
    w = weights.to(deltas.device).float()
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    flag, count = {}, np.zeros(n)
    for name, s in groups.items():
        z = np.maximum(rz(_max_other_cos(centered[:, s]).cpu().numpy()), 0.0) \
            + np.maximum(rz(centered[:, s].norm(dim=1).cpu().numpy()), 0.0)
        flag[name] = z > z_thr
        count += flag[name]
    confirmed = count >= T
    agg = torch.zeros(deltas.shape[1], device=deltas.device, dtype=deltas.dtype)
    mw = (w / w.sum()).unsqueeze(1)
    for name, s in groups.items():
        keep = (~flag[name]) & (~confirmed)                      # benignos aceitos nesse tensor
        if not keep.any():
            keep = ~confirmed if (~confirmed).any() else np.ones(n, dtype=bool)
        kmask = torch.tensor(keep, device=deltas.device)
        med = deltas[:, s][kmask].median(dim=0).values            # consenso benigno no tensor
        x = deltas[:, s].clone()
        x[torch.tensor(~keep, device=deltas.device)] = med        # substitui removidos pelo consenso (mantém o peso)
        agg[s] = (x * mw).sum(dim=0)
    excluded = sorted(np.where(confirmed)[0].tolist())
    return agg, {"excluded_indices": excluded,
                 "selected_indices": [i for i in range(n) if not confirmed[i]],
                 "kept_fraction": [1.0] * n,                       # massa preservada: todos contribuem
                 "confirm_count": count.tolist()}


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


# ----------------------------------------------------------------------------------------------------------------
# RESPOSTA AO ADVERSÁRIO ADAPTATIVO (15/11/2026; desenho FIXADO ANTES DE RODAR — pedido do usuário, prioridade 1).
# Diagnóstico (Fase 8G, 4 sementes): o atacante ciente da corroboração mantém a contagem < T em todo round
# (concentra o veneno em m tensores e dessincroniza com ruído). Testes offline com o diag_log dos portões s42–s45:
# memória temporal (EWMA, e média acumulada por cliente) NÃO separa — 2 benignos não-IID ficam sempre com contagem
# 7–8, acima da média dos atacantes (4–10). Conclusão: a DETECÇÃO por contagem não basta; a resposta vem da
# AGREGAÇÃO: o que o atacante consegue fazer passar abaixo de T precisa ter influência limitada.
# Duas variantes sobre a corr-soft (a recomendada), mesma detecção (z_thr = 3, T = 8):
#   *_tm   : após a substituição por consenso, média APARADA por coordenada em cada tensor (corta beta=0,2 de cada
#            lado: 4 de 20). Um atacante que passa abaixo de T só move o agregado dentro da faixa dos honestos.
#   *_clip : após a substituição, corta a NORMA de cada contribuição, por tensor, na mediana das normas dos aceitos
#            naquele tensor (o corte por tensor do LayerGuard, 16/10), depois média ponderada.
# Avaliação (fase16): contra o adaptativo CIENTE DA AGREGAÇÃO (o atacante simula a variante e escolhe a config que
# mais empurra o agregado na direção do backdoor), contra o LaySelFL padrão e sem ataque (custo de acurácia).
def _corr_detect(deltas, ctx, params):
    groups = ctx["state"].get("tensor_groups")
    if groups is None:
        from src.attacks.recent import tensor_groups
        groups = ctx["state"]["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
    n, z_thr = deltas.shape[0], params.get("z_thr", 3.0)
    T = params.get("confirm_tensors", 8)
    centered = deltas - deltas.median(dim=0).values
    rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    flag, count = {}, np.zeros(n)
    for name, s in groups.items():
        z = np.maximum(rz(_max_other_cos(centered[:, s]).cpu().numpy()), 0.0) \
            + np.maximum(rz(centered[:, s].norm(dim=1).cpu().numpy()), 0.0)
        flag[name] = z > z_thr
        count += flag[name]
    return groups, flag, count, count >= T


def _corr_soft_robust(deltas, weights, ctx, params, mode):
    groups, flag, count, confirmed = _corr_detect(deltas, ctx, params)
    n = deltas.shape[0]
    w = weights.to(deltas.device).float()
    mw = (w / w.sum()).unsqueeze(1)
    beta = params.get("trim_beta", 0.2)
    k = int(np.floor(beta * n))
    agg = torch.zeros(deltas.shape[1], device=deltas.device, dtype=deltas.dtype)
    for name, s in groups.items():
        keep = (~flag[name]) & (~confirmed)
        if not keep.any():
            keep = ~confirmed if (~confirmed).any() else np.ones(n, dtype=bool)
        kmask = torch.tensor(keep, device=deltas.device)
        x = deltas[:, s].clone()
        x[~kmask] = x[kmask].median(dim=0).values                 # substituição por consenso (corr-soft)
        if mode == "tm":
            xs = x.sort(dim=0).values
            agg[s] = xs[k:n - k].mean(dim=0) if n - 2 * k > 0 else xs.median(dim=0).values
        elif mode == "clip":
            nrm = x.norm(dim=1)
            cap = nrm[kmask].median()
            x = x * (cap / nrm.clamp_min(1e-12)).clamp(max=1.0).unsqueeze(1)
            agg[s] = (x * mw).sum(dim=0)
        else:
            raise ValueError(mode)
    return agg, {"excluded_indices": sorted(np.where(confirmed)[0].tolist()),
                 "selected_indices": [i for i in range(n) if not confirmed[i]],
                 "kept_fraction": [1.0] * n, "confirm_count": count.tolist()}


def layer_guard_corr_soft_tm(deltas, weights, ctx, params):
    return _corr_soft_robust(deltas, weights, ctx, params, "tm")


def layer_guard_corr_soft_clip(deltas, weights, ctx, params):
    return _corr_soft_robust(deltas, weights, ctx, params, "clip")
