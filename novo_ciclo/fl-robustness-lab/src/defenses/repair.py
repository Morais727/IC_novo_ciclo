"""Reparo do modelo global no servidor com poucas amostras limpas (Fase C, experiments/fase2_lab.py, adendo).
Não usa gatilho nem classe-alvo; só o modelo e o server_holdout (100 amostras, 10 por classe).

  fine_prune  Liu et al. (RAID 2018): ativação média (pós-ReLU) de cada neurônio de fc1 nas amostras limpas;
              poda (zera linha de fc1 e coluna de fc2) em ordem crescente, em passos de 5% dos neurônios,
              enquanto a acurácia nas amostras limpas não cair mais que max_drop pp em relação ao modelo sem poda.
  fine_tune   SGD (lr 0,01, momento 0,9 como os clientes, lote 32) por 5 épocas nas amostras limpas.
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def _acc(model, loader, device):
    model.eval()
    ok = n = 0
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu()
        ok += (p == y).sum().item(); n += len(y)
    return 100.0 * ok / max(n, 1)


@torch.no_grad()
def _fc1_activation(model, loader, device):
    model.eval()
    acts = []
    for x, _ in loader:
        h = model.flatten(model.features(x.to(device)))
        acts.append(F.relu(model.fc1(h)).cpu())
    return torch.cat(acts).mean(0)


def fine_prune(model, holdout, device, step=0.05, max_drop=5.0):
    loader = DataLoader(holdout, batch_size=256)
    base = _acc(model, loader, device)
    order = _fc1_activation(model, loader, device).argsort()   # menos ativos primeiro
    n = len(order)
    best, frac = copy.deepcopy(model), 0.0
    k = 0
    while True:
        k += max(1, int(round(step * n)))
        if k >= n:
            break
        trial = copy.deepcopy(model)
        with torch.no_grad():
            idx = order[:k].to(device)
            trial.fc1.weight[idx] = 0
            trial.fc1.bias[idx] = 0
            trial.fc2.weight[:, idx] = 0
        if _acc(trial, loader, device) < base - max_drop:
            break
        best, frac = trial, k / n
    best.pruned_idx = order[:int(round(frac * n))]
    return best, frac


def fine_tune(model, holdout, device, epochs=5, lr=0.01, batch_size=32, keep_pruned=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(holdout, batch_size=batch_size, shuffle=True, generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
            if keep_pruned is not None and len(keep_pruned):
                with torch.no_grad():   # neurônios podados continuam podados
                    idx = keep_pruned.to(device)
                    model.fc1.weight[idx] = 0; model.fc1.bias[idx] = 0; model.fc2.weight[:, idx] = 0
    return model


def fedavg_prune(deltas, weights, ctx, params):
    """FP dentro do treino (adendo 3, fase2_lab.py): FedAvg (ou a agregação em params['base']); no round
    params['prune_round'] (30) o servidor aplica fine_prune ao modelo global com o server_holdout e, daí em diante,
    mantém os neurônios podados em zero depois de cada agregação. Os rounds seguintes recuperam a acurácia com os
    dados dos honestos; os atacantes seguem atacando."""
    from src.defenses.registry import DEFENSE_REGISTRY
    from src.federated.utils import load_vector, flatten
    agg, info = DEFENSE_REGISTRY[params.get("base", "fedavg")](deltas, weights, ctx, params.get("base_params", {}))
    st = ctx["state"]
    if ctx["round_idx"] == params.get("prune_round", 30):
        dev = ctx["device"]
        m = ctx["model_ctor"]().to(dev)
        load_vector(m, (ctx["global_vector"] + agg.cpu()).to(dev))
        pruned, frac = fine_prune(m, ctx["server_holdout"], dev, max_drop=params.get("max_drop", 5.0))
        st["prune_idx"], st["prune_frac"] = pruned.pruned_idx.cpu(), frac
        groups = st.get("tensor_groups")
        if groups is None:
            from src.attacks.recent import tensor_groups
            groups = st["tensor_groups"] = tensor_groups(ctx["model_ctor"]())
        w1, b1, w2 = groups["fc1.weight"], groups["fc1.bias"], groups["fc2.weight"]
        idx = []
        for k in st["prune_idx"].tolist():
            idx += list(range(w1.start + k * 1024, w1.start + (k + 1) * 1024))
            idx.append(b1.start + k)
            idx += [w2.start + c * 128 + k for c in range(10)]
        st["prune_flat"] = torch.tensor(idx, dtype=torch.long)
    if "prune_flat" in st:
        i = st["prune_flat"].to(agg.device)
        agg = agg.clone()
        agg[i] = -ctx["global_vector"].to(agg.device)[i]   # global + agg = 0 nos podados
        info = {**info, "prune_frac": st["prune_frac"]}
    return agg, info
