"""Inversão de gatilho no servidor (Neural Cleanse, Wang et al., S&P 2019, versão enxuta) com o server_holdout
(100 amostras limpas), sem saber gatilho nem classe-alvo — adendo 4 de experiments/fase2_lab.py.

  invert(model, holdout, c)  otimiza máscara m (1x32x32, sigmoide) e padrão p (3x32x32, 3*tanh, espaço
                             normalizado) para levar x' = (1-m) x + m p à classe c nas amostras de outras classes:
                             CE + lam * |m|_1, Adam lr 0,1, `steps` passos (lote = todas as amostras).
  detect                     norma L1 da máscara de cada classe; índice de anomalia = (mediana - L1_c) /
                             (1,4826 MAD); a classe-alvo é a de menor L1 se o índice passar `ai_thr` (2, como no NC).
  nc_prune                   se houver alvo: poda neurônios de fc1 em ordem decrescente de (ativação com o gatilho
                             reconstruído - ativação limpa), em passos de 2%, até a taxa de sucesso do gatilho
                             reconstruído nas amostras cair a <= 10% ou a acurácia limpa nelas cair > max_drop pp.
                             Sem alvo: modelo intacto.
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _tensors(holdout):
    xs, ys = zip(*[holdout[i] for i in range(len(holdout))])
    return torch.stack(xs), torch.tensor(ys)


def invert(model, X, Y, c, device, steps=300, lam=1e-2, seed=0):
    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = X[Y != c].to(device)
    tgt = torch.full((len(x),), c, device=device)
    m_raw = torch.full((1, 1, 32, 32), -2.0, device=device, requires_grad=True)
    p_raw = (torch.randn(1, 3, 32, 32, device=device) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([m_raw, p_raw], lr=0.1)
    for _ in range(steps):
        m, p = torch.sigmoid(m_raw), 3 * torch.tanh(p_raw)
        xa = (1 - m) * x + m * p
        loss = F.cross_entropy(model(xa), tgt) + lam * m.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    for q in model.parameters():
        q.requires_grad_(True)
    with torch.no_grad():
        m, p = torch.sigmoid(m_raw), 3 * torch.tanh(p_raw)
        succ = (model((1 - m) * x + m * p).argmax(1) == c).float().mean().item()
    return m.detach(), p.detach(), m.abs().sum().item(), succ


def detect(model, holdout, device, ai_thr=2.0, **kw):
    X, Y = _tensors(holdout)
    res = [invert(model, X, Y, c, device, **kw) for c in range(10)]
    l1 = torch.tensor([r[2] for r in res])
    med = l1.median()
    mad = 1.4826 * (l1 - med).abs().median() + 1e-9
    ai = ((med - l1) / mad)
    c = int(l1.argmin())
    return {"target": c if ai[c] > ai_thr else None, "ai": ai.tolist(), "l1": l1.tolist(),
            "succ": [r[3] for r in res], "mask": res[c][0], "pattern": res[c][1], "cand": c}


@torch.no_grad()
def _stats(model, X, Y, m, p, c, device, idx=None):
    model.eval()
    x, y = X.to(device), Y.to(device)
    acc = (model(x).argmax(1) == y).float().mean().item() * 100
    xo = x[y != c]
    succ = (model((1 - m) * xo + m * p).argmax(1) == c).float().mean().item() * 100
    return acc, succ


def nc_prune(model, holdout, device, det=None, step=0.02, max_drop=5.0, stop_succ=10.0, **kw):
    det = det or detect(model, holdout, device, **kw)
    if det["target"] is None:
        return model, {"target": None, "pruned": 0.0, "ai": det["ai"]}
    c, m, p = det["target"], det["mask"], det["pattern"]
    X, Y = _tensors(holdout)
    x = X.to(device)
    with torch.no_grad():
        h = lambda z: F.relu(model.fc1(model.flatten(model.features(z))))
        diff = (h((1 - m) * x + m * p) - h(x)).mean(0)
    order = diff.argsort(descending=True).cpu()
    base_acc, _ = _stats(model, X, Y, m, p, c, device)
    best, frac, k = copy.deepcopy(model), 0.0, 0
    n = len(order)
    while k < n:
        k += max(1, int(round(step * n)))
        trial = copy.deepcopy(model)
        with torch.no_grad():
            i = order[:k].to(device)
            trial.fc1.weight[i] = 0; trial.fc1.bias[i] = 0; trial.fc2.weight[:, i] = 0
        acc, succ = _stats(trial, X, Y, m, p, c, device)
        if acc < base_acc - max_drop:
            break
        best, frac = trial, k / n
        if succ <= stop_succ:
            break
    best.pruned_idx = order[:int(round(frac * n))]
    return best, {"target": c, "pruned": frac, "ai": det["ai"]}


def nc_unlearn(model, holdout, device, det, lr=0.001, epochs=5, frac=0.5, batch_size=32, seed=0):
    """Desaprendizado (NC, seção de mitigação): ajuste fino nas amostras limpas, com o gatilho reconstruído
    aplicado a `frac` delas e o rótulo VERDADEIRO. Sem alvo detectado: modelo intacto."""
    if det["target"] is None:
        return model
    m, p = det["mask"], det["pattern"]
    X, Y = _tensors(holdout)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            b = perm[i:i + batch_size]
            x, y = X[b].to(device), Y[b].to(device)
            sel = (torch.rand(len(b), generator=g) < frac).to(device).view(-1, 1, 1, 1).float()
            x = sel * ((1 - m) * x + m * p) + (1 - sel) * x
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    return model
