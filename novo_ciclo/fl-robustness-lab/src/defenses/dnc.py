"""Busca diferencial de gatilho por cliente (fase 3, adendo A3 em experiments/fase3_repair.py).
Para o modelo do cliente f_i = global + Δ_i e o modelo global f_g, e cada classe c, otimiza máscara m e padrão p
(como no NC) para MAXIMIZAR o ganho do efeito do gatilho que o cliente acrescentou:
  G_i(c) = média_x [ (log p_i(c|x') - log p_i(c|x)) - (log p_g(c|x') - log p_g(c|x)) ],   x' = (1-m)x + m·p
  objetivo = -G_i(c) + lam·|m|_1      (x = amostras do holdout que não são da classe c)
O termo com x limpo desconta o cliente honesto que só puxa tudo para a classe dele; o termo do global desconta o
que o modelo já tinha. Escore do cliente = max_c G_i(c) ao fim de `steps` passos.
budget (adendo A4): troca a penalidade fraca por um teto no tamanho da máscara (penalidade relu(|m| - budget)),
porque sem teto a busca usa ~500 pixels e acha diferenças quaisquer entre dois modelos."""
import torch
import torch.nn.functional as F


def gain(fi, fg, X, Y, c, device, steps=100, lam=1e-2, seed=0, budget=None):
    torch.manual_seed(seed)
    x = X[Y != c].to(device)
    with torch.no_grad():
        base = F.log_softmax(fi(x), 1)[:, c] - F.log_softmax(fg(x), 1)[:, c]
    m_raw = torch.full((1, 1, 32, 32), -2.0, device=device, requires_grad=True)
    p_raw = (torch.randn(1, 3, 32, 32, device=device) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([m_raw, p_raw], lr=0.1)
    for _ in range(steps):
        m, p = torch.sigmoid(m_raw), 3 * torch.tanh(p_raw)
        xa = (1 - m) * x + m * p
        g = (F.log_softmax(fi(xa), 1)[:, c] - F.log_softmax(fg(xa), 1)[:, c] - base).mean()
        loss = -g + (lam * m.abs().sum() if budget is None else F.relu(m.sum() - budget))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m, p = torch.sigmoid(m_raw), 3 * torch.tanh(p_raw)
        xa = (1 - m) * x + m * p
        g = (F.log_softmax(fi(xa), 1)[:, c] - F.log_softmax(fg(xa), 1)[:, c] - base).mean().item()
    return g, m.sum().item()


def gain_funnel(fi, fg, X, Y, c, device, steps=(50, 25, 25), keep=(64, 16), lam=1e-2, seed=0):
    """Adendo A5: busca afunilada. Fase 1 livre (penalidade lam); depois fixa o suporte nos `keep[0]` pixels de maior
    máscara e otimiza; depois nos `keep[1]`. Devolve G com máscara binária no suporte final (tamanho = keep[-1])."""
    torch.manual_seed(seed)
    x = X[Y != c].to(device)
    with torch.no_grad():
        base = F.log_softmax(fi(x), 1)[:, c] - F.log_softmax(fg(x), 1)[:, c]
    m_raw = torch.full((1, 1, 32, 32), -2.0, device=device, requires_grad=True)
    p_raw = (torch.randn(1, 3, 32, 32, device=device) * 0.1).requires_grad_(True)
    support = torch.ones(1, 1, 32, 32, device=device)
    G = lambda m, p: (F.log_softmax(fi((1 - m) * x + m * p), 1)[:, c] - F.log_softmax(fg((1 - m) * x + m * p), 1)[:, c] - base).mean()
    for ph, n in enumerate(steps):
        opt = torch.optim.Adam([m_raw, p_raw], lr=0.1)
        for _ in range(n):
            m = torch.sigmoid(m_raw) * support
            loss = -G(m, 3 * torch.tanh(p_raw)) + (lam * m.sum() if ph == 0 else 0)
            opt.zero_grad(); loss.backward(); opt.step()
        if ph < len(keep):
            with torch.no_grad():
                sc = (torch.sigmoid(m_raw) * support).flatten()
                support = torch.zeros_like(sc); support[sc.topk(keep[ph]).indices] = 1; support = support.view(1, 1, 32, 32)
    with torch.no_grad():
        return G(support, 3 * torch.tanh(p_raw)).item(), support


@torch.no_grad()
def patch_scan(fi, fg, X, Y, device, sizes=(3, 4, 5), stride=1, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Adendo A6: varredura de quadrados de cor sólida (8 cores: branco/preto por canal, valores válidos de imagem),
    em todas as posições (passo `stride`) e tamanhos. Para cada classe c, G(c, patch) como em gain(), sobre as
    amostras que não são de c. Devolve max_c max_patch G e o argmax. Supõe gatilho em forma de quadrado (BadNets)."""
    hi = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device)
    lo = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device)
    colors = [torch.where(torch.tensor([(k >> b) & 1 for b in range(3)], device=device).bool(), hi, lo) for k in range(8)]
    X, Y = X.to(device), Y.to(device)
    lpi, lpg = F.log_softmax(fi(X), 1), F.log_softmax(fg(X), 1)
    best = (-1e9, None)
    for s in sizes:
        for col in colors:
            for i in range(0, 33 - s, stride):
                xs = []
                for j in range(0, 33 - s, stride):
                    z = X.clone(); z[:, :, i:i + s, j:j + s] = col.view(1, 3, 1, 1); xs.append(z)
                Z = torch.cat(xs)
                ai, ag = F.log_softmax(fi(Z), 1).view(len(xs), len(X), -1), F.log_softmax(fg(Z), 1).view(len(xs), len(X), -1)
                eff = (ai - lpi) - (ag - lpg)                        # [posições, amostras, classes]
                mask = (Y.view(1, -1, 1) != torch.arange(eff.shape[2], device=device).view(1, 1, -1)).float()
                G = (eff * mask).sum(1) / mask.sum(1)                # média só nas amostras de outra classe
                v, idx = G.flatten().max(0)
                if v.item() > best[0]:
                    p, c = divmod(idx.item(), G.shape[1]); best = (v.item(), (s, int(col.sum() > 0), i, p * stride, c))
    return best


@torch.no_grad()
def patch_scan_multi(fis, fg, X, Y, device, sizes=(3, 4, 5), stride=2, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Mesma varredura de patch_scan, para vários clientes de uma vez: o efeito no global é calculado uma vez por
    quadrado. Devolve, por cliente, (G máximo, (tamanho, cor, linha, coluna, classe)); cor = índice 0..7 (bits RGB)."""
    hi = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device)
    lo = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device)
    colors = [torch.where(torch.tensor([(k >> b) & 1 for b in range(3)], device=device).bool(), hi, lo) for k in range(8)]
    X, Y = X.to(device), Y.to(device)
    lpg = F.log_softmax(fg(X), 1); lpi = [F.log_softmax(f(X), 1) for f in fis]
    mask = (Y.view(1, -1, 1) != torch.arange(lpg.shape[1], device=device).view(1, 1, -1)).float()
    best = [(-1e9, None)] * len(fis)
    for s in sizes:
        for ci, col in enumerate(colors):
            for i in range(0, 33 - s, stride):
                js = list(range(0, 33 - s, stride))
                Z = X.unsqueeze(0).repeat(len(js), 1, 1, 1, 1)
                for k, j in enumerate(js):
                    Z[k, :, :, i:i + s, j:j + s] = col.view(1, 3, 1, 1)
                Z = Z.flatten(0, 1)
                eg = F.log_softmax(fg(Z), 1).view(len(js), len(X), -1) - lpg
                for n, f in enumerate(fis):
                    eff = F.log_softmax(f(Z), 1).view(len(js), len(X), -1) - lpi[n] - eg
                    G = (eff * mask).sum(1) / mask.sum(1)
                    v, idx = G.flatten().max(0)
                    if v.item() > best[n][0]:
                        p, c = divmod(idx.item(), G.shape[1]); best[n] = (v.item(), (s, ci, i, js[p], c))
    return best


@torch.no_grad()
def patch_effects(fis, fg, X, Y, device, sizes=(3, 4, 5), stride=2, offset=0, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Adendo A7: devolve o tensor completo G[cliente, quadrado, classe] (efeito descontado de cada quadrado de cor
    sólida, como em patch_scan) e a lista de quadrados (tamanho, cor 0..7, linha, coluna)."""
    hi = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device)
    lo = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device)
    colors = [torch.where(torch.tensor([(k >> b) & 1 for b in range(3)], device=device).bool(), hi, lo) for k in range(8)]
    X, Y = X.to(device), Y.to(device)
    lpg = F.log_softmax(fg(X), 1); lpi = [F.log_softmax(f(X), 1) for f in fis]
    mask = (Y.view(1, -1, 1) != torch.arange(lpg.shape[1], device=device).view(1, 1, -1)).float()
    out, patches = [], []
    for s in sizes:
        for ci, col in enumerate(colors):
            for i in range(offset, 33 - s, stride):
                js = list(range(offset, 33 - s, stride))
                Z = X.unsqueeze(0).repeat(len(js), 1, 1, 1, 1)
                for k, j in enumerate(js):
                    Z[k, :, :, i:i + s, j:j + s] = col.view(1, 3, 1, 1)
                Z = Z.flatten(0, 1)
                eg = F.log_softmax(fg(Z), 1).view(len(js), len(X), -1) - lpg
                Gs = [(((F.log_softmax(f(Z), 1).view(len(js), len(X), -1) - lpi[n] - eg) * mask).sum(1) / mask.sum(1)) for n, f in enumerate(fis)]
                out.append(torch.stack(Gs).cpu()); patches += [(s, ci, i, j) for j in js]
    return torch.cat(out, 1), patches
