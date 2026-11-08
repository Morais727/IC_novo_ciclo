"""Reparo do modelo no servidor por perturbação universal (fase 3, experiments/fase3_repair.py), sem gatilho, sem
classe-alvo e sem máscara pequena — a hipótese é que o LP escapa do Neural Cleanse porque o NC procura um carimbo de
poucos pixels, enquanto aqui a perturbação é densa e só limitada em norma L2.

  uap(model, X, Y, eps)       perturbação universal δ (1x3x32x32, espaço normalizado) com ||δ||_2 <= eps que
                              maximiza a perda nas amostras limpas (não dirigida); PGD em L2, `steps` passos.
  ibau_lite(model, hold, eps) desaprendizado no estilo I-BAU (Zeng et al., ICLR 2022), versão simplificada sem
                              hipergradiente implícito: `rounds` vezes, acha δ = uap(modelo atual) e faz uma época de
                              SGD em 0,5·CE(x+δ, y) + 0,5·CE(x, y) com o rótulo verdadeiro.
"""
import torch
import torch.nn.functional as F


def _tensors(holdout):
    xs, ys = zip(*[holdout[i] for i in range(len(holdout))])
    return torch.stack(xs), torch.tensor(ys)


def uap(model, X, Y, eps, steps=20, target=None):
    model.eval()
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    d = torch.zeros(1, *X.shape[1:], device=X.device, requires_grad=True)
    step = 2.5 * eps / steps
    for _ in range(steps):
        out = model(X + d)
        loss = F.cross_entropy(out, Y) if target is None else -F.cross_entropy(out, torch.full_like(Y, target))
        g, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            d += step * g / (g.norm() + 1e-12)
            n = d.norm()
            if n > eps:
                d *= eps / n
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    return d.detach()


def ibau_lite(model, holdout, device, eps=8.0, rounds=10, lr=0.005, batch_size=32, seed=0):
    X, Y = _tensors(holdout)
    X, Y = X.to(device), Y.to(device)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(rounds):
        d = uap(model, X, Y, eps)
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            b = perm[i:i + batch_size].to(device)
            loss = 0.5 * F.cross_entropy(model(X[b] + d), Y[b]) + 0.5 * F.cross_entropy(model(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model
