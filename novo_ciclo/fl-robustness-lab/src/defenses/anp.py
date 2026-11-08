"""Adversarial Neuron Pruning (Wu & Wang, NeurIPS 2021), versão enxuta para a SmallCNN (fase 3, adendo A1 em
experiments/fase3_repair.py). Neurônios = canais das 3 convoluções e unidades de fc1 (288). Cada neurônio k tem
máscara m_k ∈ [0,1] e perturbação δ_k ∈ [-eps, eps] que multiplicam a sua saída antes da ReLU (simplificação:
o ANP original perturba peso e viés separadamente). Otimiza  α·CE(m) + (1-α)·max_δ CE(m+δ)  nas amostras limpas
(um passo de sinal em δ por iteração, SGD em m) e poda (zera) os neurônios com m < thr. Sem gatilho, sem alvo."""
import copy
import torch
import torch.nn.functional as F


def _layers(model):
    convs = [l for l in model.features if isinstance(l, torch.nn.Conv2d)]
    return convs + [model.fc1]


def _forward(model, x, scales):
    convs = _layers(model)[:-1]
    i = 0
    for l in model.features:
        x = l(x)
        if isinstance(l, torch.nn.Conv2d):
            x = x * scales[i].view(1, -1, 1, 1); i += 1
    x = model.fc1(model.flatten(x)) * scales[-1].view(1, -1)
    return model.fc2(F.relu(x))


def anp(model, holdout, device, eps=0.4, alpha=0.2, iters=500, thr=0.2, lr=0.2, batch_size=32, seed=0):
    xs, ys = zip(*[holdout[i] for i in range(len(holdout))])
    X, Y = torch.stack(xs).to(device), torch.tensor(ys).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sizes = [l.weight.shape[0] for l in _layers(model)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = [torch.ones(s, device=device, requires_grad=True) for s in sizes]
    d = [((torch.rand(s, generator=g) * 2 - 1) * eps).to(device).requires_grad_(True) for s in sizes]
    opt = torch.optim.SGD(m, lr=lr, momentum=0.9)
    for _ in range(iters):
        b = torch.randint(0, len(X), (batch_size,), generator=g).to(device)
        x, y = X[b], Y[b]
        # passo de maximização em δ
        loss = F.cross_entropy(_forward(model, x, [mi + di for mi, di in zip(m, d)]), y)
        gd = torch.autograd.grad(loss, d)
        with torch.no_grad():
            for di, gi in zip(d, gd):
                di.add_(eps * gi.sign()).clamp_(-eps, eps)
        # passo de minimização em m
        loss = alpha * F.cross_entropy(_forward(model, x, m), y) + \
               (1 - alpha) * F.cross_entropy(_forward(model, x, [mi + di for mi, di in zip(m, d)]), y)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for mi in m:
                mi.clamp_(0, 1)
    out = copy.deepcopy(model)
    for p in out.parameters():
        p.requires_grad_(True)
    for p in model.parameters():
        p.requires_grad_(True)
    pruned = 0
    with torch.no_grad():
        for l, mi in zip(_layers(out), m):
            idx = (mi < thr).nonzero().flatten()
            l.weight[idx] = 0; l.bias[idx] = 0; pruned += len(idx)
    out.anp_pruned = pruned / sum(sizes)
    return out
