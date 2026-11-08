"""A3FL (Zhang et al., NeurIPS 2023; github.com/hfzhang31/A3FL) — gatilho adversarialmente adaptativo.
Reimplementado a partir de fl_utils/attacker.py do repositório oficial. Otimiza um gatilho (patch) por ascensão
de gradiente no SINAL do gradiente, de modo que o modelo global E cópias "readaptadas" (treinadas para desaprender
o gatilho) classifiquem as entradas com gatilho como a classe-alvo — tornando o backdoor robusto à defesa.

Aqui o A3FL é o GERADOR DE GATILHO do LaySelFL (o artigo do LaySelFL testa BadNet e A3FL). O gatilho é persistido
entre rodadas em params["_a3fl"] e refinado uma vez por rodada contra o modelo global atual; publicado via
datasets.set_a3fl_trigger para o envenenamento (BackdoorDataset) e a avaliação (BackdoorEvalDataset) usarem o mesmo.

ADAPTAÇÃO DE ORÇAMENTO (registrada antes de rodar): o artigo usa K=200 épocas internas e reconstrói o modelo
adversarial a cada iteração — inviável no nosso harness (6 atacantes x 121 rodadas de ataque). Mantemos o
MECANISMO fiel, mas com orçamento reduzido e configurável (padrão K=25, modelo adversarial a cada 10 iters,
3 épocas de readaptação), já que o gatilho refina de forma incremental ao longo das rodadas. Valores do artigo
mantidos: gatilho 5x5 em [2:7,2:7], lr 0,01, noise_loss_lambda 0,01, clamp do gatilho em [-2, 2] (espaço normalizado).
"""
import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.federated.utils import load_vector
from src.data import datasets as D


def _adv_model(model, dl, t, m, device, epochs, ce):
    """Cópia do modelo global readaptada para CLASSIFICAR corretamente as entradas com gatilho (desaprender o
    backdoor). Retorna (modelo, peso), peso = alinhamento de cosseno dos gradientes das convs (Eq. do A3FL)."""
    adv = copy.deepcopy(model).to(device); adv.train()
    opt = torch.optim.SGD(adv.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            x = t * m + (1 - m) * x
            opt.zero_grad(); ce(adv(x), y).backward(); opt.step()
    # peso de alinhamento: um batch, gradientes frescos de ambos
    x, y = next(iter(dl)); x, y = x.to(device), y.to(device); xt = t * m + (1 - m) * x
    model.zero_grad(); ce(model(xt), y).backward()
    adv.zero_grad(); ce(adv(xt), y).backward()
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
    ap, mp = dict(adv.named_parameters()), dict(model.named_parameters())
    s, n = 0.0, 0
    for name in ap:
        if "conv" in name and ap[name].grad is not None and mp[name].grad is not None:
            n += 1; s = s + cos(ap[name].grad.reshape(-1), mp[name].grad.reshape(-1))
    model.zero_grad()
    return adv, (s / n if n else torch.tensor(1.0, device=device))


def refine_a3fl_trigger(model_ctor, global_vector, attacker_data, device, params, target):
    st = params.setdefault("_a3fl", {})
    model = model_ctor().to(device); load_vector(model, global_vector.to(device)); model.eval()

    C, H, W = attacker_data[0][0][0].shape
    sz = params.get("a3fl_trigger_size", 5)
    if "t" not in st:
        t = torch.ones((1, C, H, W), device=device) * 0.5
        m = torch.zeros_like(t); m[:, :, 2:2 + sz, 2:2 + sz] = 1
        st["t"], st["m"] = t.cpu(), m.cpu()
    t, m = st["t"].to(device), st["m"].to(device)

    bs = params.get("_hp", {}).get("batch_size", 64)
    dl = DataLoader(ConcatDataset(list(attacker_data)), batch_size=bs, shuffle=True, drop_last=False)
    ce = torch.nn.CrossEntropyLoss()
    alpha = params.get("a3fl_lr", 0.01)
    K = params.get("a3fl_K", 25)
    adv_K = params.get("a3fl_adv_K", 10)
    adv_epochs = params.get("a3fl_adv_epochs", 3)
    lam = params.get("a3fl_noise_lambda", 0.01)

    advs, it = [], 0
    for _ in range(K):
        if it % adv_K == 0 and it != 0:
            advs = [_adv_model(model, dl, t, m, device, adv_epochs, ce)]
        for x, _y in dl:
            x = x.to(device)
            y = torch.full((x.size(0),), target, device=device, dtype=torch.long)
            t.requires_grad_(True)
            xin = t * m + (1 - m) * x
            loss = ce(model(xin), y)
            for adv, w in advs:
                loss = loss + lam * w * ce(adv(xin), y)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                t = (t - alpha * t.grad.sign()).clamp(-2, 2)
            t = t.detach()
            it += 1

    st["t"], st["m"] = t.detach().cpu(), m.detach().cpu()
    D.set_a3fl_trigger(st["t"][0], st["m"][0])          # [C,H,W] para _add_trigger (por imagem)
