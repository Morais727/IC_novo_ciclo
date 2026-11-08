"""Defesa da fase 3 (adendo A9, experiments/fase3_lab.py): varredura de gatilho compartilhado no modelo de cada
cliente + exclusão + desaprendizado. Não usa gatilho nem alvo conhecidos; usa as amostras limpas do servidor.

A cada `every` rounds a partir de `start`:
  1. f_g = modelo global; f_i = global + Δ_i para cada cliente ainda não excluído;
  2. G[i, quadrado, classe] = efeito descontado de cada quadrado de cor sólida (src/defenses/dnc.py::patch_effects:
     [log p_i(c|x+q) − log p_i(c|x)] − [o mesmo em f_g], média nas amostras que não são de c);
  3. z robusto entre clientes por (quadrado, classe); escore = 2º maior z (gatilho compartilhado por >= 2 clientes);
     se o maior escore passar de `thr` (3), os clientes com z > thr nesse candidato são excluídos para sempre e o
     candidato entra na lista de gatilhos;
  4. agregação = FedAvg (ou params['base']) só dos não excluídos;
  5. se houve gatilho novo neste round: desaprendizado no modelo agregado (como o NC-U: metade das amostras com o
     quadrado, rótulo verdadeiro, SGD lr 0,001, `unlearn_epochs` épocas) para todos os gatilhos da lista.
mode='top1' (adendo A10): em vez do passo 3, exclui só o 1º colocado por max z (escore A7) se passar de thr_top.
Nunca exclui mais da metade dos clientes num round.
Limitação: precisa de >= 2 atacantes com o mesmo gatilho e supõe gatilho que um quadrado de cor sólida ative.
"""
import torch
import torch.nn.functional as F
from src.defenses.dnc import patch_effects

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _color(ci, device):
    hi = torch.tensor([(1 - m) / s for m, s in zip(MEAN, STD)], device=device)
    lo = torch.tensor([(0 - m) / s for m, s in zip(MEAN, STD)], device=device)
    return torch.where(torch.tensor([(ci >> b) & 1 for b in range(3)], device=device).bool(), hi, lo)


def _apply(x, trig, device):
    s, ci, i, j = trig
    x = x.clone(); x[:, :, i:i + s, j:j + s] = _color(ci, device).view(1, 3, 1, 1)
    return x


def trigger_scan(deltas, weights, ctx, params):
    from src.defenses.registry import DEFENSE_REGISTRY
    from src.federated.utils import load_vector, flatten
    st, r, dev = ctx["state"], ctx["round_idx"], ctx["device"]
    flagged = st.setdefault("flagged", set()); trigs = st.setdefault("triggers", [])
    hold = ctx["server_holdout"]
    xs, ys = zip(*[hold[k] for k in range(len(hold))])
    X, Y = torch.stack(xs).to(dev), torch.tensor(ys).to(dev)
    n_scan = params.get("n_scan", 50)
    new = False
    if r >= params.get("start", 8) and (r - params.get("start", 8)) % params.get("every", 4) == 0:
        live = [i for i in range(deltas.shape[0]) if i not in flagged]
        fg = ctx["model_ctor"]().to(dev); load_vector(fg, ctx["global_vector"].to(dev)); fg.eval()
        fis = []
        for i in live:
            f = ctx["model_ctor"]().to(dev); load_vector(f, (ctx["global_vector"] + deltas[i]).to(dev)); f.eval(); fis.append(f)
        G, P = patch_effects(fis, fg, X[:n_scan], Y[:n_scan], dev, sizes=params.get("sizes", (3, 5)),
                             stride=params.get("stride", 3), offset=params.get("offset", 1))
        med = G.median(0).values; mad = 1.4826 * (G - med).abs().median(0).values + 0.05
        Z = (G - med) / mad
        log = st.setdefault("scan_log", [])
        hit = []
        if params.get("mode", "shared") == "top1":   # adendo A10: exclui o 1º colocado por A7 se passar de thr_top
            a7 = Z.flatten(1).max(1).values
            k = int(a7.argmax()); best = float(a7[k]); q, c = divmod(int(Z[k].flatten().argmax()), Z.shape[2])
            if best > params.get("thr_top", 8.0):
                hit = [live[k]]; flagged.update(hit); trigs.append(P[q]); new = True
            log.append({"round": r, "score": best, "cand": list(P[q]), "cls": c, "flagged": hit})
            best = -1e9   # pula o ramo "shared"
        else:
            sc = Z.topk(2, dim=0).values[1]
            best = float(sc.max()); q, c = divmod(int(sc.flatten().argmax()), sc.shape[1])
        if best > params.get("thr", 3.0):
            hit = [live[k] for k in (Z[:, q, c] > params.get("thr", 3.0)).nonzero().flatten().tolist()]
            if len(hit) <= deltas.shape[0] // 2:   # nunca exclui mais da metade (sem consenso de gatilho)
                flagged.update(hit); trigs.append(P[q]); new = True
            else:
                hit = []
        if params.get("mode", "shared") != "top1":
            log.append({"round": r, "score": best, "cand": list(P[q]), "cls": c, "flagged": hit})
        del fis, fg
    keep = [i for i in range(deltas.shape[0]) if i not in flagged]
    agg, info = DEFENSE_REGISTRY[params.get("base", "fedavg")](deltas[keep], weights[keep], ctx, params.get("base_params", {}))
    if new and trigs:
        m = ctx["model_ctor"]().to(dev); load_vector(m, (ctx["global_vector"] + agg.cpu()).to(dev))
        g = torch.Generator().manual_seed(r)
        opt = torch.optim.SGD(m.parameters(), lr=params.get("unlearn_lr", 0.001), momentum=0.9)
        m.train()
        for _ in range(params.get("unlearn_epochs", 10)):
            perm = torch.randperm(len(X), generator=g).to(dev)
            for b0 in range(0, len(X), 32):
                b = perm[b0:b0 + 32]; x, y = X[b], Y[b]
                half = len(b) // 2
                xt = x.clone()
                for k in range(half):   # alterna os gatilhos encontrados
                    xt[k:k + 1] = _apply(x[k:k + 1], trigs[k % len(trigs)], dev)
                opt.zero_grad(); F.cross_entropy(m(xt), y).backward(); opt.step()
        agg = flatten(m).cpu() - ctx["global_vector"]
    return agg, {**{k: v for k, v in info.items() if k != "excluded_indices"},
                 "excluded_indices": sorted(flagged), "trigger_scan": st.get("scan_log", [])[-1:] }
