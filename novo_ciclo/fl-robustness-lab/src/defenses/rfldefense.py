"""Fase 6 (25/10/2026): RFLDefense (Shi, Gong e He, Cybersecurity 9:234, out/2026), reimplementada a partir do texto,
sem código público. É a defesa publicada mais parecida com o nosso sinal: também procura conluio comparando clientes
na mesma camada. Os autores não a avaliaram contra o LP.

  1. pré-filtro: CKA linear entre a penúltima camada de pesos de cada cliente (global + atualização) e a do global
     anterior (linhas = amostras); k-means com k = 2 nos valores de CKA (exato em 1-D); fica o grupo com a maior
     mediana de CKA (empate: o maior grupo);
  2. por tensor, conluio absoluto (Eq. 10, segue a FPD): cs_k = nº de outros com cosseno > beta; varrendo os pares
     com cosseno > beta, entre os dois fica o de menor contagem (empate: fica o de menor índice);
  3. por tensor, teste de sinal (Eq. 11–14): sinal majoritário; escore = nº dos top-k |valores| do cliente que
     concordam com ele; MZ = 0,6745·|escore − mediana| / MAD; sai se MZ > Z; pula o teste se MAD < 1;
  4. por tensor, Isolation Forest (scikit-learn) nas atualizações que sobraram; s = −decision_function normalizado
     em [0, 1] (mín-máx); a = mediana + 1,5·MAD, b = mediana + 3·MAD; peso 1 até a, ((b − s)/(b − a))² até b,
     0,1 acima de b (Eq. 15–18; usamos 0,1 também como piso entre a e b, para o peso não
     chegar a zero e depois saltar para 0,1);
  5. tensor agregado = κ · Σ w_i·d_i / Σ w_i, d_i = x_i/‖x_i‖, κ = mediana das normas dos que sobraram (Eq. 19–20).

Valores que o artigo não dá, fixados ANTES de rodar: beta = 0,8 (valor da FPD, arXiv 2308.03331, que o artigo
segue); top-k = 30% do tensor (o mesmo da AlignIns); Z = 3,5 (corte usual do MZ de Iglewicz e Hoaglin com 0,6745);
Isolation Forest com os padrões do scikit-learn (100 árvores, contamination='auto') e semente fixa; kernel linear
no CKA. Interpretação: a Eq. 19 normaliza x_i·w_i, o que anula o peso; usamos a média dos d_i ponderada por w_i,
que é o efeito descrito no texto ("direções anômalas são enfraquecidas").
"""
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from src.defenses.lasa_alignins import _shapes


def _cka(x, y):
    x = x - x.mean(0, keepdim=True); y = y - y.mean(0, keepdim=True)
    k, l = x @ x.T, y @ y.T
    return float((k * l).sum() / (k.norm() * l.norm()).clamp_min(1e-12))


def _kmeans2(v):
    """k-means 1-D com k = 2, solução exata pelo melhor corte da lista ordenada. Devolve a máscara do grupo mantido."""
    order = np.argsort(v); sv = v[order]; n = len(v)
    if n < 2 or sv[-1] - sv[0] < 1e-12:
        return np.ones(n, bool)
    sse = [((sv[:c] - sv[:c].mean()) ** 2).sum() + ((sv[c:] - sv[c:].mean()) ** 2).sum() for c in range(1, n)]
    c = int(np.argmin(sse)) + 1
    lo, hi = order[:c], order[c:]
    keep_hi = np.median(v[hi]) > np.median(v[lo]) or (np.median(v[hi]) == np.median(v[lo]) and len(hi) >= len(lo))
    m = np.zeros(n, bool); m[hi if keep_hi else lo] = True
    return m


def _mad(v):
    return np.median(np.abs(v - np.median(v)))


def rfldefense(deltas, weights, ctx, params):
    beta, topk, z, wmin = params.get("beta", 0.8), params.get("topk", 0.3), params.get("z", 3.5), params.get("w_min", 0.1)
    shapes = _shapes(ctx)
    n = deltas.shape[0]
    gv = ctx["global_vector"].to(deltas.dtype)
    rng = 1000 * int(ctx.get("seed", 0) or 0) + int(ctx.get("round_idx", 0) or 0)

    # 1. pré-filtro CKA na penúltima camada de pesos (linhas = canais/neurônios de saída)
    if "_pen" not in ctx["state"]:
        params_ = [(p.shape[0], s) for (_, p), (_, s, d) in zip(ctx["model_ctor"]().named_parameters(), shapes) if d >= 2]
        ctx["state"]["_pen"] = params_[-2]
    rows, ps = ctx["state"]["_pen"]
    ref = gv[ps].view(rows, -1)
    cka = np.array([_cka((gv[ps] + deltas[i, ps]).view(rows, -1), ref) for i in range(n)])
    sel = np.where(_kmeans2(cka))[0]

    agg = torch.zeros(deltas.shape[1], dtype=deltas.dtype)
    kept = np.zeros(n); flagged = set(range(n)) - set(sel.tolist())
    for _, s, _ in shapes:
        g = deltas[sel, s]
        m = len(sel)
        alive = np.ones(m, bool)
        # 2. conluio absoluto
        if m > 1:
            u = torch.nn.functional.normalize(g, dim=1, eps=1e-12)
            cos = (u @ u.T).numpy(); np.fill_diagonal(cos, -1)
            cs = (cos > beta).sum(1)
            for a in range(m):
                for b in range(a + 1, m):
                    if alive[a] and alive[b] and cos[a, b] > beta:
                        alive[b if cs[b] >= cs[a] else a] = False
        # 3. teste de sinal
        idx = np.where(alive)[0]
        if len(idx) > 2:
            ga = g[idx]
            major = torch.sign(torch.sign(ga).sum(0))
            k = max(1, int(ga.shape[1] * topk))
            sc = np.array([float((torch.sign(r[t]) == major[t]).sum()) for r in ga for t in [torch.topk(r.abs(), k).indices]])
            mad = _mad(sc)
            if mad >= 1:
                alive[idx[np.abs(0.6745 * (sc - np.median(sc)) / mad) > z]] = False
        idx = np.where(alive)[0]
        if len(idx) == 0:
            idx = np.arange(m)
        ga = g[idx]
        # 4. Isolation Forest -> pesos
        w = np.ones(len(idx))
        if len(idx) > 2:
            sc = -IsolationForest(random_state=rng).fit(ga.numpy()).decision_function(ga.numpy())
            sc = (sc - sc.min()) / (sc.max() - sc.min()) if sc.max() > sc.min() else np.zeros_like(sc)
            med, mad = np.median(sc), _mad(sc)
            a, b = med + 1.5 * mad, med + 3 * mad
            for i, si in enumerate(sc):
                if si > b:
                    w[i] = wmin
                elif si > a:
                    w[i] = max(((b - si) / (b - a)) ** 2, wmin)
        # 5. direções normalizadas × norma mediana
        nr = ga.norm(dim=1)
        d = ga / nr.clamp_min(1e-12).unsqueeze(1)
        wt = torch.tensor(w, dtype=deltas.dtype)
        agg[s] = nr.median() * (d * wt.unsqueeze(1)).sum(0) / wt.sum()
        flagged.update(sel[np.setdiff1d(np.arange(m), idx)].tolist())
        kept[sel[idx]] += (s.stop - s.start)
    kept /= deltas.shape[1]
    return agg, {"excluded_indices": sorted(flagged), "kept_fraction": kept.tolist(), "cka_kept": sel.tolist()}

