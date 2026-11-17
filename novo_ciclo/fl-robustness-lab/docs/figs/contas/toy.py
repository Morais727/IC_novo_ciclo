"""Exemplo de brinquedo: 6 clientes (5 e 6 atacantes), 3 tensores de 3 coordenadas; verificado contra layer_guard.py."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/marcos.morais/IC/novo_ciclo/fl-robustness-lab")
from src.defenses import layer_guard as LG
G = {"A": slice(0, 3), "B": slice(3, 6), "C": slice(6, 9)}
W = torch.tensor([400., 200, 200, 400, 400, 400])
rz = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)

def scores(D):
    C = D - D.median(0).values
    out = {}
    for g, s in G.items():
        x = C[:, s]
        u = x / x.norm(dim=1, keepdim=True)
        cm = u @ u.T
        cs = LG._max_other_cos(x).numpy(); nr = x.norm(dim=1).numpy()
        zs, zn = rz(cs), rz(nr)
        out[g] = dict(C=x.numpy(), cm=cm.numpy(), s=cs, nr=nr, zs=zs, zn=zn, sc=np.maximum(zs, 0) + np.maximum(zn, 0))
    return C, out

def ok(D):
    _, o = scores(D)
    f = np.array([o[g]["sc"] > 3 for g in G])           # 3 x 6
    zs = np.concatenate([np.abs(o[g]["zs"]) for g in G] + [np.abs(o[g]["zn"]) for g in G])
    if zs.max() > 25: return False
    cnt = f.sum(0)
    return (f[0, 4] and f[0, 5] and not f[0, :4].any()       # A: os dois atacantes, nenhum honesto
            and f[1, 1] and f[1, [0, 2, 3]].sum() == 0        # B: honesto 2 acusado por engano
            and cnt[4] >= 2 and cnt[5] >= 2 and cnt[1] == 1 and cnt[[0, 2, 3]].sum() == 0
            and (f[1, 4] + f[1, 5]) >= 1 and not (f[2, 4] and f[2, 5] and f[1, 4] and f[1, 5]))

rng = np.random.default_rng(0)
for t in range(200000):
    D = np.round(rng.normal(0, 1, (6, 9)), 1)
    b = np.round(rng.normal(0, 1.5, 9), 1)
    D[4] = np.round(D[4] * 0.3 + b, 1); D[5] = np.round(D[5] * 0.3 + b, 1)
    Dt = torch.tensor(D, dtype=torch.float32)
    if ok(Dt):
        break
print("tentativa", t)
C, o = scores(Dt)
ctx = lambda: {"state": {"tensor_groups": G}, "model_ctor": None}
res = {}
for name, fn, p in [("fedavg", None, {}), ("sn", LG.layer_guard_sn, {}), ("corr", LG.layer_guard_corr, {"confirm_tensors": 2}),
                    ("corr_soft", LG.layer_guard_corr_soft, {"confirm_tensors": 2}), ("client", LG.layer_guard_client, {}),
                    ("tm", LG.layer_guard_corr_soft_tm, {"confirm_tensors": 2}), ("clip", LG.layer_guard_corr_soft_clip, {"confirm_tensors": 2})]:
    res[name] = ((Dt * (W / W.sum()).unsqueeze(1)).sum(0) if fn is None else fn(Dt, W, ctx(), p)[0]).numpy().round(4).tolist()
honest = (Dt[:4] * (W[:4] / W[:4].sum()).unsqueeze(1)).sum(0).numpy()
json.dump({"D": D.tolist(), "W": W.tolist(), "med": Dt.median(0).values.numpy().tolist(), "C": C.numpy().round(4).tolist(),
           "tens": {g: {k: np.round(v, 4).tolist() for k, v in o[g].items()} for g in G}, "agg": res,
           "honest": honest.round(4).tolist()}, open("toy.json", "w"), indent=1)
print(D); print({g: o[g]["sc"].round(2) for g in G}); print({k: np.round(v, 2) for k, v in res.items()}); print("honest", honest.round(2))
