"""Desenvolvimento (adendo 5): para cada classe, inverte a máscara (NC), mantém os k pixels de maior máscara
(binária), reotimiza só o padrão e mede o sucesso nas amostras do holdout; z robusto entre classes."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn.functional as F
from experiments.fase2_repair import cfg_from_name, holdout_for, OUT
from src.defenses.nc import invert, _tensors
from src.federated.utils import load_vector
from src.models.cnn import build_model
dev = torch.device("cuda")


def kbudget(m, X, Y, c, k, seed, steps=300):
    mk, p, l1, _ = invert(m, X, Y, c, dev, lam=3e-2, steps=500, seed=seed)
    flat = mk.flatten(); keep = torch.zeros_like(flat); keep[flat.topk(k).indices] = 1
    M = keep.view_as(mk)
    x = X[Y != c].to(dev); tgt = torch.full((len(x),), c, device=dev)
    for q in m.parameters(): q.requires_grad_(False)
    pr = p.clone().atanh().clamp(-3, 3).div(1).requires_grad_(True) if False else (torch.atanh((p / 3).clamp(-0.999, 0.999))).clone().requires_grad_(True)
    opt = torch.optim.Adam([pr], lr=0.1)
    for _ in range(steps):
        loss = F.cross_entropy(m((1 - M) * x + M * 3 * torch.tanh(pr)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = (m((1 - M) * x + M * 3 * torch.tanh(pr)).argmax(1) == c).float().mean().item()
    for q in m.parameters(): q.requires_grad_(True)
    return s


if __name__ == "__main__":
    k = int(sys.argv[1])
    for name in sys.argv[2:]:
        cfg = cfg_from_name(name); hold, _ = holdout_for(cfg)
        m = build_model(cfg.model).to(dev); load_vector(m, torch.load(OUT / "models" / f"{name}.pt", weights_only=True).to(dev))
        X, Y = _tensors(hold)
        s = torch.tensor([kbudget(m, X, Y, c, k, cfg.seed) for c in range(10)])
        med = s.median(); mad = 1.4826 * (s - med).abs().median() + 1e-9
        print(f"k={k} {name[:42]:42s} succ {[round(v, 2) for v in s.tolist()]} z máx {((s - med) / mad).max():.1f} classe {int(s.argmax())}", flush=True)
