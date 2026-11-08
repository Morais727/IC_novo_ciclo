"""Desenvolvimento (adendo 4): compacidade das máscaras invertidas por classe = fração da massa na melhor
janela 6x6; z robusto entre as classes."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn.functional as F
from experiments.fase2_repair import cfg_from_name, holdout_for, OUT
from src.defenses.nc import invert, _tensors
from src.federated.utils import load_vector
from src.models.cnn import build_model
dev = torch.device("cuda")
out = json.load(open(OUT / "compact.json")) if (OUT / "compact.json").exists() else {}
for name in sys.argv[1:]:
    if name in out: continue
    cfg = cfg_from_name(name); hold, _ = holdout_for(cfg)
    m = build_model(cfg.model).to(dev); load_vector(m, torch.load(OUT / "models" / f"{name}.pt", weights_only=True).to(dev))
    X, Y = _tensors(hold); rows = []
    for c in range(10):
        mk, p, l1, succ = invert(m, X, Y, c, dev, lam=3e-2, steps=500, seed=cfg.seed)
        win = F.avg_pool2d(mk, 6, stride=1) * 36
        rows.append({"l1": l1, "succ": succ, "win": win.max().item(), "frac": win.max().item() / l1})
    out[name] = rows
    json.dump(out, open(OUT / "compact.json", "w"), indent=1)
for name, rows in out.items():
    fr = torch.tensor([r["frac"] for r in rows]); med = fr.median(); mad = 1.4826 * (fr - med).abs().median() + 1e-9
    z = (fr - med) / mad
    print(f"{name[:40]:40s} frac {[round(r['frac'], 2) for r in rows]} | z máx {z.max():.1f} classe {int(z.argmax())}")
