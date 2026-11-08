"""Desenvolvimento (adendo 6): NC-U forçado — desaprende o gatilho reconstruído das `top` classes de menor máscara,
sem limiar de detecção. Mede custo no modelo limpo e efeito nos LP não detectados. Só modelos de desenvolvimento."""
import copy, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from experiments.fase2_repair import cfg_from_name, holdout_for, OUT
from src.data.datasets import BackdoorEvalDataset
from src.defenses.nc import _tensors, invert, nc_unlearn
from src.federated.utils import load_vector
from src.metrics import accuracy, backdoor_success_rate
from src.models.cnn import build_model
dev = torch.device("cuda")


def forced(m0, hold, top, seed):
    X, Y = _tensors(hold)
    inv = [invert(m0, X, Y, c, dev, seed=seed) for c in range(10)]
    order = sorted(range(10), key=lambda c: inv[c][2])[:top]
    m = copy.deepcopy(m0)
    for c in order:
        m = nc_unlearn(m, hold, dev, {"target": c, "mask": inv[c][0], "pattern": inv[c][1]}, lr=0.001, epochs=10, seed=seed)
    return m, order


if __name__ == "__main__":
    for name in sys.argv[1:]:
        cfg = cfg_from_name(name); hold, test = holdout_for(cfg)
        bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
        m0 = build_model(cfg.model).to(dev); load_vector(m0, torch.load(OUT / "models" / f"{name}.pt", weights_only=True).to(dev))
        for top in (1, 2):
            m, order = forced(m0, hold, top, cfg.seed)
            print(f"{name[:40]:40s} top{top} {order}: acc {accuracy(m, test, dev):.1f} ASR {backdoor_success_rate(m, bd, 0, dev) if bd else float('nan'):.1f}", flush=True)
