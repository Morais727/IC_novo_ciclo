"""Diagnóstico de desenvolvimento: ASR e acurácia (teste) x fração podada de fc1, e acurácia no holdout."""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
from experiments.fase2_repair import cfg_from_name, holdout_for, OUT
from src.data.datasets import BackdoorEvalDataset
from src.defenses.repair import _acc, _fc1_activation
from src.federated.utils import load_vector
from src.metrics import accuracy, backdoor_success_rate
from src.models.cnn import build_model

dev = torch.device("cuda")
for name in sys.argv[1:]:
    cfg = cfg_from_name(name); hold, test = holdout_for(cfg)
    bd = BackdoorEvalDataset(test, 0) if cfg.attack != "none" else None
    m0 = build_model(cfg.model).to(dev); load_vector(m0, torch.load(OUT / "models" / f"{name}.pt").to(dev))
    hl = DataLoader(hold, batch_size=256)
    order = _fc1_activation(m0, hl, dev).argsort()
    print(name)
    for frac in (0, .2, .4, .5, .6, .7, .8, .9):
        k = int(frac * 128); m = copy.deepcopy(m0)
        with torch.no_grad():
            i = order[:k].to(dev); m.fc1.weight[i] = 0; m.fc1.bias[i] = 0; m.fc2.weight[:, i] = 0
        print(f"  {frac:.0%}: hold {_acc(m, hl, dev):.0f}  test {accuracy(m, test, dev):.1f}  ASR {backdoor_success_rate(m, bd, 0, dev) if bd else float('nan'):.1f}")
