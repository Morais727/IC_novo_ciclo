#!/usr/bin/env python
"""Testes dos ataques recentes de src/attacks/recent.py (dados sintéticos, CPU, segundos).
    python tests/test_recent_attacks.py        # ou: pytest tests/test_recent_attacks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.attacks.recent import DBADataset, craft_lp, layer_groups, mimic, select_edge_cases, tensor_groups
from src.data.datasets import TRIGGER_VALUE
from src.models.cnn import build_model
from src.federated.utils import flatten


def test_mimic_copies_the_same_honest_client():
    g = torch.Generator().manual_seed(0)
    benign = torch.randn(16, 50, generator=g)
    benign[5] += 10 * torch.randn(50, generator=g)      # cliente mais "extremo" na direção principal
    params = {}
    first = mimic(benign, params)
    assert params["_mimic_idx"] == 5 and torch.equal(first, benign[5])
    # nos rounds seguintes continua imitando o mesmo cliente, mesmo que outro fique mais extremo
    benign2 = torch.randn(16, 50, generator=g)
    assert torch.equal(mimic(benign2, params), benign2[5])


def test_dba_parts_tile_the_global_trigger():
    x = torch.zeros(8, 3, 32, 32)
    base = TensorDataset(x, torch.ones(8, dtype=torch.long))
    union = torch.zeros(3, 32, 32, dtype=torch.bool)
    for part in range(4):
        ds = DBADataset(base, target_label=0, part=part, poison_fraction=1.0)
        img, y = ds[0]
        mark = img == TRIGGER_VALUE
        assert y == 0 and mark.sum() == 3 * 4          # uma faixa 1x4 em 3 canais
        assert not (union & mark).any()                 # faixas disjuntas
        union |= mark
    assert union[:, -4:, -4:].all() and union.sum() == 3 * 16   # juntas = o gatilho 4x4 da avaliação


def test_edge_cases_are_the_tail_of_the_source_class():
    class Fake:
        def __init__(self, n, seed):
            r = np.random.default_rng(seed)
            self.targets = list(r.integers(0, 3, n))
            self.data = r.integers(100, 110, (n, 4, 4, 3)).astype(np.uint8)
            src = np.where(np.array(self.targets) == 0)[0]
            self.data[src[:3]] = 250                       # 3 amostras atípicas da classe 0
            self.outliers = set(src[:3].tolist())
    tr, te = Fake(300, 0), Fake(90, 1)
    edge_tr, edge_te = select_edge_cases(tr, te, source=0, fraction=0.03)
    assert all(tr.targets[i] == 0 for i in edge_tr) and all(te.targets[i] == 0 for i in edge_te)
    assert tr.outliers & set(edge_tr.tolist())


def test_layer_groups_cover_the_parameter_vector():
    model = build_model("smallcnn")
    groups = layer_groups(model)
    assert list(groups) == ["features.0", "features.3", "features.6", "fc1", "fc2"]
    total = sum(sl.stop - sl.start for sl in groups.values())
    assert total == flatten(model).numel()


def test_craft_lp_follows_eq4_of_the_paper():
    model = build_model("smallcnn")
    groups = tensor_groups(model)
    assert sum(sl.stop - sl.start for sl in groups.values()) == flatten(model).numel()
    D = flatten(model).numel()
    g = torch.Generator().manual_seed(0)
    center, mal = torch.randn(D, generator=g), torch.randn(D, generator=g)
    bc = groups["fc1.weight"]
    d = craft_lp(center, mal, ["fc1.weight"], groups, 0.5)
    assert torch.allclose(d[bc], 0.5 * mal[bc] + 0.5 * center[bc])     # camada crítica: interpolação
    rest = torch.ones(D, dtype=torch.bool); rest[bc] = False
    assert torch.equal(d[rest], center[rest])                           # demais camadas: o centro
    d2 = craft_lp(center, mal, ["fc1.weight"], groups, 2.0)
    assert torch.allclose(d2[bc], 2.0 * mal[bc])                        # lambda > 1: ReLU(1 - lambda) = 0


def _toy(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    return [(x[i], int(y[i])) for i in range(n)]          # rótulo int, como o CIFAR-10 e o BackdoorDataset


HP = dict(epochs=1, batch_size=16, lr=0.01)


def test_train_ext_freezes_the_other_tensors():
    from src.attacks.recent import _train_ext
    ctor = lambda: build_model("smallcnn")
    torch.manual_seed(0)
    gv = flatten(ctor())
    groups = tensor_groups(ctor())
    start = torch.zeros_like(gv)
    d = _train_ext(ctor, start, gv, _toy(), torch.device("cpu"), HP, trainable=["fc1.weight"], anchor_beta=20.0)
    for n, sl in groups.items():
        assert (d[sl].abs().sum() > 0) == (n == "fc1.weight"), n   # só o tensor treinável muda


def test_lsa_sends_the_attackers_mean_outside_the_critical_layers():
    from src.attacks.recent import lsa_attack
    from src.federated.client import local_train
    ctor, dev = (lambda: build_model("smallcnn")), torch.device("cpu")
    torch.manual_seed(0)
    gv = flatten(ctor())
    groups = tensor_groups(ctor())
    seen = []
    def train(ds):
        out = local_train(ctor, gv, ds, dev, 1, 16, 0.01)
        seen.append(out[0])
        return out
    params = {"_hp": HP, "target_label": 0, "lambda": 1.0}
    d = lsa_attack(train, ctor, gv, [_toy(seed=1), _toy(seed=2)], torch.zeros(3, gv.numel()), dev, params)
    u_a = torch.stack(seen[:2]).mean(0)                          # os 2 treinos limpos dos atacantes
    layers = params["_bc_layers"]
    assert layers and params["_round"] == 0
    for n, sl in groups.items():
        if n not in layers:
            assert torch.allclose(d[sl], u_a[sl], atol=1e-6), n     # fora de L*: média limpa dos atacantes


def test_laysel_clips_the_normal_layers():
    from src.attacks.recent import laysel_attack
    from src.federated.client import local_train
    ctor, dev = (lambda: build_model("smallcnn")), torch.device("cpu")
    torch.manual_seed(0)
    gv = flatten(ctor())
    groups = tensor_groups(ctor())
    train = lambda ds: local_train(ctor, gv, ds, dev, 1, 16, 0.5)   # lr alto: força deltas > 0,05
    params = {"_hp": dict(HP, lr=0.5), "target_label": 0, "N": 1}
    laysel_attack(train, ctor, gv, [_toy(seed=1), _toy(seed=2)], torch.zeros(3, gv.numel()), dev, params)
    ds = params.pop("_lp_individual")
    log = params["_lp_log"][0]["attackers"]
    assert len(ds) == 2 and all(a["n_sens"] == len(groups) - 2 for a in log)
    for d, a in zip(ds, log):
        for n in a["normal"]:
            assert d[groups[n]].abs().max() <= 0.05 + 1e-6, n           # camadas normais cortadas em ±0,05


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} testes passaram")
