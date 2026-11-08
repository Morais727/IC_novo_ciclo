#!/usr/bin/env python
"""Testes unitários das correções de backdoor_pgd, BOBA e RAGNAR.

Usam dados sintéticos (sem CIFAR-10) e rodam em segundos em CPU.
    python tests/test_fixes.py        # ou: pytest tests/test_fixes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from src.attacks.backdoor import constrain_and_scale, pgd_norm_bound
from src.defenses.boba import _class_reference_deltas
from src.defenses.ragnar import extract_features
from src.federated.client import local_train
from src.federated.simulate import ragnar_bootstrap_pool
from src.federated.utils import flatten, load_vector

CPU = torch.device("cpu")
NUM_CLASSES, DIM = 4, 8


def _model_ctor():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, NUM_CLASSES))


def _synthetic(n_per_class=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(NUM_CLASSES, DIM, generator=g) * 3
    xs = torch.cat([centers[k] + torch.randn(n_per_class, DIM, generator=g) for k in range(NUM_CLASSES)])
    ys = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)
    return TensorDataset(xs, ys)


def test_pgd_projection_bounds_local_delta():
    data = _synthetic()
    g0 = flatten(_model_ctor())
    free, _ = local_train(_model_ctor, g0, data, CPU, epochs=1, batch_size=16, lr=0.1)
    radius = 0.25 * free.norm().item()
    bounded, _ = local_train(_model_ctor, g0, data, CPU, epochs=1, batch_size=16, lr=0.1, project_radius=radius)
    assert bounded.norm().item() <= radius * (1 + 1e-4)
    assert free.norm().item() > radius


def test_backdoor_pgd_is_not_a_noop():
    """Regressão: antes, max_norm = norma do próprio delta tornava scale+clip uma identidade."""
    data = _synthetic()
    g0 = flatten(_model_ctor())
    benign = torch.stack([local_train(_model_ctor, g0, _synthetic(seed=s), CPU, 1, 16, 0.1)[0] for s in range(5)])
    params = {"scale": 3.0}
    eps = pgd_norm_bound(benign, params)
    assert abs(eps - benign.norm(dim=1).median().item()) < 1e-6

    raw, _ = local_train(_model_ctor, g0, data, CPU, 1, 16, 0.1, project_radius=eps / params["scale"])
    final = constrain_and_scale(raw, {**params, "max_norm": eps})
    assert final.norm().item() <= eps * (1 + 1e-4)
    assert torch.allclose(final, raw * params["scale"], atol=1e-5)  # a escala foi de fato aplicada
    assert pgd_norm_bound(benign, {"max_norm": 1.5}) == 1.5


def test_boba_reference_deltas_descend_class_loss():
    """Os vértices do simplex devem ter o sinal de um delta de treino (reduzir a perda da classe)."""
    holdout = _synthetic(n_per_class=10, seed=1)
    g0 = flatten(_model_ctor())
    refs = _class_reference_deltas(_model_ctor, g0, holdout, NUM_CLASSES, CPU, steps=5, batch_size=10, lr=0.05)
    assert refs.shape == (NUM_CLASSES, g0.numel())

    xs, ys = holdout.tensors
    for k in range(NUM_CLASSES):
        mask = ys == k
        model = _model_ctor()
        load_vector(model, g0)
        loss_before = F.cross_entropy(model(xs[mask]), ys[mask])
        grad = torch.autograd.grad(loss_before, list(model.parameters()))
        grad = torch.cat([g.reshape(-1) for g in grad])
        load_vector(model, g0 + refs[k])
        loss_after = F.cross_entropy(model(xs[mask]), ys[mask])
        assert loss_after < loss_before, f"classe {k}: delta de referência não reduz a perda"
        assert F.cosine_similarity(refs[k], -grad, dim=0) > 0, f"classe {k}: sinal oposto ao delta de treino"


def test_ragnar_bootstrap_pool_excludes_eval_family():
    pool = ragnar_bootstrap_pool("backdoor_pixel")
    assert "backdoor_pixel" not in pool and "backdoor_pgd" not in pool
    assert "label_flip" in pool
    pool = ragnar_bootstrap_pool("minmax")
    assert "minmax" not in pool and "minsum" not in pool
    pool = ragnar_bootstrap_pool("label_flip")
    assert "label_flip" not in pool and "backdoor_pixel" in pool
    assert "label_flip" in ragnar_bootstrap_pool("label_flip", exclude_eval_attack=False)


def test_ragnar_cos_ref_option():
    torch.manual_seed(0)
    deltas = torch.randn(6, 20)
    ref = deltas.mean(dim=0, keepdim=True)
    global_vec = torch.randn(20)
    default = extract_features(deltas, ref)
    with_global = extract_features(deltas, ref, global_vec)
    assert default.shape == (6, 4)
    assert (default[:, :3] == with_global[:, :3]).all()  # normas não mudam
    expected = F.cosine_similarity(deltas, global_vec.unsqueeze(0).expand_as(deltas), dim=1).numpy()
    assert abs(with_global[:, 3] - expected).max() < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} testes passaram")
