#!/usr/bin/env python
"""Testes das formulações robustas de src/signals.py (dados sintéticos, CPU).
    python tests/test_signals.py        # ou: pytest tests/test_signals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.signals import geometric_median, mcd_mahalanobis, robust_z, spectral_scores


def _honest_plus_attackers(n_honest=16, n_mal=4, d=200, shift=8.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    honest = torch.randn(n_honest, d, generator=g)
    direction = torch.randn(d, generator=g)
    mal = torch.randn(n_mal, d, generator=g) * 0.3 + shift * direction / direction.norm() * d ** 0.5 / 4
    return torch.cat([honest, mal]), n_honest


def test_geometric_median_resists_attackers_better_than_mean():
    """O desvio causado pelos atacantes (erro acima do piso amostral = média só dos honestos)
    deve ser bem menor na mediana geométrica do que na média."""
    x, nh = _honest_plus_attackers()
    floor = x[:nh].mean(dim=0).norm()
    excess_gm = geometric_median(x).norm() - floor
    excess_mean = x.mean(dim=0).norm() - floor
    assert excess_gm < 0.5 * excess_mean


def test_robust_z_ignores_outliers():
    v = torch.cat([torch.randn(18), torch.tensor([50.0, 60.0])])
    z = robust_z(v)
    assert z[-2:].min() > 10          # outliers ficam bem destacados
    assert z[:18].abs().max() < 5     # honestos não são inflados pelo outlier (ao contrário do z clássico)


def test_spectral_score_flags_colluding_direction():
    x, nh = _honest_plus_attackers()
    s = spectral_scores(x, x.median(dim=0).values)
    assert s[nh:].min() > s[:nh].max()


def test_mcd_mahalanobis_avoids_masking():
    rng = np.random.default_rng(0)
    honest = rng.normal(size=(16, 3))
    mal = rng.normal(size=(4, 3)) * 0.1 + np.array([6.0, 6.0, 6.0])  # grupo coeso de outliers
    d = mcd_mahalanobis(np.vstack([honest, mal]))
    assert d[16:].min() > d[:16].max()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} testes passaram")
