"""Os 8 ataques do fl-robustness-lab portados para o framework do CoUrb (NumPy/Keras).

Mesma lógica de fl-robustness-lab/src/attacks:
  - de modelo, "oniscientes" (gaussian_noise, ipm, lie, minmax, minsum): sintetizados no
    servidor a partir dos deltas benignos do round e injetados no lugar das atualizações
    dos clientes maliciosos (mesma modelagem do laboratório / BOBA / Blades);
  - de dados (label_flip, backdoor_pixel, backdoor_pgd): aplicados no treino local do cliente.

Diferença de escala: aqui as imagens estão em [0, 1] (pipeline do CoUrb), então o gatilho
do backdoor usa valor 1.0 (pixel branco saturado) em vez de 2.5 (pós-normalização no lab).
"""
import numpy as np

MODEL_ATTACKS = {"gaussian_noise", "sign_flip", "ipm", "lie", "minmax", "minsum"}
DATA_ATTACKS = {"label_flip", "backdoor_pixel", "backdoor_pgd"}

# parâmetros iguais aos de fl-robustness-lab/experiments/ragnar_eval.py
ATTACK_PARAMS = {
    "gaussian_noise": {"std": 5.0},
    "sign_flip": {"scale": 1.0},
    "ipm": {"epsilon": 1.0},
    "lie": {"z": 1.5},
    "minmax": {"perturbation": "std"},
    "minsum": {"perturbation": "std"},
    "label_flip": {"source": 0, "target": 2},
    "backdoor_pixel": {"target_label": 0, "poison_fraction": 0.5},
    "backdoor_pgd": {"target_label": 0, "poison_fraction": 0.5, "scale": 3.0},
}
NUM_MALICIOUS = {a: (2 if a.startswith("backdoor") else 4) for a in ATTACK_PARAMS}

TRIGGER_VALUE = 1.0


# ---------------------------------------------------------------- ataques de modelo
def gaussian_noise(benign: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, p.get("std", 1.0), size=benign.shape[1]).astype(np.float32)


def sign_flip(benign, p, rng):
    return -p.get("scale", 1.0) * benign.mean(0)


def ipm(benign, p, rng):
    return -p.get("epsilon", 1.0) * benign.mean(0)


def lie(benign, p, rng):
    return benign.mean(0) + p.get("z", 1.5) * benign.std(0, ddof=1)


def _pairwise_sq(x):
    sq = (x ** 2).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0)


def _binary_search(benign, p, within):
    mean = benign.mean(0)
    direction = -benign.std(0, ddof=1) if p.get("perturbation", "std") == "std" else -np.sign(mean)
    lo, hi = 0.0, 50.0
    for _ in range(25):
        g = (lo + hi) / 2
        if within(mean + g * direction):
            lo = g
        else:
            hi = g
    return mean + lo * direction


def minmax(benign, p, rng):
    limit = _pairwise_sq(benign).max()
    return _binary_search(benign, p, lambda m: ((benign - m) ** 2).sum(1).max() <= limit)


def minsum(benign, p, rng):
    limit = _pairwise_sq(benign).sum(1).max()
    return _binary_search(benign, p, lambda m: ((benign - m) ** 2).sum() <= limit)


CRAFT = {"gaussian_noise": gaussian_noise, "sign_flip": sign_flip, "ipm": ipm,
         "lie": lie, "minmax": minmax, "minsum": minsum}


# ---------------------------------------------------------------- ataques de dados
def poison_data(x: np.ndarray, y: np.ndarray, attack: str, seed: int = 0):
    """Devolve (x, y) envenenados para o treino local do cliente malicioso."""
    p = ATTACK_PARAMS[attack]
    x, y = x.copy(), y.copy().reshape(-1)
    if attack == "label_flip":
        y[y == p["source"]] = p["target"]
    elif attack in ("backdoor_pixel", "backdoor_pgd"):
        mask = np.random.default_rng(seed).random(len(y)) < p["poison_fraction"]
        x[mask] = add_trigger(x[mask])
        y[mask] = p["target_label"]
    return x, y


def add_trigger(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[:, -4:, -4:, :] = TRIGGER_VALUE  # NHWC (Keras)
    return x


# ---------------------------------------------------------------- utilidades
def flatten(weights: list) -> np.ndarray:
    return np.concatenate([w.reshape(-1) for w in weights]).astype(np.float64)


def unflatten(vector: np.ndarray, like: list) -> list:
    out, pos = [], 0
    for w in like:
        out.append(vector[pos:pos + w.size].reshape(w.shape).astype(w.dtype))
        pos += w.size
    return out
