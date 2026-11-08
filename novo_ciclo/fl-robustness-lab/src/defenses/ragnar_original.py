"""RAGNAR como no artigo (CoUrb-2024/servidor_principal.py), referência adotada no projeto.

Features por tensor de parâmetros i = 0..camada_alvo (8 na CNN -> 9 tensores, na ordem do
state_dict), comparando os pesos enviados pelo cliente (w_c = global + delta) com o modelo
global anterior (w_g):

    [L1(w_c), L1(w_c) - L1(w_g), L2(w_c), L2(w_c) - L2(w_g), L3(w_c), L3(w_c) - L3(w_g)]

-> 6 x 9 tensores = 54 features. As features passam pelo MinMaxScaler treinado junto com o
classificador; cliente é excluído se P(malicioso) > 0.5. O filtro atua a partir do round 2.

Classificadores (defense_params["classifier"]):
  "pretrained" — classificador do CoUrb (CoUrb-2024/MODELOS/*_XGB_CNN), cópia convertida para
                 JSON em modelos_courb/ (o original está no formato binário do XGBoost, que só
                 abre na versão 3.0.x; previsões idênticas, verificado)
  "bootstrap5" — mesmo pipeline (54 features, MinMax, XGBClassifier(max_depth=25) como em
                 CoUrb-2024/XGB_CLASSIFIER.py), treinado com 5 rounds simulados no próprio
                 laboratório a partir do modelo inicial (construído em simulate.py)
"""
import pickle

import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

PRETRAINED_DIR = str(Path(__file__).resolve().parents[2] / "modelos_courb")
CAMADA_ALVO = 8        # como em CoUrb-2024/cliente_principal.py (CNN)
THRESHOLD = 0.5        # como em CoUrb-2024/servidor_principal.py


def param_slices(model: torch.nn.Module, camada_alvo: int = CAMADA_ALVO) -> list[slice]:
    """Fatias dos tensores 0..camada_alvo do vetor achatado (o servidor do CoUrb usa range(camada_alvo+1))."""
    out, pos = [], 0
    for _, p in model.named_parameters():
        out.append(slice(pos, pos + p.numel()))
        pos += p.numel()
    return out[:camada_alvo + 1]


def original_features(deltas: torch.Tensor, global_vector: torch.Tensor, slices: list[slice]) -> np.ndarray:
    """[N, 6·n_tensores] — idem CoUrb-2024/servidor_principal.py::aggregate_fit."""
    w_g = global_vector.double()
    w_c = (global_vector.unsqueeze(0) + deltas).double()
    cols = []
    for sl in slices:
        g, c = w_g[sl], w_c[:, sl]
        for p in (1, 2, 3):
            ng = g.abs().pow(p).sum().pow(1 / p)
            nc = c.abs().pow(p).sum(dim=1).pow(1 / p)
            cols += [nc, nc - ng]
    return torch.stack(cols, dim=1).numpy()


def load_pretrained(params: dict) -> dict:
    model_path = params.get("model_path", f"{PRETRAINED_DIR}/CLASSIFICADOR_XGB_CNN.json")
    scaler_path = params.get("scaler_path", f"{PRETRAINED_DIR}/MINMAX_XGB_CNN.pkl")
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return {"booster": booster, "scaler": scaler}


def train_like_original(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """MinMax + XGBClassifier(max_depth=25), treino em 80% dos dados — como CoUrb-2024/XGB_CLASSIFIER.py."""
    from sklearn.model_selection import train_test_split
    scaler = MinMaxScaler().fit(X)
    X_tr, _, y_tr, _ = train_test_split(scaler.transform(X), y, test_size=0.2, random_state=seed)
    clf = xgb.XGBClassifier(max_depth=25, random_state=seed)
    clf.fit(X_tr, y_tr)
    return {"booster": clf.get_booster(), "scaler": scaler}


def ragnar_original(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    clf = ctx.get("state", {}).get("ragnar_original")
    w = weights.float()
    if clf is None or ctx.get("round_idx", 1) < params.get("start_round", 2):
        agg = (deltas * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        return agg, {"excluded_indices": []}

    slices = ctx.setdefault("state", {}).setdefault("ragnar_slices", param_slices(ctx["model_ctor"]()))
    feats = original_features(deltas, ctx["global_vector"], slices)
    scaled = clf["scaler"].transform(feats)
    prob = clf["booster"].predict(xgb.DMatrix(scaled))
    thr = params.get("threshold", THRESHOLD)
    excluded = [i for i in range(n) if prob[i] > thr]
    if len(excluded) == n:  # nunca exclui todos: agrega sem filtro
        excluded = []
    keep = [i for i in range(n) if i not in excluded]
    wk = w[keep]
    agg = (deltas[keep] * (wk / wk.sum()).unsqueeze(1)).sum(dim=0)
    out_of_range = float(((scaled < -0.05) | (scaled > 1.05)).mean())
    return agg, {"excluded_indices": excluded, "selected_indices": keep,
                 "prob_max": float(prob.max()), "features_out_of_range": out_of_range}
