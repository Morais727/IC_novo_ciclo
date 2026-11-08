"""RAGNAR (Morais et al., 2024) — filtro pré-agregação leve baseado em
características geométricas (L1, L2, L3, similaridade de cosseno com o
modelo global) + classificador supervisionado XGBoost treinado com
histórico de simulação (bootstrap).

O bootstrap (coleta de (features, rótulo malicioso/benigno) e treino do
classificador) é conduzido por `src/federated/simulate.py` antes das rounds
de avaliação e o classificador treinado é armazenado em
ctx["state"]["ragnar_classifier"].

Parâmetros relevantes (defense_params):
  - cos_ref: referência da similaridade de cosseno. "mean_delta" (padrão,
    média dos deltas do round) ou "global_model" (vetor de pesos do modelo
    global, como descrito na revisão).
  - bootstrap_exclude_eval_attack (padrão True): exclui a família do ataque
    avaliado do bootstrap (ver `ragnar_bootstrap_pool` em simulate.py).
  - offline_dataset: diretório com CSVs de simulações prévias completas (gerados por
    experiments/signal_analysis.py com outra semente). Quando dado, o classificador é
    treinado uma vez com esses dados, antes do FL, e fica congelado — fiel ao RAGNAR
    original. Sem ele, usa o bootstrap curto antigo (5 rounds a partir do modelo inicial).
  - start_round (padrão 2): o filtro atua a partir deste round.
  - bootstrap_rounds, n_estimators, max_depth, learning_rate.
"""
import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb


def extract_features(deltas: torch.Tensor, ref: torch.Tensor,
                     cos_ref: torch.Tensor | None = None) -> np.ndarray:
    """Features [L1, L2, L3, cosseno]. As normas são da diferença para `ref`;
    o cosseno é contra `cos_ref` (ou `ref`, se None)."""
    diff = deltas - ref
    l1 = diff.abs().sum(dim=1)
    l2 = diff.norm(p=2, dim=1)
    l3 = diff.abs().pow(3).sum(dim=1).pow(1.0 / 3.0)
    cref = ref if cos_ref is None else cos_ref.reshape(1, -1)
    cos = F.cosine_similarity(deltas, cref.expand_as(deltas), dim=1)
    return torch.stack([l1, l2, l3, cos], dim=1).numpy()


def train_classifier(features: np.ndarray, labels: np.ndarray, params: dict) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 60),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.2),
        eval_metric="logloss",
    )
    clf.fit(features, labels)
    return clf


def feature_columns(cos_ref: str = "mean_delta") -> list[str]:
    """Colunas de src/signals.py equivalentes às features de `extract_features`."""
    return ["l1_mean", "l2_mean", "l3_mean", "cos_global" if cos_ref == "global_model" else "cos_mean"]


def train_offline(dataset_dir: str, pool: list[str], params: dict) -> xgb.XGBClassifier:
    """Treina o classificador com um conjunto offline de simulações prévias completas
    (como no RAGNAR original): CSVs de experiments/signal_analysis.py, um por ataque,
    com as features de cada cliente em cada round e o rótulo malicioso/honesto."""
    import pandas as pd
    from pathlib import Path
    frames = [pd.read_csv(Path(dataset_dir) / f"{a}.csv") for a in pool if (Path(dataset_dir) / f"{a}.csv").exists()]
    if not frames:
        raise FileNotFoundError(f"nenhum CSV do pool {pool} em {dataset_dir}")
    df = pd.concat(frames)
    cols = feature_columns(params.get("cos_ref", "mean_delta"))
    return train_classifier(df[cols].values, df["malicious"].values, params)


def ragnar(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    ref = deltas.mean(dim=0, keepdim=True)
    cos_ref = ctx["global_vector"] if params.get("cos_ref", "mean_delta") == "global_model" else None
    feats = extract_features(deltas, ref, cos_ref)

    clf = ctx.get("state", {}).get("ragnar_classifier")
    # o RAGNAR original começa a filtrar no 2º round (round 1: agregação sem filtro)
    if clf is None or ctx.get("round_idx", 1) < params.get("start_round", 2):
        w = weights.float()
        w = w / w.sum()
        agg = (deltas * w.unsqueeze(1)).sum(dim=0)
        return agg, {"excluded_indices": [], "warning": "RAGNAR inativo neste round (sem classificador ou antes de start_round)"}

    preds = clf.predict(feats)
    keep = [i for i in range(n) if preds[i] == 0]
    excluded = [i for i in range(n) if preds[i] == 1]
    if not keep:
        keep = list(range(n))
        excluded = []

    w = weights[keep].float()
    w = w / w.sum()
    agg = (deltas[keep] * w.unsqueeze(1)).sum(dim=0)
    return agg, {"excluded_indices": excluded, "selected_indices": keep}
