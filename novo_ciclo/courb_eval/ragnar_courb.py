"""RAGNAR como em CoUrb-2024/servidor_principal.py (CNN).

Para cada tensor i = 0..camada_alvo (8 na CNN -> 9 tensores), com w_c = pesos do cliente e
w_g = último modelo global:
    [L1(w_c), L1(w_c)-L1(w_g), L2(w_c), L2(w_c)-L2(w_g), L3(w_c), L3(w_c)-L3(w_g)]
-> 54 features -> MinMaxScaler -> XGBoost; exclui se P(malicioso) > 0.5. Atua a partir do round 2.

Classificadores:
  pretrained  — CoUrb-2024/MODELOS/{MINMAX,CLASSIFICADOR}_XGB_CNN (formato binário do XGBoost,
                exige xgboost 3.0.x)
  bootstrap5  — mesma receita de CoUrb-2024/XGB_CLASSIFIER.py (MinMax + XGBClassifier(max_depth=25),
                treino em 80% dos dados), com features coletadas em 5 rounds simulados no início
"""
import pickle

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

COURB_MODELOS = "/home/marcos-morais/Documentos/IC/CoUrb-2024/MODELOS"
CAMADA_ALVO = 8
THRESHOLD = 0.5


def courb_features(client_w: list, global_w: list, camada_alvo: int = CAMADA_ALVO) -> list:
    feats = []
    for i in range(camada_alvo + 1):
        c = client_w[i].reshape(-1).astype(np.float64)
        g = global_w[i].reshape(-1).astype(np.float64)
        n1c, n1g = np.abs(c).sum(), np.abs(g).sum()
        n2c, n2g = np.sqrt((c ** 2).sum()), np.sqrt((g ** 2).sum())
        n3c, n3g = np.cbrt((np.abs(c) ** 3).sum()), np.cbrt((np.abs(g) ** 3).sum())
        feats += [n1c, n1c - n1g, n2c, n2c - n2g, n3c, n3c - n3g]
    return feats


def load_pretrained() -> dict:
    with open(f"{COURB_MODELOS}/MINMAX_XGB_CNN.pkl", "rb") as f:
        scaler = pickle.load(f)
    return {"scaler": scaler, "booster": xgb.Booster(model_file=f"{COURB_MODELOS}/CLASSIFICADOR_XGB_CNN.model")}


def train_like_courb(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    scaler = MinMaxScaler().fit(X)
    Xs = scaler.transform(X)
    X_tr, _, y_tr, _ = train_test_split(Xs, y, test_size=0.2, random_state=seed)
    clf = xgb.XGBClassifier(max_depth=25, random_state=seed)
    clf.fit(X_tr, y_tr)
    return {"scaler": scaler, "booster": clf.get_booster()}


def predict_proba(clf: dict, feats: np.ndarray) -> tuple[np.ndarray, float]:
    scaled = clf["scaler"].transform(feats)
    out_of_range = float(((scaled < -0.05) | (scaled > 1.05)).mean())
    return clf["booster"].predict(xgb.DMatrix(scaled)), out_of_range
