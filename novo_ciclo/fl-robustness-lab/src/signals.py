"""Sinais observáveis pelo servidor para detecção de clientes maliciosos.

Cada sinal é um escalar por cliente por round, calculado só com o que o
servidor vê (deltas do round, modelo global, holdout limpo e histórico).
Agrupados por família:

  geometria (RAGNAR)   l1_mean, l2_mean, l3_mean, cos_mean — distância/cosseno à média do round
                       cos_global — cosseno com o vetor do modelo global (variante do RAGNAR)
  geometria (robusta)  norm, l2_median, cos_median, mahalanobis, krum_score
  conluio              max_cos_peer — maior cosseno com outro cliente no round
  camada crítica       fc1_l2_median, fc1_cos_median — só os pesos da fc1 (BC layer)
  referência servidor  holdout_loss_gain, holdout_acc, fltrust_cos
  histórico            hist_cos_self, hist_max_cos_cum, hist_ema_l2_median

Formulações robustas (v2):
  referência robusta   l2_geomed, cos_geomed — mediana geométrica (Weiszfeld, ruptura 50%)
                       no lugar da média (ruptura 0) usada pelo RAGNAR
  dispersão robusta    mcd_mahalanobis — Mahalanobis com média/covariância do MCD
                       (Rousseeuw), imune ao mascaramento pelos próprios atacantes
  espectral            spectral_score — projeção² no vetor singular principal dos deltas
                       centrados na mediana (assinaturas espectrais, Tran et al. 2018)
  invariante à escala  sign_agreement — fração de coordenadas com o sinal da mediana
                       (SignGuard / RLR); insensível à limitação de norma do PGD
  comportamento        holdout_max_class_gain, holdout_class_excess — piora de perda por
    por classe         classe no holdout (máxima, e em excesso sobre a mediana dos clientes)
                       holdout_js — divergência de Jensen-Shannon das predições vs. modelo global
  sequencial           cusum_cos_median, cusum_l2_median — CUSUM de |z robusto| entre rounds

Os sinais de histórico são NaN no 1º round em que não há histórico.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import MinCovDet

from src.federated.client import local_train
from src.federated.utils import load_vector

SIGNAL_FAMILIES = {
    "l1_mean": "geometria (RAGNAR)", "l2_mean": "geometria (RAGNAR)",
    "l3_mean": "geometria (RAGNAR)", "cos_mean": "geometria (RAGNAR)",
    "cos_global": "geometria (RAGNAR)",
    "norm": "geometria robusta", "l2_median": "geometria robusta", "cos_median": "geometria robusta",
    "mahalanobis": "geometria robusta", "krum_score": "geometria robusta",
    "max_cos_peer": "conluio",
    "fc1_l2_median": "camada crítica (fc1)", "fc1_cos_median": "camada crítica (fc1)",
    "holdout_loss_gain": "referência no servidor", "holdout_acc": "referência no servidor",
    "fltrust_cos": "referência no servidor",
    "hist_cos_self": "histórico", "hist_max_cos_cum": "histórico", "hist_ema_l2_median": "histórico",
    # --- v2: formulações robustas
    "l2_geomed": "robusto: mediana geométrica", "cos_geomed": "robusto: mediana geométrica",
    "mcd_mahalanobis": "robusto: MCD",
    "spectral_score": "robusto: espectral",
    "sign_agreement": "robusto: sinais",
    "holdout_max_class_gain": "robusto: por classe", "holdout_class_excess": "robusto: por classe",
    "holdout_js": "robusto: por classe",
    "cusum_cos_median": "robusto: sequencial", "cusum_l2_median": "robusto: sequencial",
}
SIGNALS = list(SIGNAL_FAMILIES)

# Direção esperada, declarada a priori (antes de ver os dados): +1 = malicioso tem valor MAIOR,
# -1 = malicioso tem valor MENOR. Distâncias, dispersão, dano no holdout e conluio: +1;
# alinhamento com uma referência honesta (cossenos, acurácia, concordância de sinais): -1;
# consistência entre rounds (atacante persegue o mesmo objetivo): +1.
DIRECTION = {
    "l1_mean": +1, "l2_mean": +1, "l3_mean": +1, "cos_mean": -1, "cos_global": -1,
    "norm": +1, "l2_median": +1, "cos_median": -1, "mahalanobis": +1, "krum_score": +1,
    "max_cos_peer": +1,
    "fc1_l2_median": +1, "fc1_cos_median": -1,
    "holdout_loss_gain": +1, "holdout_acc": -1, "fltrust_cos": -1,
    "hist_cos_self": +1, "hist_max_cos_cum": +1, "hist_ema_l2_median": +1,
    "l2_geomed": +1, "cos_geomed": -1, "mcd_mahalanobis": +1, "spectral_score": +1,
    "sign_agreement": -1, "holdout_max_class_gain": +1, "holdout_class_excess": +1, "holdout_js": +1,
    "cusum_cos_median": +1, "cusum_l2_median": +1,
}
ROBUST_SIGNALS = [s for s, fam in SIGNAL_FAMILIES.items() if fam.startswith("robusto")]


def layer_slice(model: torch.nn.Module, prefix: str) -> slice:
    """Fatia do vetor achatado de parâmetros que corresponde às camadas `prefix.*`."""
    start, pos, end = None, 0, None
    for name, p in model.named_parameters():
        if name.startswith(prefix + "."):
            start = pos if start is None else start
            end = pos + p.numel()
        pos += p.numel()
    if start is None:
        raise ValueError(f"camada {prefix} não encontrada")
    return slice(start, end)


def _cos_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, ref.reshape(1, -1).expand_as(x), dim=1)


def _pairwise_cos(x: torch.Tensor) -> torch.Tensor:
    xn = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return xn @ xn.T


def _max_off_diag(m: torch.Tensor) -> torch.Tensor:
    m = m.clone()
    m.fill_diagonal_(-float("inf"))
    return m.max(dim=1).values


def _zscore(v: torch.Tensor) -> torch.Tensor:
    return (v - v.mean()) / (v.std() + 1e-12)


def robust_z(v: torch.Tensor) -> torch.Tensor:
    """(v - mediana) / (1.4826·MAD): z-score com ruptura de 50%."""
    med = v.median()
    return (v - med) / (1.4826 * (v - med).abs().median() + 1e-12)


def geometric_median(x: torch.Tensor, iters: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """Mediana geométrica (minimiza a soma das distâncias L2) pelo algoritmo de Weiszfeld."""
    z = x.median(dim=0).values
    for _ in range(iters):
        w = 1.0 / (x - z).norm(dim=1).clamp_min(1e-12)
        z_new = (w.unsqueeze(1) * x).sum(dim=0) / w.sum()
        converged = (z_new - z).norm() <= tol * z.norm().clamp_min(1e-12)
        z = z_new
        if converged:
            break
    return z


def spectral_scores(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Projeção² de cada linha centrada no vetor singular principal (Tran et al. 2018).

    Usa a matriz de Gram N×N (N clientes ≪ D parâmetros): se c = x - center,
    c·v1 = σ1·u1, com (σ1², u1) o maior autopar de c·cᵀ.
    """
    c = (x - center).double()
    vals, vecs = torch.linalg.eigh(c @ c.T)
    return (vals[-1].clamp_min(0) * vecs[:, -1].pow(2)).float()


def mcd_mahalanobis(feats: np.ndarray) -> np.ndarray:
    """Distância de Mahalanobis com localização/dispersão do MCD (Rousseeuw)."""
    scale = np.median(np.abs(feats), axis=0) + 1e-12  # só condicionamento numérico; MCD é afim-equivariante
    f = feats / scale
    try:
        return np.sqrt(MinCovDet(random_state=0).fit(f).mahalanobis(f))
    except ValueError:
        return np.full(len(feats), np.nan)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(dim=1)
    return (0.5 * (kl(p, m) + kl(q, m))).mean().item()


class SignalExtractor:
    """Calcula todos os sinais de um round e mantém o histórico entre rounds."""

    def __init__(self, model_ctor, server_holdout, device, num_malicious_assumed: int,
                 batch_size: int = 32, lr: float = 0.01, fltrust_epochs: int = 5, ema: float = 0.5,
                 cusum_k: float = 1.0):
        self.model_ctor = model_ctor
        self.device = device
        self.f = num_malicious_assumed
        self.batch_size, self.lr, self.fltrust_epochs, self.ema = batch_size, lr, fltrust_epochs, ema
        self.holdout = server_holdout
        self.hx = torch.stack([server_holdout[i][0] for i in range(len(server_holdout))]).to(device)
        self.hy = torch.tensor([server_holdout[i][1] for i in range(len(server_holdout))]).to(device)
        self.fc1 = layer_slice(model_ctor(), "fc1")
        self.prev = None          # deltas do round anterior [N, D]
        self.cum = None           # soma histórica dos deltas por cliente [N, D]
        self.ema_score = None     # EMA do z-score de l2_median
        self.cusum_k = cusum_k
        self.cusum = {}           # estatística CUSUM por sinal-base [N]
        self.num_classes = int(self.hy.max().item()) + 1
        self.class_counts = torch.bincount(self.hy, minlength=self.num_classes).clamp_min(1).float()

    @torch.no_grad()
    def _holdout_eval(self, vector: torch.Tensor) -> dict:
        """Perda, acurácia, perda por classe e probabilidades do modelo `vector` no holdout."""
        model = self.model_ctor().to(self.device)
        load_vector(model, vector.to(self.device))
        model.eval()
        out = model(self.hx)
        ce = F.cross_entropy(out, self.hy, reduction="none")
        class_loss = torch.zeros(self.num_classes, device=ce.device).index_add_(0, self.hy, ce)
        return {"loss": ce.mean().item(), "acc": (out.argmax(1) == self.hy).float().mean().item(),
                "class_loss": (class_loss / self.class_counts.to(ce.device)).cpu(),
                "probs": out.softmax(dim=1).cpu()}

    def _cusum(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """CUSUM bilateral sobre |z robusto| do sinal-base: S_t = max(0, S_{t-1} + |z_t| - k)."""
        prev = self.cusum.get(name, torch.zeros_like(value))
        self.cusum[name] = (prev + robust_z(value).abs() - self.cusum_k).clamp_min(0)
        return self.cusum[name].clone()

    def compute(self, deltas: torch.Tensor, global_vector: torch.Tensor) -> dict[str, np.ndarray]:
        n = deltas.shape[0]
        s = {}

        # --- geometria (features do RAGNAR: diferença para a média do round)
        mean = deltas.mean(dim=0)
        diff = deltas - mean
        s["l1_mean"] = diff.abs().sum(dim=1)
        s["l2_mean"] = diff.norm(dim=1)
        s["l3_mean"] = diff.abs().pow(3).sum(dim=1).pow(1 / 3)
        s["cos_mean"] = _cos_to(deltas, mean)
        s["cos_global"] = _cos_to(deltas, global_vector)

        # --- geometria robusta
        median = deltas.median(dim=0).values
        s["norm"] = deltas.norm(dim=1)
        s["l2_median"] = (deltas - median).norm(dim=1)
        s["cos_median"] = _cos_to(deltas, median)

        feats = torch.stack([s["l1_mean"], s["l2_mean"], s["cos_mean"]], dim=1).double()
        centered = feats - feats.mean(dim=0)
        cov = centered.T @ centered / max(n - 1, 1) + 1e-9 * torch.eye(3, dtype=feats.dtype)
        s["mahalanobis"] = ((centered @ torch.linalg.pinv(cov)) * centered).sum(dim=1).sqrt().float()

        dist_sq = torch.cdist(deltas, deltas).pow(2)
        k = max(1, n - self.f - 2)
        s["krum_score"] = dist_sq.sort(dim=1).values[:, 1:k + 1].sum(dim=1)

        # --- conluio
        pair_cos = _pairwise_cos(deltas)
        s["max_cos_peer"] = _max_off_diag(pair_cos)

        # --- camada crítica (fc1)
        d_fc1 = deltas[:, self.fc1]
        med_fc1 = d_fc1.median(dim=0).values
        s["fc1_l2_median"] = (d_fc1 - med_fc1).norm(dim=1)
        s["fc1_cos_median"] = _cos_to(d_fc1, med_fc1)

        # --- referência no servidor (holdout limpo)
        base = self._holdout_eval(global_vector)
        evals = [self._holdout_eval(global_vector + deltas[i]) for i in range(n)]
        s["holdout_loss_gain"] = torch.tensor([e["loss"] - base["loss"] for e in evals])
        s["holdout_acc"] = torch.tensor([e["acc"] for e in evals])
        server_delta, _ = local_train(self.model_ctor, global_vector, self.holdout, self.device,
                                      self.fltrust_epochs, self.batch_size, self.lr)
        s["fltrust_cos"] = _cos_to(deltas, server_delta)

        # --- histórico
        nan = torch.full((n,), float("nan"))
        s["hist_cos_self"] = F.cosine_similarity(deltas, self.prev, dim=1) if self.prev is not None else nan
        self.cum = deltas.clone() if self.cum is None else self.cum + deltas
        s["hist_max_cos_cum"] = _max_off_diag(_pairwise_cos(self.cum)) if self.prev is not None else nan
        z = _zscore(s["l2_median"])
        self.ema_score = z if self.ema_score is None else self.ema * self.ema_score + (1 - self.ema) * z
        s["hist_ema_l2_median"] = self.ema_score.clone() if self.prev is not None else nan
        self.prev = deltas.clone()

        # --- v2: referência robusta (mediana geométrica)
        geomed = geometric_median(deltas)
        s["l2_geomed"] = (deltas - geomed).norm(dim=1)
        s["cos_geomed"] = _cos_to(deltas, geomed)

        # --- v2: dispersão robusta (MCD) sobre as mesmas features do Mahalanobis clássico
        s["mcd_mahalanobis"] = torch.tensor(mcd_mahalanobis(feats.numpy()), dtype=torch.float32)

        # --- v2: espectral e invariante à escala (referência: mediana coordenada a coordenada)
        s["spectral_score"] = spectral_scores(deltas, median)
        med_sign = torch.sign(median)
        mask = med_sign != 0
        s["sign_agreement"] = (torch.sign(deltas[:, mask]) == med_sign[mask]).float().mean(dim=1)

        # --- v2: comportamento por classe no holdout
        class_gain = torch.stack([e["class_loss"] - base["class_loss"] for e in evals])  # [N, C]
        s["holdout_max_class_gain"] = class_gain.max(dim=1).values
        s["holdout_class_excess"] = (class_gain - class_gain.median(dim=0).values).max(dim=1).values
        s["holdout_js"] = torch.tensor([_js_divergence(e["probs"], base["probs"]) for e in evals])

        # --- v2: acumulação sequencial (CUSUM)
        s["cusum_cos_median"] = self._cusum("cos_median", s["cos_median"])
        s["cusum_l2_median"] = self._cusum("l2_median", s["l2_median"])

        return {name: s[name].detach().cpu().numpy().astype(float) for name in SIGNALS}
