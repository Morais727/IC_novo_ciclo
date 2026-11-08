"""Ataques de backdoor (targeted): gatilho pixel-pattern + rótulo alvo.

`backdoor_pgd` é uma simplificação operacional do conceito "Edge-case PGD"
discutido na literatura: em vez de minerar amostras naturais raras
(edge-case), usa o mesmo gatilho pixel-pattern combinado com PGD + model
replacement (Wang et al. 2020; Bagdasaryan et al.):

  1. o raio final `max_norm` (ε) vem de `attack_params["max_norm"]` ou, por
     padrão, da mediana das normas dos deltas benignos do round (adversário
     onisciente, mesma modelagem dos ataques de modelo) vezes
     `attack_params["norm_multiplier"]` (padrão 1.0);
  2. o treino local malicioso é projetado a cada passo na bola L2 de raio
     ε / scale em torno do modelo global;
  3. o delta resultante é escalado por `scale` (compensa a diluição do
     FedAvg) — a norma final fica ≤ ε, indistinguível em magnitude de um
     cliente benigno típico.

Está documentado como simplificação no documento de cenário (não reproduz a
mineração de edge-cases do artigo original).
"""
import torch
from src.data.datasets import BackdoorDataset


def pgd_norm_bound(benign_deltas: torch.Tensor, params: dict) -> float:
    """Raio final ε do ataque `backdoor_pgd` (ver docstring do módulo)."""
    if params.get("max_norm") is not None:
        return float(params["max_norm"])
    median_norm = benign_deltas.norm(dim=1).median().item()
    return median_norm * float(params.get("norm_multiplier", 1.0))


def backdoor_pixel(base_dataset, params: dict):
    target_label = params.get("target_label", 0)
    poison_fraction = params.get("poison_fraction", 1.0)
    seed = params.get("seed", 0)
    return BackdoorDataset(base_dataset, target_label=target_label,
                            poison_fraction=poison_fraction, seed=seed)


def constrain_and_scale(delta: torch.Tensor, params: dict) -> torch.Tensor:
    scale = params.get("scale", 1.0)
    max_norm = params.get("max_norm", None)
    scaled = delta * scale
    if max_norm is not None:
        norm = scaled.norm()
        if norm > max_norm:
            scaled = scaled * (max_norm / (norm + 1e-12))
    return scaled


BACKDOOR_DATA_ATTACKS = {
    "backdoor_pixel": backdoor_pixel,
    "backdoor_pgd": backdoor_pixel,  # mesmo gatilho; o post-processing é que difere
}

POST_PROCESS_ATTACKS = {
    "constrain_and_scale": constrain_and_scale,
}
