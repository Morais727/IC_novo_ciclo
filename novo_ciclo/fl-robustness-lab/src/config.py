"""Definições de configuração de cenário de avaliação (dataclasses)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    name: str = "unnamed_scenario"

    # Modelo
    model: str = "smallcnn"

    # Dados / não-IID
    dataset: str = "cifar10"
    num_clients: int = 20
    dirichlet_alpha: float = 0.5          # menor = mais label skew (não-IID severo)
    server_holdout_size: int = 100        # amostras limpas no servidor (BOBA/RAGNAR bootstrap)

    # Adversário
    num_malicious: int = 4
    attack: str = "none"                  # ver src/attacks/registry.py
    attack_params: dict = field(default_factory=dict)
    backdoor_target_label: int = 0

    # Defesa / agregação
    defense: str = "fedavg"               # ver src/defenses/registry.py
    defense_params: dict = field(default_factory=dict)

    # Treino federado
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    lr_schedule: Optional[list] = None    # None = lr fixo; senão [[ate_round, lr], ...] (1ª faixa cujo r <= ate_round)
    optimizer: str = "sgd"                # "sgd" (momentum 0.9) | "adam" (usado no ataca_FEDML, lr=1e-3)
    grad_clip: Optional[float] = None     # None = sem clipping; senão norma L2 máx. do gradiente por passo (estabiliza VGG19-GN)
    client_samples_per_round: Optional[int] = None  # None = todos os clientes

    # Execução
    seed: int = 42
    device: str = "auto"                  # "auto" | "cuda" | "cpu"
    eval_every: int = 5

    # Modo rápido de validação (subamostra o dataset e reduz rounds/epochs)
    quick: bool = False
    quick_train_per_client: int = 100
    quick_test_size: int = 500
