"""Treino local de um cliente FL: retorna o vetor de atualização (delta = w_local - w_global)."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.federated.utils import flatten, load_vector


def local_train(model_ctor, global_vector: torch.Tensor, dataset, device: torch.device,
                 epochs: int, batch_size: int, lr: float,
                 project_radius: float | None = None, optimizer: str = "sgd",
                 grad_clip: float | None = None) -> tuple[torch.Tensor, int]:
    """Treina uma cópia do modelo global sobre `dataset`.

    Se `project_radius` for dado, após cada passo de SGD os pesos locais são
    projetados na bola L2 de raio `project_radius` centrada no modelo global
    (PGD, Wang et al. 2020 — usado pelo ataque `backdoor_pgd`).

    Retorna (delta_vector, n_amostras).
    """
    model = model_ctor().to(device)
    global_dev = global_vector.to(device)
    load_vector(model, global_dev)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    n_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if project_radius is not None:
                with torch.no_grad():
                    diff = flatten(model) - global_dev
                    norm = diff.norm()
                    if norm > project_radius:
                        load_vector(model, global_dev + diff * (project_radius / (norm + 1e-12)))
            if _ == 0:
                n_samples += x.size(0)

    local_vector = flatten(model)
    delta = (local_vector - global_dev).detach().cpu()
    return delta, max(n_samples, 1)
