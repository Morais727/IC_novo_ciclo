"""Métricas de avaliação: MA% (Main Accuracy), BA%/ASR (Backdoor Success Rate),
e métricas de detecção (precisão/revocação de clientes maliciosos filtrados)."""
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy(model, dataset, device, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def backdoor_success_rate(model, backdoor_eval_dataset, target_label: int, device, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(backdoor_eval_dataset, batch_size=batch_size, shuffle=False)
    hit, total = 0, 0
    for x, _ in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1)
        hit += (pred == target_label).sum().item()
        total += x.size(0)
    return 100.0 * hit / max(total, 1)


def detection_prf(true_malicious: set, excluded_indices) -> dict:
    excluded = set(excluded_indices)
    all_ids = true_malicious | (excluded - true_malicious)
    tp = len(true_malicious & excluded)
    fp = len(excluded - true_malicious)
    fn = len(true_malicious - excluded)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
