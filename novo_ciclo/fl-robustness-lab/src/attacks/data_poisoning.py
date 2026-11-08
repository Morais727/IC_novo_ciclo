"""Envenenamento de dados (canal: dados do cliente, antes do treino local)."""
from src.data.datasets import LabelFlipDataset


def label_flip(base_dataset, params: dict):
    mode = params.get("mode", "fixed")     # "fixed" (source->target) ou "random"
    source = params.get("source", None)
    target = params.get("target", None)
    seed = params.get("seed", 0)
    return LabelFlipDataset(base_dataset, mode=mode, source=source, target=target, seed=seed)


DATA_POISONING_ATTACKS = {
    "label_flip": label_flip,
}
