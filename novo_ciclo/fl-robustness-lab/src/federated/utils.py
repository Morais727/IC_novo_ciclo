"""Utilidades de (des)vetorização de parâmetros do modelo."""
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def flatten(model: torch.nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def load_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    # vector_to_parameters faz os parâmetros virarem *views* de `vector`; sem o
    # clone, treinar o modelo alteraria o vetor original (ex: o modelo global em CPU).
    vector_to_parameters(vector.detach().clone(), model.parameters())


def clone_model(model: torch.nn.Module):
    import copy
    return copy.deepcopy(model)
