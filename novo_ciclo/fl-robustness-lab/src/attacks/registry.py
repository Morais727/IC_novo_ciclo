"""Registro unificado de ataques: mapeia nome -> (kind, funções relevantes).

kind:
  "none"            -> cliente benigno.
  "data"            -> envenena o dataset local antes do treino (label_flip, backdoor_pixel).
  "data+postprocess"-> envenena o dataset E pós-processa o delta resultante (backdoor_pgd).
  "model"           -> ataque "onisciente": ignora o treino local e sintetiza o delta
                        diretamente a partir dos deltas benignos do round (Blades/BOBA-style).
  "layerwise"       -> treina local (limpo e/ou envenenado) e monta o delta camada a camada (LP/LF).
  "layerwise_full"  -> LP completo: um delta criado em conjunto pelos atacantes e enviado por todos.

eval: "trigger" (ASR com o gatilho 4x4 no teste) ou "edge" (ASR nas amostras edge-case do teste).
Ataques recentes (mimic, dba, lp, lf, edge_case_pgd): ver src/attacks/recent.py.
"""
from src.attacks.data_poisoning import DATA_POISONING_ATTACKS
from src.attacks.backdoor import BACKDOOR_DATA_ATTACKS, POST_PROCESS_ATTACKS
from src.attacks.model_poisoning import MODEL_POISONING_ATTACKS
from src.attacks import recent

ATTACK_REGISTRY = {
    "none": {"kind": "none"},

    "label_flip": {"kind": "data", "dataset_fn": DATA_POISONING_ATTACKS["label_flip"]},

    "backdoor_pixel": {"kind": "data", "dataset_fn": BACKDOOR_DATA_ATTACKS["backdoor_pixel"], "eval": "trigger"},
    "backdoor_pgd": {
        "kind": "data+postprocess",
        "dataset_fn": BACKDOOR_DATA_ATTACKS["backdoor_pgd"],
        "post_fn": POST_PROCESS_ATTACKS["constrain_and_scale"],
        "eval": "trigger",
    },

    "gaussian_noise": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["gaussian_noise"]},
    "sign_flip": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["sign_flip"]},
    "ipm": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["ipm"]},
    "lie": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["lie"]},
    "minmax": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["minmax"]},
    "minsum": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["minsum"]},
    "byzmean": {"kind": "model", "craft_fn": MODEL_POISONING_ATTACKS["byzmean"]},
    # 16/11 (fase 19): ataques usados por FLAME, AlignIns, FLD e FedSurrogate que faltavam no laboratório
    # Scaling / constrain-and-scale (Bagdasaryan et al.): backdoor com gatilho e o delta multiplicado por `scale`,
    # SEM limite de norma (AlignIns usa fator 2).
    "backdoor_scaling": {"kind": "data+postprocess", "dataset_fn": BACKDOOR_DATA_ATTACKS["backdoor_pixel"],
                         "post_fn": POST_PROCESS_ATTACKS["constrain_and_scale"], "no_bound": True, "eval": "trigger"},
    # Neurotoxin (Zhang et al., ICML 2022): o atacante só escreve nas coordenadas que os honestos MENOS atualizam
    # (zera o top-k% por magnitude da média benigna da rodada), para o backdoor durar. Aproximação: projeção feita
    # uma vez depois do treino local (o original projeta a cada passo).
    # FLIP direcionado (Bhagoji et al. 2019; RFLDefense, FLD): conjunto auxiliar da classe `source` com rótulo alvo,
    # misturado a dados limpos locais (mesmo encaixe do edge-case, sem PGD e sem escala); ASR na classe `source` do teste.
    "flip_targeted": {"kind": "data", "dataset_fn": recent.edge_case_dataset, "eval": "edge", "edge_case": True,
                      "flip_targeted": True},
    "neurotoxin": {"kind": "data+postprocess", "dataset_fn": BACKDOOR_DATA_ATTACKS["backdoor_pixel"],
                   "post_fn": recent.neurotoxin_mask, "no_bound": True, "eval": "trigger"},

    # --- ataques recentes (2020–2024)
    "mimic": {"kind": "model", "craft_fn": recent.mimic},
    "dba": {"kind": "data", "dataset_fn": recent.dba_dataset, "eval": "trigger"},
    "lp": {"kind": "layerwise", "mode": "lp", "eval": "trigger"},
    "lf": {"kind": "layerwise", "mode": "lf"},
    # LP completo como no código dos autores (lambda, modelo médio, controle adaptativo de camadas)
    "lp_full": {"kind": "layerwise_full", "eval": "trigger"},
    "lp_full_indiv": {"kind": "layerwise_full", "eval": "trigger"},
    # ataques de 2026 (Fase 8A): LSA (ICC 2026) e LaySelFL (CVPR 2026), mesmo encaixe do LP completo
    "lsa": {"kind": "layerwise_full", "fn": recent.lsa_attack, "eval": "trigger"},
    "laysel": {"kind": "layerwise_full", "fn": recent.laysel_attack, "eval": "trigger"},
    "edge_case_pgd": {
        "kind": "data+postprocess",
        "dataset_fn": recent.edge_case_dataset,
        "post_fn": POST_PROCESS_ATTACKS["constrain_and_scale"],
        "eval": "edge",
        "edge_case": True,
    },
}


def list_attacks():
    return list(ATTACK_REGISTRY.keys())
