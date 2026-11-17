"""Orquestração da simulação de FL: particionamento, rounds, ataque, defesa, avaliação."""
import math
import time

import numpy as np
import torch
from torch.utils.data import Subset

from src.config import ScenarioConfig
from src.models.cnn import build_model
from src.data.datasets import load_dataset, BackdoorEvalDataset, quick_subset, DATASET_CLASSES
from src.data.partition import dirichlet_partition
from src.federated.client import local_train
from src.federated.utils import flatten, load_vector
from src.attacks.registry import ATTACK_REGISTRY
from src.attacks.backdoor import pgd_norm_bound
from src.attacks.recent import layerwise_attack, lp_full_attack, select_edge_cases
from src.defenses.registry import DEFENSE_REGISTRY
from src.defenses.ragnar import extract_features, train_classifier, train_offline
from src.defenses.ragnar_original import load_pretrained, original_features, param_slices, train_like_original
from src.metrics import accuracy, backdoor_success_rate, detection_prf

NUM_CLASSES = 10

# Pool de ataques simulados no bootstrap do RAGNAR, com os parâmetros usados.
RAGNAR_BOOTSTRAP_POOL = {
    "gaussian_noise": {},
    "sign_flip": {},
    "ipm": {},
    "lie": {},
    "minmax": {},
    "minsum": {},
    "label_flip": {},
    "backdoor_pixel": {},
    "backdoor_pgd": {"scale": 2.0},
}

# Ataques da mesma família são excluídos juntos do bootstrap (leave-one-family-out).
ATTACK_FAMILIES = {
    "backdoor_pixel": "backdoor",
    "backdoor_pgd": "backdoor",
    "minmax": "minmax",
    "minsum": "minmax",
    "dba": "backdoor",
    "lp": "backdoor",
    "edge_case_pgd": "backdoor",
}


def attack_family(name: str) -> str:
    return ATTACK_FAMILIES.get(name, name)


def ragnar_bootstrap_pool(eval_attack: str, exclude_eval_attack: bool = True,
                          only_eval_attack: bool = False) -> list[str]:
    """Ataques usados no bootstrap do RAGNAR. Por padrão exclui a família do
    ataque avaliado, para que a detecção meça generalização a um ataque não
    visto no treino do classificador (e não memorização do próprio ataque).

    `only_eval_attack=True` restringe o bootstrap ao próprio ataque avaliado —
    garante que ele foi visto (com o pool inteiro, os poucos rounds de
    bootstrap sorteiam só alguns ataques e o avaliado pode não aparecer)."""
    if only_eval_attack:
        return [eval_attack]
    pool = list(RAGNAR_BOOTSTRAP_POOL)
    if exclude_eval_attack:
        pool = [a for a in pool if attack_family(a) != attack_family(eval_attack)]
    return pool


def _resolve_device(cfg: ScenarioConfig) -> torch.device:
    if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _build_data(cfg: ScenarioConfig, rng: np.random.Generator):
    train_full, test_full = load_dataset(cfg.dataset)
    labels = np.array(train_full.targets)

    nc = DATASET_CLASSES.get(cfg.dataset, NUM_CLASSES)
    per_class = max(1, cfg.server_holdout_size // nc)
    holdout_idx = []
    for c in range(nc):
        cls_idx = np.where(labels == c)[0]
        chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        holdout_idx.extend(chosen.tolist())
    holdout_idx = np.array(holdout_idx)
    remaining_idx = np.setdiff1d(np.arange(len(train_full)), holdout_idx)

    cfg._edge = None
    if ATTACK_REGISTRY.get(cfg.attack, {}).get("edge_case"):
        # edge-case (Wang et al. 2020): amostras raras só dos atacantes, fora dos dados dos honestos
        p = cfg.attack_params
        if p.get("edge_source") == "southwest":   # imagens externas do artigo (nada é retirado do CIFAR-10)
            from src.attacks.recent import SouthwestDataset
            cfg._edge = {"train": SouthwestDataset("train"), "test": SouthwestDataset("test")}
    if ATTACK_REGISTRY.get(cfg.attack, {}).get("flip_targeted"):
        # 16/11 (fase 19): troca de rótulo DIRECIONADA (FLIP; Bhagoji et al. 2019, como avaliada no RFLDefense e no
        # FLD): os atacantes têm um conjunto auxiliar de `aux_n` imagens da classe `source` rotuladas como o alvo,
        # sem gatilho; as mesmas imagens podem existir, com o rótulo certo, nos dados dos honestos.
        # Sucesso = imagem de teste da classe `source` classificada como o alvo.
        p = cfg.attack_params
        src_c = p.get("source", 0)
        tr_src = np.setdiff1d(np.where(np.array(train_full.targets) == src_c)[0], holdout_idx)
        aux = np.random.default_rng(cfg.seed).choice(tr_src, size=min(int(p.get("aux_n", 500)), len(tr_src)), replace=False)
        te_src = np.where(np.array(test_full.targets) == src_c)[0]
        cfg._edge = {"train": Subset(train_full, aux.tolist()), "test": Subset(test_full, te_src.tolist())}
    if cfg._edge is None and ATTACK_REGISTRY.get(cfg.attack, {}).get("edge_case"):
        p = cfg.attack_params
        edge_tr, edge_te = select_edge_cases(train_full, test_full, p.get("source", 0), p.get("edge_fraction", 0.05))
        edge_tr = np.setdiff1d(edge_tr, holdout_idx)
        remaining_idx = np.setdiff1d(remaining_idx, edge_tr)
        cfg._edge = {"train": Subset(train_full, edge_tr.tolist()), "test": Subset(test_full, edge_te.tolist())}

    if cfg.quick:
        target_n = min(len(remaining_idx), cfg.num_clients * cfg.quick_train_per_client)
        remaining_idx = rng.choice(remaining_idx, size=target_n, replace=False)

    remaining_labels = labels[remaining_idx]
    partitions = dirichlet_partition(remaining_labels, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
    client_data = [Subset(train_full, remaining_idx[part].tolist()) for part in partitions]
    server_holdout = Subset(train_full, holdout_idx.tolist())

    test_eval = quick_subset(test_full, cfg.quick_test_size, seed=cfg.seed) if cfg.quick else test_full
    return client_data, server_holdout, test_full, test_eval


def _train_round(model_ctor, global_vector, client_data, device, cfg: ScenarioConfig,
                 malicious_ids: set, attack_spec: dict, attack_params: dict):
    """Treina todos os clientes de um round e aplica o ataque.

    Os benignos treinam primeiro: ataques oniscientes (de modelo e o raio do
    `backdoor_pgd`) são sintetizados a partir dos deltas benignos do round.
    Retorna (deltas [N, D], weights [N]).
    """
    n = len(client_data)   # = cfg.num_clients, ou os participantes da rodada (client_samples_per_round)
    deltas, weights = [None] * n, [0] * n
    benign_ids = [c for c in range(n) if c not in malicious_ids]

    def train(dataset, project_radius=None):
        return local_train(model_ctor, global_vector, dataset, device,
                           cfg.local_epochs, cfg.batch_size, cfg.lr, project_radius=project_radius,
                           optimizer=cfg.optimizer, grad_clip=getattr(cfg, "grad_clip", None))

    for cid in benign_ids:
        deltas[cid], weights[cid] = train(client_data[cid])

    kind = attack_spec["kind"]
    if malicious_ids:
        benign_deltas = torch.stack([deltas[c] for c in benign_ids])
        if kind == "layerwise_full":
            mids = sorted(malicious_ids)
            attack_params["_hp"] = dict(epochs=cfg.local_epochs, batch_size=cfg.batch_size, lr=cfg.lr)
            crafted = attack_spec.get("fn", lp_full_attack)(lambda d: train(d), model_ctor, global_vector,
                                     [client_data[m] for m in mids], benign_deltas, device, attack_params)
            counts = attack_params.setdefault("_layers_count", {})
            for g in attack_params.get("_bc_layers", []):
                counts[g] = counts.get(g, 0) + 1
            individual = attack_params.pop("_lp_individual", None)
            for k, mid in enumerate(mids):
                deltas[mid] = individual[k] if individual is not None else crafted.clone()
                weights[mid] = max(len(client_data[mid]), 1)
        elif kind == "model":
            malicious_vector = attack_spec["craft_fn"](benign_deltas, {**attack_params, "_n_mal": len(malicious_ids),
                                                                        "_n_total": len(malicious_ids) + len(benign_deltas)})
            for k, mid in enumerate(sorted(malicious_ids)):
                # 16/11: craft_fn pode devolver um vetor por atacante (ByzMean); senão, todos enviam o mesmo
                deltas[mid] = malicious_vector[k] if malicious_vector.dim() == 2 else malicious_vector
                weights[mid] = max(len(client_data[mid]), 1)
        else:
            radius, eps = None, None
            if kind == "data+postprocess" and not attack_spec.get("no_bound"):   # 16/11: Scaling/Neurotoxin sem limite
                eps = pgd_norm_bound(benign_deltas, attack_params)
                radius = eps / attack_params.get("scale", 1.0)
            for k, mid in enumerate(sorted(malicious_ids)):
                dataset = client_data[mid]
                if kind == "layerwise":
                    p = {**attack_params, "mode": attack_spec["mode"], "attacker_index": k}
                    deltas[mid] = layerwise_attack(lambda d: train(d), model_ctor, global_vector, dataset, device, p)
                    weights[mid] = max(len(dataset), 1)
                    counts = attack_params.setdefault("_layers_count", {})
                    for g in p.get("_bc_layers", p.get("_critical_layers", [])):
                        counts[g] = counts.get(g, 0) + 1
                    continue
                if kind in ("data", "data+postprocess"):
                    dataset = attack_spec["dataset_fn"](dataset, {**attack_params, "attacker_index": k,
                                                                  "edge_dataset": (getattr(cfg, "_edge", None) or {}).get("train")})
                delta, _ = train(dataset, project_radius=radius)
                if kind == "data+postprocess":
                    delta = attack_spec["post_fn"](delta, {**attack_params, "max_norm": eps,
                                                           "_benign_mean": benign_deltas.mean(0)})
                # peso declarado = tamanho dos dados locais do cliente (como um honesto e como nos ataques de
                # modelo), não o do conjunto envenenado — o edge-case treina num conjunto menor e misturado
                deltas[mid], weights[mid] = delta, max(len(client_data[mid]), 1)

    return torch.stack(deltas), torch.tensor(weights)


def _bootstrap_dataset(model_ctor, client_data, device, cfg: ScenarioConfig, pool: list[str],
                       n_rounds: int, feature_fn):
    """Simula `n_rounds` rounds a partir do modelo inicial, sorteando um ataque do `pool` por
    round, e devolve (X, y) com `feature_fn(deltas, global_vector)` de cada cliente."""
    rng = np.random.default_rng(cfg.seed + 999)
    global_vector = flatten(model_ctor().to(device)).cpu()
    all_feats, all_labels = [], []

    for _ in range(n_rounds):
        n_mal = cfg.num_malicious
        malicious_ids = set(rng.choice(cfg.num_clients, size=n_mal, replace=False).tolist()) if n_mal > 0 else set()
        attack_name = str(rng.choice(pool))
        params = {"target_label": cfg.backdoor_target_label, **RAGNAR_BOOTSTRAP_POOL[attack_name]}

        deltas, weights = _train_round(model_ctor, global_vector, client_data, device, cfg,
                                       malicious_ids, ATTACK_REGISTRY[attack_name], params)
        all_feats.append(feature_fn(deltas, global_vector))
        all_labels.append(np.array([1 if i in malicious_ids else 0 for i in range(cfg.num_clients)]))

        w = weights.float()
        w = w / w.sum()
        global_vector = global_vector + (deltas * w.unsqueeze(1)).sum(dim=0)

    return np.concatenate(all_feats), np.concatenate(all_labels)


def _bootstrap_ragnar(model_ctor, client_data, device, cfg: ScenarioConfig, pool: list[str],
                       n_bootstrap_rounds: int = 5, cos_ref: str = "mean_delta"):
    feature_fn = lambda d, g: extract_features(d, d.mean(dim=0, keepdim=True),
                                               g if cos_ref == "global_model" else None)
    X, y = _bootstrap_dataset(model_ctor, client_data, device, cfg, pool, n_bootstrap_rounds, feature_fn)
    if y.sum() == 0 or y.sum() == len(y):
        return None  # sem exemplos das duas classes; não é possível treinar
    return train_classifier(X, y, cfg.defense_params)


def _maybe_init(global_vector):
    """Fase 12 (13/11/2026): se FL_INIT_VECTOR aponta um .pt (vetor global salvo), o FL começa dele em vez do modelo
    aleatório. Sem a variável, nada muda."""
    import os
    f = os.environ.get("FL_INIT_VECTOR")
    if not f:
        return global_vector
    v = torch.load(f, map_location="cpu", weights_only=True)
    assert v.shape == global_vector.shape, f"FL_INIT_VECTOR {f}: forma {tuple(v.shape)} != {tuple(global_vector.shape)}"
    return v.clone()


def _maybe_ckpt(cfg, r, global_vector):
    """Fase 10 (12/11/2026): salva o vetor global ao FIM das rodadas listadas em FL_SAVE_ROUNDS (ex. "79,100,205"),
    em FL_SAVE_FINAL/<nome>_r<r>.pt. Só age com as duas variáveis de ambiente; sem elas, nada muda."""
    import os
    rs, d = os.environ.get("FL_SAVE_ROUNDS"), os.environ.get("FL_SAVE_FINAL")
    if rs and d and str(r) in rs.split(","):
        os.makedirs(d, exist_ok=True)
        torch.save(global_vector.cpu(), os.path.join(d, f"{cfg.name}_r{r}.pt"))


def run(cfg: ScenarioConfig) -> dict:
    t_start = time.time()
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n_classes = DATASET_CLASSES.get(cfg.dataset, NUM_CLASSES)
    model_ctor = lambda: build_model(cfg.model, num_classes=n_classes, dataset=cfg.dataset)
    client_data, server_holdout, test_full, test_eval = _build_data(cfg, rng)

    attack_spec = ATTACK_REGISTRY[cfg.attack]
    malicious_ids = set()
    if cfg.attack != "none" and cfg.num_malicious > 0:
        malicious_ids = set(rng.choice(cfg.num_clients, size=cfg.num_malicious, replace=False).tolist())

    backdoor_eval = None
    target_label = cfg.attack_params.get("target_label", cfg.backdoor_target_label)
    eval_kind = attack_spec.get("eval")
    if eval_kind == "trigger":
        base_test = test_eval if cfg.quick else test_full
        backdoor_eval = BackdoorEvalDataset(base_test, target_label)
        cfg.attack_params.setdefault("target_label", target_label)
    elif eval_kind == "edge":
        backdoor_eval = cfg._edge["test"]  # sucesso = amostra edge-case do teste classificada como o alvo
        cfg.attack_params.setdefault("target_label", target_label)

    global_vector = flatten(model_ctor().to(device)).cpu()
    global_vector = _maybe_init(global_vector)

    # nº típico de passos de SGD de um cliente por round (BOBA usa para as referências por classe)
    client_steps = [cfg.local_epochs * math.ceil(len(d) / cfg.batch_size) for d in client_data]

    ctx = {
        "model_ctor": model_ctor,
        "device": device,
        "num_classes": n_classes,
        "server_holdout": server_holdout,
        "global_vector": global_vector,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "ref_steps": max(1, int(np.median(client_steps))),
        "state": {},
        "round_idx": 0,
        "_malicious_ids_diag_only": list(malicious_ids),   # só para fedavg_dump (diagnóstico); defesas não leem
    }

    extra = {}
    if cfg.defense == "ragnar" and malicious_ids:
        pool = ragnar_bootstrap_pool(cfg.attack, cfg.defense_params.get("bootstrap_exclude_eval_attack", True),
                                     cfg.defense_params.get("bootstrap_only_eval_attack", False))
        if cfg.defense_params.get("offline_dataset"):
            clf = train_offline(cfg.defense_params["offline_dataset"], pool, cfg.defense_params)
        else:
            n_boot = cfg.defense_params.get("bootstrap_rounds", 2 if cfg.quick else 5)
            cos_ref = cfg.defense_params.get("cos_ref", "mean_delta")
            clf = _bootstrap_ragnar(model_ctor, client_data, device, cfg, pool,
                                    n_bootstrap_rounds=n_boot, cos_ref=cos_ref)
        ctx["state"]["ragnar_classifier"] = clf
        extra["ragnar_bootstrap_pool"] = pool

    if cfg.defense == "ragnar_original" and malicious_ids:
        if cfg.defense_params.get("classifier", "pretrained") == "pretrained":
            ctx["state"]["ragnar_original"] = load_pretrained(cfg.defense_params)
        else:  # "bootstrap5": mesmo pipeline do original, treinado com rounds iniciais simulados aqui
            pool = ragnar_bootstrap_pool(cfg.attack, cfg.defense_params.get("bootstrap_exclude_eval_attack", True),
                                         cfg.defense_params.get("bootstrap_only_eval_attack", False))
            slices = param_slices(model_ctor())
            X, y = _bootstrap_dataset(model_ctor, client_data, device, cfg, pool,
                                      cfg.defense_params.get("bootstrap_rounds", 2 if cfg.quick else 5),
                                      lambda d, g: original_features(d, g, slices))
            ctx["state"]["ragnar_original"] = train_like_original(X, y)
            extra["ragnar_bootstrap_pool"] = pool

    history = []
    # aceitação por round (todos os rounds, não só os de avaliação): fração de honestos e de
    # maliciosos que a defesa deixou entrar na agregação
    accepted = {"benign": 0, "benign_total": 0, "malicious": 0, "malicious_total": 0}
    kept_params = {"benign": [], "malicious": []}   # defesas por tensor (layer_guard): fração de parâmetros aceitos
    t_rounds = time.time()
    _base_lr = cfg.lr
    # 13/11/2026 (experimento 1.7, escala): participação parcial — K clientes sorteados por rodada, com gerador
    # próprio (não consome o `rng` do cenário, então os cenários com todos os clientes ficam idênticos).
    K = getattr(cfg, "client_samples_per_round", None)
    part_rng = np.random.default_rng(cfg.seed + 7919)
    for r in range(1, cfg.rounds + 1):
        ctx["round_idx"] = r
        ctx["global_vector"] = global_vector

        # agendamento de LR por rodada (cfg.lr_schedule = [[ate_round, lr], ...]; None = lr fixo).
        # Ex. LaySelFL: 0,01 ate a rodada 50 e 0,001 depois. Preserva o lr base.
        sched = getattr(cfg, "lr_schedule", None)
        if sched:
            cfg.lr = next((lr for until, lr in sched if r <= until), _base_lr)
        # janela de ataque: start_round <= r <= end_round (default: do start ate o fim). Fora dela, atacantes honestos.
        start_r = cfg.attack_params.get("start_round", 1)
        end_r = cfg.attack_params.get("end_round", cfg.rounds)
        if K and K < cfg.num_clients:
            sel = sorted(part_rng.choice(cfg.num_clients, size=K, replace=False).tolist())
            data_r = [client_data[c] for c in sel]
            mal_r = {i for i, c in enumerate(sel) if c in malicious_ids}   # índices LOCAIS da rodada
            ctx["_malicious_ids_diag_only"] = sorted(mal_r)
        else:
            data_r, mal_r = client_data, malicious_ids
        active = mal_r if start_r <= r <= end_r else set()
        deltas, weights_t = _train_round(model_ctor, global_vector, data_r, device, cfg,
                                         active, attack_spec, cfg.attack_params)

        defense_fn = DEFENSE_REGISTRY[cfg.defense]
        try:
            agg_delta, info = defense_fn(deltas, weights_t, ctx, cfg.defense_params)
        except torch.OutOfMemoryError:   # 15/11: GPU compartilhada cheia (2 adaptativos caíram assim) -> agrega na CPU
            torch.cuda.empty_cache()
            print(f"[r{r}] defesa sem memória na GPU -> refeita na CPU", flush=True)
            agg_delta, info = defense_fn(deltas.cpu(), weights_t.cpu(), ctx, cfg.defense_params)
        global_vector = global_vector + agg_delta.to(global_vector.device)
        _maybe_ckpt(cfg, r, global_vector)
        excluded = set(info.get("excluded_indices", []))
        for c in range(deltas.shape[0]):
            k = "malicious" if c in mal_r else "benign"
            accepted[k + "_total"] += 1
            accepted[k] += c not in excluded
            if "kept_fraction" in info:
                kept_params[k].append(info["kept_fraction"][c])

        if r % cfg.eval_every == 0 or r == cfg.rounds:
            model = model_ctor().to(device)
            load_vector(model, global_vector.to(device))
            ma = accuracy(model, test_eval, device)
            ba = backdoor_success_rate(model, backdoor_eval, target_label, device) if backdoor_eval is not None else None
            det = detection_prf(mal_r, info.get("excluded_indices", [])) if mal_r else None
            entry = {"round": r, "main_accuracy": ma, "backdoor_asr": ba, "detection": det,
                     "elapsed_s": time.time() - t_start,
                     **{k: info[k] for k in ("prob_max", "features_out_of_range") if k in info}}
            history.append(entry)
            msg = f"[{cfg.name}] round {r}/{cfg.rounds} MA={ma:.2f}%"
            if ba is not None:
                msg += f" BA={ba:.2f}%"
            if det is not None:
                msg += f" det(P/R/F1)={det['precision']:.2f}/{det['recall']:.2f}/{det['f1']:.2f}"
            msg += f" ({time.time() - t_start:.0f}s)"
            print(msg, flush=True)   # flush: progresso visível no log mesmo sob nohup/redirecionamento (sem depender de -u)

    import os
    if os.environ.get("FL_SAVE_FINAL"):   # vetor do modelo global final, para reparos pós-treino fora do circuito
        os.makedirs(os.environ["FL_SAVE_FINAL"], exist_ok=True)
        torch.save(global_vector.cpu(), os.path.join(os.environ["FL_SAVE_FINAL"], f"{cfg.name}.pt"))

    return {
        "scenario": cfg.name,
        "attack": cfg.attack,
        "defense": cfg.defense,
        "attack_params": cfg.attack_params,
        "defense_params": cfg.defense_params,
        "num_clients": cfg.num_clients,
        "num_malicious": cfg.num_malicious,
        "malicious_ids": sorted(malicious_ids),
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "rounds": cfg.rounds,
        "seed": cfg.seed,
        "quick": cfg.quick,
        "device": str(device),
        "wall_time_s": time.time() - t_start,
        "sec_per_round": (time.time() - t_rounds) / max(cfg.rounds, 1),
        "acceptance": {"benign": accepted["benign"] / max(accepted["benign_total"], 1),
                       "malicious": accepted["malicious"] / max(accepted["malicious_total"], 1)
                       if accepted["malicious_total"] else None,
                       **({"benign_params": float(np.mean(kept_params["benign"])),
                           "malicious_params": float(np.mean(kept_params["malicious"])) if kept_params["malicious"] else None}
                          if kept_params["benign"] else {})},
        **extra,
        **({"diag_log": ctx["state"]["diag_log"]} if "diag_log" in ctx["state"] else {}),
        **({"scan_log": ctx["state"]["scan_log"], "flagged": sorted(ctx["state"].get("flagged", []))} if "scan_log" in ctx["state"] else {}),
        "history": history,
    }
