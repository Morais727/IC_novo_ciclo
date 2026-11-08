"""Estratégia Flower: FedAvg + injeção dos ataques de modelo + RAGNAR do CoUrb.

Baseada em CoUrb-2024/servidor_principal.py::Timming.aggregate_fit (features, filtro a partir
do round 2, agregação ponderada dos não excluídos, mantém o último modelo se todos forem
excluídos), com avaliação centralizada (MA no teste completo do CIFAR-10 e BA do backdoor)
no mesmo formato de resultados do fl-robustness-lab.
"""
import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate

from attacks import ATTACK_PARAMS, CRAFT, MODEL_ATTACKS, add_trigger, flatten, unflatten
from ragnar_courb import THRESHOLD, courb_features, predict_proba


def detection_prf(true_mal: set, excluded: set) -> dict:
    tp, fp, fn = len(true_mal & excluded), len(excluded - true_mal), len(true_mal - excluded)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


class RagnarCoUrb(fl.server.strategy.FedAvg):
    def __init__(self, *, initial_weights, schedule, mode="fedavg", classifier=None, seed=42,
                 eval_model=None, test_data=None, eval_every=5, num_rounds=40, **kw):
        """schedule(round) -> (attack, set de cids maliciosos); mode: fedavg | ragnar | collect."""
        self.schedule = schedule
        self.mode = mode
        self.classifier = classifier
        self.rng = np.random.default_rng(seed + 7)
        self.last_model = [w.copy() for w in initial_weights]
        self.pgd_eps = 0.0
        self.eval_model, self.test_data = eval_model, test_data
        self.eval_every, self.num_rounds = eval_every, num_rounds
        self.collected_X, self.collected_y = [], []
        self.round_log, self.history = {}, []
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=2, min_available_clients=2,
                         initial_parameters=ndarrays_to_parameters(initial_weights),
                         on_fit_config_fn=self._fit_config,
                         evaluate_fn=self._evaluate if eval_model is not None else None, **kw)

    def _fit_config(self, rnd: int) -> dict:
        attack, mal = self.schedule(rnd)
        return {"server_round": rnd, "attack": attack, "malicious": ",".join(map(str, sorted(mal))),
                "pgd_eps": float(self.pgd_eps)}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        attack, mal = self.schedule(server_round)
        cids = [int(r.metrics["cid"]) for _, r in results]
        weights = [parameters_to_ndarrays(r.parameters) for _, r in results]
        nums = [r.num_examples for _, r in results]
        last_flat = flatten(self.last_model)

        benign_idx = [i for i, c in enumerate(cids) if c not in mal]
        benign_deltas = np.stack([flatten(weights[i]) - last_flat for i in benign_idx])
        self.pgd_eps = float(np.median(np.linalg.norm(benign_deltas, axis=1)))  # raio do PGD no próximo round

        if attack in MODEL_ATTACKS and mal:
            mv = CRAFT[attack](benign_deltas, ATTACK_PARAMS[attack], self.rng)
            forged = unflatten(last_flat + mv, self.last_model)
            weights = [forged if c in mal else w for c, w in zip(cids, weights)]

        feats = np.array([courb_features(w, self.last_model) for w in weights])
        labels = np.array([int(c in mal) for c in cids])
        excluded, info = set(), {}
        if self.mode == "collect":
            self.collected_X.append(feats)
            self.collected_y.append(labels)
        elif self.mode == "ragnar" and server_round > 1:
            prob, oor = predict_proba(self.classifier, feats)
            excluded = {c for c, p in zip(cids, prob) if p > THRESHOLD}
            info = {"prob_max": float(prob.max()), "features_out_of_range": oor}

        kept = [(w, n) for c, w, n in zip(cids, weights, nums) if c not in excluded]
        if kept:
            self.last_model = aggregate(kept)
        det = detection_prf(set(mal), excluded) if mal else None
        self.round_log[server_round] = {"detection": det, "excluded": sorted(excluded), **info}
        return ndarrays_to_parameters(self.last_model), {}

    def _evaluate(self, server_round, parameters, config):
        if server_round == 0 or not (server_round % self.eval_every == 0 or server_round == self.num_rounds):
            return None
        x, y = self.test_data
        self.eval_model.set_weights(parameters)
        pred = self.eval_model.predict(x, batch_size=512, verbose=0).argmax(1)
        ma = 100.0 * float((pred == y).mean())
        attack, _ = self.schedule(server_round)
        ba = None
        if attack.startswith("backdoor"):
            tgt = ATTACK_PARAMS[attack]["target_label"]
            keep = y != tgt
            pb = self.eval_model.predict(add_trigger(x[keep]), batch_size=512, verbose=0).argmax(1)
            ba = 100.0 * float((pb == tgt).mean())
        log = self.round_log.get(server_round, {})
        self.history.append({"round": server_round, "main_accuracy": ma, "backdoor_asr": ba,
                             "detection": log.get("detection"),
                             **{k: log[k] for k in ("prob_max", "features_out_of_range") if k in log}})
        msg = f"  round {server_round}: MA={ma:.2f}%" + (f" BA={ba:.2f}%" if ba is not None else "")
        if log.get("detection"):
            d = log["detection"]
            msg += f" det(P/R/F1)={d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}"
        print(msg, flush=True)
        return 0.0, {"accuracy": ma}
