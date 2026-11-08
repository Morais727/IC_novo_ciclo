"""fedavg_dump (diagnóstico, fase 3): agrega como o FedAvg; acumula a soma dos deltas de cada cliente ao longo dos
rounds e, nos rounds params['rounds'], salva em params['dir']/{tag}_r{round}.pt o vetor global, os deltas do round,
a soma acumulada por cliente (rounds 1..r) e quais são atacantes."""
import os
import torch
from src.defenses.aggregators import fedavg


def fedavg_dump(deltas, weights, ctx, params):
    st = ctx["state"]
    st["cum"] = deltas.cpu().clone() if "cum" not in st else st["cum"] + deltas.cpu()
    if ctx["round_idx"] in params.get("rounds", (10, 20, 30, 40)):
        os.makedirs(params["dir"], exist_ok=True)
        torch.save({"global": ctx["global_vector"].clone(), "deltas": deltas.cpu().clone(), "cum": st["cum"].clone(),
                    "malicious": ctx["_malicious_ids_diag_only"], "round": ctx["round_idx"]},
                   os.path.join(params["dir"], f"{params['tag']}_r{ctx['round_idx']}.pt"))
    return fedavg(deltas, weights, ctx, params)
