#!/usr/bin/env python
"""Roda UM treino de main_fed.py (código original dos autores, SEM alterá-lo).

- Semeia random/numpy/torch ANTES de importar o código dos autores (reprodutibilidade).
- Com a flag --inject-ours, troca utils.defense.flame por um adaptador que chama a nossa
  proposta (layer_guard_sn, limiar 3) e você roda com --defence flame. Sem a flag, roda a
  defesa dos autores tal como está (--defence avg, flame, multikrum, ...).
- O adaptador agrega por tensor do state_dict (mesma granularidade do LP), pesos iguais
  (como o FedAvg dos autores). NÃO mexe nos buffers de BatchNorm (running_mean/var): essa é
  a semântica da nossa defesa — ver nota no LEIA-ME.

Uso (a partir de qualquer lugar; o script já entra no diretório do repositório):
  python replica_artigo/rodar_um.py [--inject-ours] <ARGS de main_fed.py ...>
Ex.:
  python replica_artigo/rodar_um.py --inject-ours --dataset cifar --model resnet \
      --attack lp_attack --defence flame --seed 1 --save save/rep_ours_m0.1_s1 ...
"""
import os
import random
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent     # .../replica_artigo
REPO = HERE.parent                         # raiz do repositório dos autores
os.chdir(REPO)                             # main_fed.py usa caminhos relativos (./data, ../data)
sys.path.insert(0, str(REPO))              # para importar utils/, models/, main_fed
sys.path.insert(0, str(HERE))              # para importar a nossa layer_guard.py (copiada aqui)

import numpy as np
import torch

argv = sys.argv[1:]
inject = "--inject-ours" in argv
argv = [a for a in argv if a != "--inject-ours"]

seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if inject:
    from layer_guard import layer_guard_sn   # cópia local — sem depender de outro diretório
    import utils.defense as D

    def adapter(w_locals, w_updates, w_glob, args, debug=False):
        keys = [k for k in w_glob if w_glob[k].dtype.is_floating_point
                and k.split(".")[-1] not in ("running_mean", "running_var")]
        groups, pos = {}, 0
        for k in keys:
            groups[k] = slice(pos, pos + w_glob[k].numel())
            pos += w_glob[k].numel()
        dev = w_glob[keys[0]].device
        deltas = torch.stack([torch.cat([u[k].reshape(-1).float().to(dev) for k in keys])
                              for u in w_updates])
        agg, info = layer_guard_sn(deltas, torch.ones(len(w_updates)),
                                   {"state": {"tensor_groups": groups}}, {})
        for k in keys:
            w_glob[k] = w_glob[k] + agg[groups[k]].reshape(w_glob[k].shape).to(w_glob[k].dtype)
        return w_glob

    D.flame = adapter
    print(">>> nossa proposta INJETADA (utils.defense.flame substituída; rode com --defence flame)",
          flush=True)

sys.argv = ["main_fed.py"] + argv
runpy.run_path("main_fed.py", run_name="__main__")
