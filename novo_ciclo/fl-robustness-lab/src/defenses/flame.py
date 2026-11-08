"""FLAME (Nguyen et al., USENIX Security 2022), fiel a utils/defense.py::flame do repositório do LP
(Zhuang et al., ICLR 2024), que é a versão contra a qual o artigo do LP reporta resultados:

  1. distância de cosseno entre os MODELOS locais (w_global + delta), não entre os deltas;
  2. HDBSCAN(min_cluster_size = n//2 + 1, min_samples = 1, allow_single_cluster = True) sobre as
     linhas da matriz de distâncias (métrica euclidiana padrão, como no original);
     aceita o maior cluster; se tudo for ruído, aceita todos;
  3. corta a norma de cada delta aceito na mediana das normas dos aceitos;
  4. média simples dos aceitos + ruído gaussiano com desvio noise * mediana (noise = 0,001).
"""
import hdbscan
import numpy as np
import torch


def flame(deltas: torch.Tensor, weights: torch.Tensor, ctx: dict, params: dict):
    n = deltas.shape[0]
    models = ctx["global_vector"].to(deltas.device).unsqueeze(0) + deltas
    unit = models / models.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dist = (1 - unit @ unit.T).double().cpu().numpy()
    labels = hdbscan.HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1,
                             allow_single_cluster=True).fit(dist).labels_
    if labels.max() < 0:
        accepted = list(range(n))
    else:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        accepted = [i for i in range(n) if labels[i] == biggest]
    norms = deltas[accepted].norm(dim=1)
    clip = norms.median()
    scale = (clip / norms).clamp(max=1.0)
    agg = (deltas[accepted] * scale.unsqueeze(1)).mean(dim=0)
    agg = agg + torch.randn_like(agg) * params.get("noise", 0.001) * clip
    excluded = [i for i in range(n) if i not in accepted]
    return agg, {"excluded_indices": excluded, "selected_indices": accepted}
