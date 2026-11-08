#!/usr/bin/env python
"""Regra A8 (docstring de fase3_scan2.py) sobre os efeitos salvos results/fase3/scan2_*.pt."""
import glob, sys, torch
for f in sorted(glob.glob("results/fase3/scan2_s*_r*.pt")):
    d = torch.load(f, weights_only=False); G = d["G"].float(); P = d["patches"]; mal = d["malicious"]
    med = G.median(0).values; mad = 1.4826 * (G - med).abs().median(0).values + 0.05; Z = (G - med) / mad
    sc = Z.topk(2, dim=0).values[1]                        # 2º maior z por (quadrado, classe)
    k = int(sc.flatten().argmax()); q, c = divmod(k, sc.shape[1])
    flagged = (Z[:, q, c] > 3).nonzero().flatten().tolist()
    print(f.split("scan2_")[1][:-3], "candidato", P[q], "classe", c, "escore", round(sc.flatten().max().item(), 1),
          "| marcados", flagged, "| atacantes", mal, "|", "OK" if sorted(flagged) == sorted(mal) else "FALHA")
