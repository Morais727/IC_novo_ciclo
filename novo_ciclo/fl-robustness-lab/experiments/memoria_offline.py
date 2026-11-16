#!/usr/bin/env python
"""Teste OFFLINE (13/11/2026): uma corroboração COM MEMÓRIA pegaria o adversário adaptativo?
Usa o diag_log do portão adaptativo (fase8g, lg_shadow): z_sim/z_norm por tensor × cliente × rodada.
count_i(r) = nº de tensores que acusam i (z_sim+ + z_norm+ > 3), como na layer_guard_corr.
Memória: S_i(r) = λ·S_i(r−1) + count_i(r). Limiar calibrado SÓ nas rodadas sem ataque (20–79): o máximo de S entre
TODOS os clientes nessas rodadas (nenhum conhecimento do ataque). Avalia nas rodadas 80–200 (janela de ataque).
Comparação: a regra sem memória (count >= T=8), que o atacante foi construído para evadir."""
import json, sys
import numpy as np

Z = 3.0
for s in sys.argv[1:] or ["42", "43"]:
    d = json.load(open(f"results/fase8g/raw/laysel_adapt_T8__lg_shadow__a0.9_s{s}.json"))
    mal = set(d["malicious_ids"]); n = d["num_clients"]; dl = d["diag_log"]
    C = np.zeros((len(dl), n))
    for r, cells in enumerate(dl):
        for cell in cells.values():
            C[r] += (np.maximum(np.array(cell["z_sim"]), 0) + np.maximum(np.array(cell["z_norm"]), 0)) > Z
    del d, dl
    ism = np.array([i in mal for i in range(n)])
    att = slice(79, 200)          # rodadas 80..200 (índice 0 = rodada 1)
    pre = slice(19, 79)           # rodadas 20..79, sem ataque
    print(f"\n== semente {s}: contagem média por rodada de ataque — maliciosos {C[att][:, ism].mean():.1f}, "
          f"benignos {C[att][:, ~ism].mean():.1f} (pré-ataque, todos: {C[pre].mean():.1f}, máx {C[pre].max():.0f})")
    def rate(F):
        tp = F[att][:, ism].sum(); fp = F[att][:, ~ism].sum()
        rec = tp / F[att][:, ism].size; prec = tp / max(tp + fp, 1)
        rounds_all = (F[att][:, ism].sum(1) >= ism.sum()).mean()      # rodadas em que TODOS os 6 maliciosos são pegos
        return prec, rec, rounds_all
    p, r_, a = rate(C >= 8)
    print(f"  sem memória (T=8):            precisão {p:.2f}  recall {r_:.2f}  rodadas c/ os 6 pegos {a:.2f}")
    for lam in (0.5, 0.7, 0.8, 0.9):
        S = np.zeros_like(C)
        for r in range(len(C)):
            S[r] = (lam * S[r - 1] if r else 0) + C[r]
        thr = S[pre].max()
        p, r_, a = rate(S > thr)
        print(f"  memória λ={lam} (limiar {thr:5.1f}): precisão {p:.2f}  recall {r_:.2f}  rodadas c/ os 6 pegos {a:.2f}")


# ---- v2: contagem RELATIVA à rodada (c − mediana da rodada), memória EWMA, limiar = quantil das rodadas 50–79
def v2(seeds=("42", "43")):
    for s in seeds:
        d = json.load(open(f"results/fase8g/raw/laysel_adapt_T8__lg_shadow__a0.9_s{s}.json"))
        mal = set(d["malicious_ids"]); n = d["num_clients"]; dl = d["diag_log"]
        C = np.zeros((len(dl), n))
        for r, cells in enumerate(dl):
            for cell in cells.values():
                C[r] += (np.maximum(np.array(cell["z_sim"]), 0) + np.maximum(np.array(cell["z_norm"]), 0)) > Z
        del d, dl
        ism = np.array([i in mal for i in range(n)])
        R = C - np.median(C, axis=1, keepdims=True)
        att, cal = slice(79, 200), slice(49, 79)
        print(f"\n== v2 semente {s}")
        for lam in (0.0, 0.5, 0.8, 0.9, 0.95):
            S = np.zeros_like(R)
            for r in range(len(R)):
                S[r] = (lam * S[r - 1] if r else 0) + R[r]
            for q in (99, 100):
                thr = np.percentile(S[cal], q)
                F = S > thr
                tp = F[att][:, ism].sum(); fp = F[att][:, ~ism].sum()
                rec = tp / F[att][:, ism].size; prec = tp / max(tp + fp, 1)
                fpr = fp / F[att][:, ~ism].size
                print(f"  λ={lam:<4} limiar q{q} ({thr:6.1f}): precisão {prec:.2f}  recall {rec:.2f}  FP/benigno/rodada {fpr:.3f}")


if __name__ == "__main__" and "--v2" in sys.argv:
    pass
