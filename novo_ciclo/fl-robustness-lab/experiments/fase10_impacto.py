#!/usr/bin/env python
"""Fase 10 (12/11/2026) — O que significa, na prática, um modelo final com ASR de ~10% ou ~20%?

Pergunta: o lab trata "ASR final <= 20%" como contenção. Esse número sozinho não diz se o backdoor ainda está lá.
Esta fase mede o comportamento dos modelos finais, no cenário 8E (LaySelFL BadNet, ResNet18-GN/CIFAR-10, alvo 4).

Treino (refaz 3 corridas do 8E salvando o vetor global; os .json originais do 8E NÃO são tocados):
  - corr  s44  (no 8E terminou em 21,4%: o "modelo de 20%")
  - corr  s42  (no 8E terminou em 7,9%:  o "modelo de ~10%")
  - lg_shadow s44 (sem defesa, 89,2%: controle positivo; os checkpoints da subida dão modelos com o backdoor
    sendo implantado, com ASR ~10% e ~20%, para comparar com os modelos DEFENDIDOS de mesmo ASR)
  Checkpoints: rodada 79 (modelo limpo, antes do ataque), 80–140 de 5 em 5, 150–200 de 10 em 10, 205.
  O GPU não é determinístico: o ASR refeito pode diferir do 8E; a análise usa o que for medido aqui.

Análise (offline, todas as métricas sobre as 9.000 imagens de teste que não são da classe-alvo), FIXADA ANTES:
  1. Piso natural: taxa de "classe 4" sem gatilho e com o gatilho no modelo limpo (rodada 79).
  2. Especificidade: ASR com o gatilho verdadeiro menos o maior ASR entre gatilhos de controle (mesmas tiras
     deslocadas para baixo; mesmas tiras com cores aleatórias, média de 5 sorteios). Backdoor implantado => o
     gatilho verdadeiro funciona e os controles não. Excesso ~0 => o "ASR" residual é confusão genérica.
  3. Sequestro efetivo: fração das imagens CORRETAMENTE classificadas sem gatilho que viram classe 4 com ele.
  4. Confiança: p(classe 4) média nos acertos do ataque e fração de imagens com p(4) > 0,9.
  5. Confiabilidade: 10 variações por imagem (deslocamento de 1 px, espelhamento, brilho ±10%, ruído gaussiano
     σ=0,03 por cima do gatilho). Por imagem, h = fração das variações sequestradas. "Confiável" = h >= 0,9;
     "alguma" = h > 0 (o atacante tenta 10 vezes). Mostra se o ASR é um backdoor estável ou ruído de fronteira.
  6. ASR por classe de origem.
Uso:
  CUDA_VISIBLE_DEVICES=3 python -u experiments/fase10_impacto.py --train --defense layer_guard_corr --seed 44
  python -u experiments/fase10_impacto.py --analyze [--limit 500]
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path("results/fase10")
MODELS = OUT / "models"
ROUNDS = [79] + list(range(80, 141, 5)) + list(range(150, 201, 10)) + [205]
TARGET = 4


def name(defense, seed):
    return f"impacto__{defense}__s{seed}"


def train(defense, seed, final_only=False):
    import experiments.fase8e as f8e      # fixa o gatilho badnet_cifar
    from experiments.fase4_alpha import mem_ok
    from src.federated.simulate import run
    f8e.SEED = seed
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    n = name(defense, seed)
    f = OUT / "raw" / f"{n}.json"
    if f.exists():
        print(f"[skip] {n} já existe", flush=True); return
    lock = OUT / "raw" / f"{n}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0); print(f"[lock vivo] {n}", flush=True); return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    while not mem_ok():
        time.sleep(30)
    c = f8e.cfg(defense)
    c.name = n
    os.environ["FL_SAVE_FINAL"] = str(MODELS)
    os.environ["FL_SAVE_ROUNDS"] = "79,205" if final_only else ",".join(map(str, ROUNDS))   # final_only: limpo + final
    print(f"\n=== {n} pid={os.getpid()} device={c.device} checkpoints={ROUNDS} ===", flush=True)
    json.dump(run(c), open(f, "w"), default=str)
    lock.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------- análise
def _strips(x, rows, values):
    x = x.clone()
    for i, c0 in enumerate((1, 7, 13, 19)):
        v = values if not hasattr(values, "shape") else values[i]
        x[:, :, rows[0]:rows[1], c0:c0 + 5] = v
    return x


def trig_real(x):
    from src.data.datasets import TRIGGER_VALUE
    return _strips(x, (1, 4), TRIGGER_VALUE)


def trig_shift(x):
    from src.data.datasets import TRIGGER_VALUE
    return _strips(x, (28, 31), TRIGGER_VALUE)


def trig_randcolor(x, g):
    import torch
    vals = torch.empty(4, 3, 3, 5).uniform_(-1.9, 2.1, generator=g)   # faixa de pixel válida pós-normalização
    return _strips(x, (1, 4), vals.unsqueeze(1).to(x.device))          # [4,1,C,3,5] -> broadcast no lote


def variants(x, k, g):
    """k-ésima variação de ambiente: transforma a imagem, carimba o gatilho verdadeiro, ruído por cima."""
    import torch
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0), (1, 1), (0, 0), (-1, -1), (0, 0)]
    flip = k in (5, 6)
    bright = {7: 1.1, 8: 0.9, 9: 1.1}.get(k, 1.0)
    y = torch.roll(x, shifts=shifts[k], dims=(2, 3))
    if flip:
        y = torch.flip(y, dims=(3,))
    y = y * bright
    y = trig_real(y)
    return y + 0.03 * torch.randn(y.shape, generator=g).to(y.device) if k > 0 else y


def analyze_model(vec_path, X, Y, Xall, Yall, device, ctor):
    import torch
    from src.federated.utils import load_vector
    model = ctor().to(device)
    load_vector(model, torch.load(vec_path, map_location="cpu").to(device))
    model.eval()

    @torch.no_grad()
    def probs(fn, data, bs=500):
        out = []
        for i in range(0, len(data), bs):
            out.append(torch.softmax(model(fn(data[i:i + bs].to(device))), 1).cpu())
        return torch.cat(out)

    g = torch.Generator().manual_seed(0)
    ident = lambda b: b
    p_clean = probs(ident, X)
    p_real = probs(trig_real, X)
    ma = (probs(ident, Xall).argmax(1) == Yall).float().mean().item() * 100
    hit = p_real.argmax(1) == TARGET
    clean_ok = p_clean.argmax(1) == Y
    r = {"ma": ma,
         "asr_real": hit.float().mean().item() * 100,
         "asr_sem_gatilho": (p_clean.argmax(1) == TARGET).float().mean().item() * 100,
         "asr_tiras_deslocadas": (probs(trig_shift, X).argmax(1) == TARGET).float().mean().item() * 100}
    rc = []
    for _ in range(5):
        gs = torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,), generator=g)))
        rc.append((probs(lambda b: trig_randcolor(b, gs), X).argmax(1) == TARGET).float().mean().item() * 100)
    r["asr_tiras_cor_aleatoria"] = sum(rc) / len(rc)
    r["excesso_sobre_controles"] = r["asr_real"] - max(r["asr_tiras_deslocadas"], r["asr_tiras_cor_aleatoria"])
    r["sequestro_efetivo"] = (hit & clean_ok).float().sum().item() / max(1, clean_ok.sum().item()) * 100
    p4 = p_real[:, TARGET]
    r["p4_media_nos_acertos"] = p4[hit].mean().item() if hit.any() else 0.0
    r["frac_p4_maior_0.9"] = (p4 > 0.9).float().mean().item() * 100
    H = torch.zeros(len(X))
    for k in range(10):
        gk = torch.Generator().manual_seed(100 + k)
        H += (probs(lambda b: variants(b, k, gk), X).argmax(1) == TARGET).float()
    H /= 10
    r["confiavel_h>=0.9"] = (H >= 0.9).float().mean().item() * 100
    r["alguma_h>0"] = (H > 0).float().mean().item() * 100
    r["h_medio"] = H.mean().item() * 100
    r["asr_por_classe"] = {int(c): (hit[Y == c].float().mean().item() * 100) for c in Y.unique().tolist()}
    return r


def analyze(limit=None, marker=None):
    import torch
    import experiments.fase8e  # noqa: F401  (gatilho badnet_cifar no processo)
    from src.data.datasets import load_cifar10
    from src.models.cnn import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_cifar10()
    Xall = torch.stack([test[i][0] for i in range(len(test))]); Yall = torch.tensor(test.targets)
    if limit:
        Xall, Yall = Xall[:limit], Yall[:limit]
    keep = Yall != TARGET
    X, Y = Xall[keep], Yall[keep]
    ctor = lambda: build_model("resnet18gn", dataset="cifar10")
    res_f = OUT / "IMPACTO.json"
    res = json.load(open(res_f)) if res_f.exists() and not limit else {}
    paths = [p for p in sorted(glob.glob(str(MODELS / "*.pt"))) if re.search(r"_r\d+$", Path(p).stem)]   # sem a cópia final duplicada
    paths += sorted(glob.glob("results/models_final/*.pt"))      # modelos finais das corridas normais (13/11: artigo 2)
    for p in paths:
        k = Path(p).stem
        if k in res:
            continue
        t = time.time()
        res[k] = analyze_model(p, X, Y, Xall, Yall, device, ctor)
        print(f"{k}: ASR {res[k]['asr_real']:.1f}  excesso {res[k]['excesso_sobre_controles']:.1f}  "
              f"confiável {res[k]['confiavel_h>=0.9']:.1f}  ({time.time() - t:.0f}s)", flush=True)
        if not limit:
            json.dump(res, open(res_f, "w"), indent=1)
    if limit:
        print(json.dumps(res, indent=1)[:3000])
    else:
        report(res)
        if marker:
            Path(marker).write_text(time.strftime("%F %T") + "\n")


def report(res):
    def rnd(k):
        m = re.search(r"_r(\d+)$", k); return int(m.group(1)) if m else -1
    cols = [("asr_real", "ASR"), ("asr_sem_gatilho", "sem gat."), ("asr_tiras_deslocadas", "ctrl desl."),
            ("asr_tiras_cor_aleatoria", "ctrl cor"), ("excesso_sobre_controles", "excesso"),
            ("sequestro_efetivo", "seq. efetivo"), ("p4_media_nos_acertos", "p4 acertos"),
            ("confiavel_h>=0.9", "confiável"), ("alguma_h>0", "alguma em 10"), ("ma", "MA")]
    L = ["# Fase 10 — impacto prático de um ASR residual de ~10% / ~20%", "",
         "Gerado por experiments/fase10_impacto.py --analyze. Percentuais sobre as 9.000 imagens de teste fora da classe-alvo.", "",
         "| modelo | rodada | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for k in sorted(res, key=lambda k: (k.rsplit("_r", 1)[0], rnd(k))):
        v = res[k]
        cells = [f"{v[c]:.2f}" if c == "p4_media_nos_acertos" else f"{v[c]:.1f}" for c, _ in cols]
        L.append(f"| {k.rsplit('_r', 1)[0].replace('impacto__', '')} | {rnd(k)} | " + " | ".join(cells) + " |")
    (OUT / "RESULTADO.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--defense"); ap.add_argument("--seed", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--final-only", action="store_true", help="salva só as rodadas 79 e 205")
    ap.add_argument("--marker", help="arquivo criado ao fim da análise (para a fila)")
    a = ap.parse_args()
    if a.train:
        train(a.defense, a.seed, a.final_only)
    elif a.analyze:
        analyze(a.limit, a.marker)
