"""Ataques recentes da revisão (2020–2024), além dos 8 avaliados até o mês 3.

  mimic          Karimireddy, He & Jaggi, ICLR 2022 ("Byzantine-Robust Learning on Heterogeneous
                 Datasets via Bucketing"). Os atacantes copiam, em todos os rounds, a atualização de
                 um mesmo cliente honesto: o que mais se projeta na direção de maior variância dos
                 honestos (1º vetor singular), escolhido no 1º round e mantido fixo. O vetor é
                 legítimo; o dano vem de super-representar um cliente não-IID na média.
  dba            Xie et al., ICLR 2020 ("DBA: Distributed Backdoor Attacks against FL"). O gatilho
                 global (quadrado 4x4 no canto inferior direito, o mesmo do backdoor_pixel) é
                 dividido em 4 faixas 1x4; o atacante k só usa a faixa k mod 4. A avaliação usa o
                 gatilho completo.
  lp             Zhuang et al., ICLR 2024 ("Backdoor FL by Poisoning Backdoor-Critical Layers"),
                 Layer-wise Poisoning. O atacante treina um modelo limpo e um com backdoor, acha as
                 camadas críticas (BC) por substituição de camadas e envia o delta limpo com só as
                 camadas BC trocadas pelas do modelo envenenado.
  lf             Idem, Layer-wise Flipping (versão não direcionada): delta limpo com a direção
                 invertida (x -scale) só nas camadas mais críticas para a tarefa principal.
  edge_case_pgd  Wang et al., NeurIPS 2020 ("Attack of the Tails"). Backdoor semântico com amostras
                 raras reais, sem gatilho: aviões de um modo raro e coerente da classe (o extremo do
                 1º componente principal dos pixels) rotulados como caminhão, retirados dos dados dos
                 honestos e dados só aos atacantes, que os misturam com dados limpos em proporção fixa;
                 treino com PGD na bola L2 + escala (como backdoor_pgd).

Simplificações (documentadas): o LP/LF identifica as camadas por grupo de camada da CNN
(conv1..conv3, fc1, fc2) com uma análise de substituição gulosa, sem o ajuste adaptativo de
lambda do artigo; o edge-case usa um modo raro da própria classe "avião" do CIFAR-10 no lugar
das imagens externas (aviões da Southwest) do artigo.

Revisão do edge-case (30/09): a 1ª versão tomava a cauda *radial* (aviões mais distantes do
centroide em qualquer direção) e diluía o pool em todos os dados locais. Avaliada em FedAvg
(40 rounds), teve sucesso final de 4%: o pool era heterogêneo (escuros, claros, coloridos), sem
subpopulação coerente para aprender e generalizar ao teste. A versão atual toma a cauda num único
sentido do 1º componente principal (subpopulação rara e coerente, como os aviões da Southwest) e
mistura edge:limpo em proporção fixa (`clean_ratio`), como o atacante do artigo.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.datasets import TRIGGER_VALUE, BackdoorDataset
from src.federated.utils import load_vector


# ----------------------------------------------------------------------------- Mimic
def mimic(benign_stack: torch.Tensor, params: dict) -> torch.Tensor:
    if "_mimic_idx" not in params:
        c = (benign_stack - benign_stack.mean(0)).double()
        _, vecs = torch.linalg.eigh(c @ c.T)            # Gram n x n: autovetor = projeção no 1º vetor singular
        params["_mimic_idx"] = int(vecs[:, -1].abs().argmax())
    return benign_stack[params["_mimic_idx"]].clone()


# ----------------------------------------------------------------------------- DBA
class DBADataset(Dataset):
    """Como BackdoorDataset, mas com só a faixa `part` (0..3) do gatilho 4x4."""

    def __init__(self, base: Dataset, target_label: int, part: int, poison_fraction: float = 0.5, seed: int = 0):
        self.base, self.target_label, self.row = base, target_label, -4 + part % 4
        self._mask = np.random.default_rng(seed + part).random(len(base)) < poison_fraction

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self._mask[i]:
            x = x.clone()
            x[:, self.row, -4:] = TRIGGER_VALUE
            y = self.target_label
        return x, y


def dba_dataset(base, params: dict):
    return DBADataset(base, params.get("target_label", 0), params.get("attacker_index", 0),
                      params.get("poison_fraction", 0.5), params.get("seed", 0))


# ----------------------------------------------------------------------------- Edge-case
def select_edge_cases(train_full, test_full, source: int, fraction: float):
    """Modo raro e coerente da classe `source`: as amostras no extremo de UM sentido do 1º
    componente principal dos pixels (PCA ajustado no treino; o sentido é o da cauda mais longa).
    O mesmo eixo e o mesmo sentido selecionam as amostras edge-case do teste."""
    tr_y, te_y = np.array(train_full.targets), np.array(test_full.targets)
    tr_idx, te_idx = np.where(tr_y == source)[0], np.where(te_y == source)[0]
    tr_x = train_full.data[tr_idx].reshape(len(tr_idx), -1).astype(np.float32) / 255.0
    te_x = test_full.data[te_idx].reshape(len(te_idx), -1).astype(np.float32) / 255.0
    center = tr_x.mean(0)
    _, _, vt = np.linalg.svd(tr_x - center, full_matrices=False)
    tr_p, te_p = (tr_x - center) @ vt[0], (te_x - center) @ vt[0]
    if abs(tr_p.min()) > abs(tr_p.max()):          # sentido da cauda mais longa
        tr_p, te_p = -tr_p, -te_p
    k_tr, k_te = max(1, int(fraction * len(tr_idx))), max(1, int(fraction * len(te_idx)))
    return tr_idx[np.argsort(-tr_p)[:k_tr]], te_idx[np.argsort(-te_p)[:k_te]]


class _Relabel(Dataset):
    def __init__(self, base: Dataset, label: int):
        self.base, self.label = base, label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i][0], self.label


class SouthwestDataset(Dataset):
    """Aviões da Southwest Airlines do artigo do Edge-case (Wang et al. 2020; repositório OOD_Federated_Learning,
    saved_datasets/southwest_images_new_{train,test}.pkl, convertidos para .npy em data_cache/southwest/):
    784 de treino e 196 de teste, 32x32 RGB, imagens EXTERNAS ao CIFAR-10. Mesma transformação do CIFAR-10."""

    def __init__(self, split: str, root: str = "./data_cache/southwest"):
        from src.data.datasets import TRANSFORM
        self.x, self.tf = np.load(f"{root}/southwest_{split}.npy"), TRANSFORM

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.fromarray(self.x[i])), 0     # rótulo verdadeiro: avião


def edge_case_dataset(base, params: dict):
    """Conjunto edge-case inteiro com o rótulo alvo + `clean_ratio` x |edge| amostras limpas locais."""
    edge = params["edge_dataset"]
    n_clean = min(len(base), int(params.get("clean_ratio", 1.0) * len(edge)))
    rng = np.random.default_rng(params.get("seed", 0) + params.get("attacker_index", 0))
    clean = Subset(base, rng.choice(len(base), size=n_clean, replace=False).tolist())
    return torch.utils.data.ConcatDataset([clean, _Relabel(edge, params.get("target_label", 9))])


# ----------------------------------------------------------------------------- LP / LF
def layer_groups(model: torch.nn.Module) -> dict[str, slice]:
    """Fatias do vetor achatado por módulo com parâmetros (peso + viés juntos)."""
    groups, pos = {}, 0
    for name, p in model.named_parameters():
        mod = name.rsplit(".", 1)[0]
        start = groups[mod].start if mod in groups else pos
        groups[mod] = slice(start, pos + p.numel())
        pos += p.numel()
    return groups


@torch.no_grad()
def _rate(model_ctor, vector, loader, device, target: int | None) -> float:
    """Fração predita como `target` (backdoor) ou acurácia (target=None)."""
    model = model_ctor().to(device)
    load_vector(model, vector.to(device))
    model.eval()
    hit = tot = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        hit += (pred == target).sum().item() if target is not None else (pred == y).sum().item()
        tot += len(y)
    return hit / max(tot, 1)


def _probe(dataset, n: int = 256, seed: int = 0, exclude_label: int | None = None):
    idx = np.random.default_rng(seed).permutation(len(dataset))
    if exclude_label is not None:
        idx = [i for i in idx if dataset[i][1] != exclude_label]
    return Subset(dataset, list(idx[:n]))


class _Triggered(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        x = x.clone()
        x[:, -4:, -4:] = TRIGGER_VALUE
        return x, y


def layerwise_attack(train, model_ctor, global_vector, local_data, device, params: dict) -> torch.Tensor:
    """LP (params["mode"]="lp") ou LF ("lf"). `train(dataset) -> (delta, n)` é o treino local."""
    groups = layer_groups(model_ctor())
    benign, _ = train(local_data)
    tau = params.get("tau", 0.8)

    if params.get("mode", "lp") == "lp":
        target = params.get("target_label", 0)
        poisoned, _ = train(BackdoorDataset(local_data, target, params.get("poison_fraction", 0.5),
                                            params.get("seed", 0)))
        loader = DataLoader(_Triggered(_probe(local_data, exclude_label=target)), batch_size=256)
        score = lambda d: _rate(model_ctor, global_vector + d, loader, device, target)
        goal = tau * score(poisoned)

        def swap(names):
            d = benign.clone()
            for g in names:
                d[groups[g]] = poisoned[groups[g]]
            return d
        # substituição de camada: quanto o backdoor volta ao trocar só a camada g
        ranked = sorted(groups, key=lambda g: -score(swap([g])))
        chosen = []
        for g in ranked:
            chosen.append(g)
            if score(swap(chosen)) >= goal:
                break
        params["_bc_layers"] = chosen
        return swap(chosen)

    # LF: camadas cuja inversão mais derruba a acurácia local; inverte até a acurácia cair a (1-tau)
    loader = DataLoader(_probe(local_data), batch_size=256)
    acc = lambda d: _rate(model_ctor, global_vector + d, loader, device, None)
    scale = params.get("scale", 1.0)

    def flip(names):
        d = benign.clone()
        for g in names:
            d[groups[g]] = -scale * benign[groups[g]]
        return d
    base_acc = acc(benign)
    ranked = sorted(groups, key=lambda g: acc(flip([g])))
    chosen = []
    for g in ranked[: params.get("max_layers", 2)]:
        chosen.append(g)
        if acc(flip(chosen)) <= (1 - tau) * base_acc:
            break
    params["_critical_layers"] = chosen
    return flip(chosen)


# ----------------------------------------------------------------------------- LP completo
def tensor_groups(model: torch.nn.Module) -> dict[str, slice]:
    """Fatias do vetor achatado por tensor de parâmetro (peso e viés separados), a granularidade
    das "camadas" no código dos autores do LP (state_dict keys)."""
    groups, pos = {}, 0
    for name, p in model.named_parameters():
        groups[name] = slice(pos, pos + p.numel())
        pos += p.numel()
    return groups


def craft_lp(center: torch.Tensor, malicious: torch.Tensor, layers, groups, lam: float) -> torch.Tensor:
    """Eq. 4 do artigo no espaço dos deltas (= craft_model de models/Attacker.py):
    camadas em L*: lam*delta_mal + max(0, 1-lam)*delta_centro; demais camadas: delta_centro."""
    d = center.clone()
    for g in layers:
        s = groups[g]
        d[s] = lam * malicious[s] + max(0.0, 1.0 - lam) * center[s]
    return d


def lp_full_attack(train, model_ctor, global_vector, attacker_data: list, benign_deltas: torch.Tensor,
                   device, params: dict) -> torch.Tensor:
    """LP como no código dos autores (models/Attacker.py, attack=lp_attack -> "adaptive"):

    1. cada atacante treina limpo; centro = média desses deltas (u_average, Eq. 4);
    2. um atacante treina com backdoor (BSR medido nos dados dele, com gatilho);
    3. FLS: troca cada tensor do modelo envenenado pelo do limpo e mede a queda do BSR
       (mais queda = mais crítico);
    4. BLS: acrescenta tensores, do mais crítico ao menos, ao modelo limpo até BSR >= tau*BSR
       (tau = mode10_tau = 0.95 nos autores);
    5. controle adaptativo (BLS_adaptive): simula a defesa do servidor com o delta criado (lambda=1)
       repetido para cada atacante; enquanto for rejeitado, tira o tensor menos crítico de L*;
    6. delta final com lambda (cnn_scale = 0.5 para CNN nos autores), o mesmo para todos os atacantes.

    Diferença documentada: no passo 5 o atacante conhece os deltas honestos do round (onisciente,
    como os ataques de modelo do laboratório); os autores estimam os honestos treinando modelos
    com dados de clientes. Isso torna o ataque *mais* forte, ou seja, a avaliação da defesa conservadora.
    """
    from src.data.datasets import BackdoorDataset
    from src.defenses.registry import DEFENSE_REGISTRY

    groups = tensor_groups(model_ctor())
    target = params.get("target_label", 0)
    tau = params.get("tau", 0.95)
    lam = params.get("lambda", 0.5)

    benign = [train(d)[0] for d in attacker_data]
    center = torch.stack(benign).mean(dim=0)
    data0 = attacker_data[0]
    poisoned, _ = train(BackdoorDataset(data0, target, params.get("poison_fraction", 0.5), params.get("seed", 0)))

    loader = DataLoader(_Triggered(_probe(data0, exclude_label=target)), batch_size=256)
    bsr = lambda d: _rate(model_ctor, global_vector + d, loader, device, target)
    bsr_mal = bsr(poisoned)

    # FLS: valor = BSR(envenenado com o tensor g limpo) - BSR(envenenado); menor = mais crítico
    def b2m(g):
        d = poisoned.clone()
        d[groups[g]] = benign[0][groups[g]]
        return d
    value = {g: bsr(b2m(g)) - bsr_mal for g in groups}
    order = sorted(groups, key=lambda g: value[g])

    # BLS: menor n tal que o limpo com os n tensores mais críticos recupera tau do BSR
    n = 0
    while n < len(order):
        n += 1
        d = benign[0].clone()
        for g in order[:n]:
            d[groups[g]] = poisoned[groups[g]]
        if bsr(d) >= tau * bsr_mal:
            break
    n_bls = n

    # controle adaptativo contra a defesa simulada
    sim = params.get("simulate_defense", "multi_krum")
    n_mal = len(attacker_data)
    accepted = None
    if sim and sim != "none":
        defense_fn = DEFENSE_REGISTRY[sim]
        sim_params = params.get("simulate_defense_params", {})
        while n > 0:
            trial = craft_lp(center, poisoned, order[:n], groups, 1.0)
            stack = torch.cat([trial.unsqueeze(0).repeat(n_mal, 1), benign_deltas])
            _, info = defense_fn(stack, torch.ones(len(stack)), {}, sim_params)
            accepted = any(i not in info.get("excluded_indices", []) for i in range(n_mal))
            if accepted or n == 1:
                break
            n -= 1

    layers = order[:n]
    params["_bc_layers"] = layers
    log = params.setdefault("_lp_log", [])
    log.append({"bsr_local": bsr_mal, "n_bls": n_bls, "n_final": n, "accepted_sim": accepted, "layers": layers})
    if params.get("individual"):
        # variante evasiva (não está no artigo): cada atacante usa o PRÓPRIO delta limpo como base, em
        # vez do centro comum, então só as camadas de L* carregam a parte em comum (lambda * envenenado)
        params["_lp_individual"] = [craft_lp(b, poisoned, layers, groups, lam) for b in benign]
        nu = params.get("l_noise", 0.0)
        if nu > 0:
            # H4 (atacante que conhece a defesa por tensor): ruído gaussiano independente em cada tensor
            # de L*, com norma = nu x norma da componente maliciosa (lam * (envenenado - limpo próprio))
            for d, b in zip(params["_lp_individual"], benign):
                for g in layers:
                    s = groups[g]
                    xi = torch.randn_like(d[s])
                    comp = d[s] - b[s]
                    noisy = comp + nu * comp.norm() * xi / xi.norm()
                    if params.get("l_noise_keepnorm"):
                        # contra a norma por tensor: reescala para a norma da componente original
                        noisy = noisy * comp.norm() / noisy.norm()
                    d[s] = b[s] + noisy
        if params.get("evade"):
            params["_lp_individual"] = _evade(params, benign, poisoned, layers, groups, lam, benign_deltas, model_ctor)
    return craft_lp(center, poisoned, layers, groups, lam)


EVADE_S = (3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
EVADE_NU = (0.0, 0.5, 1.0, 2.0)


def _evade(params, benign, poisoned, layers, groups, lam, benign_deltas, model_ctor):
    """Fase 5 (21/10/2026), atacante adaptativo que otimiza contra o escore da nossa proposta (não está no
    artigo do LP). Onisciente: conhece a regra, o limiar e os deltas honestos do round. Para cada atacante k e
    tensor g de L*, a componente maliciosa é c_k[g] = lam * (envenenado - limpo_k)[g]; o atacante envia
    limpo_k + s * (c_k[g] + nu * |c_k[g]| * xi_k[g] / |xi_k[g]|), com xi gaussiano fixo no round.
    Varre (s, nu) na grade EVADE_S x EVADE_NU, simula a defesa (params["evade"], ex.: layer_guard_sn) sobre
    honestos + atacantes e escolhe o par que maximiza s x fração dos pares (atacante, tensor de L*) aceitos
    (empate: maior s, depois menor nu). Ou seja, a maior força de backdoor que passa pelo filtro."""
    from src.defenses.registry import DEFENSE_REGISTRY
    fn = DEFENSE_REGISTRY[params["evade"]]
    ctx = {"state": {"tensor_groups": tensor_groups(model_ctor())}, "model_ctor": model_ctor}
    comps = [{g: lam * (poisoned[groups[g]] - b[groups[g]]) for g in layers} for b in benign]
    xis = [{g: torch.randn_like(c[g]) for g in layers} for c in comps]
    n_mal = len(benign)

    def build(sc, nu):
        out = []
        for b, c, xi in zip(benign, comps, xis):
            d = b.clone()
            for g in layers:
                v = c[g] + nu * c[g].norm() * xi[g] / xi[g].norm()
                d[groups[g]] = b[groups[g]] + sc * v
            out.append(d)
        return out

    best, tried = None, []
    for sc in EVADE_S:
        for nu in EVADE_NU:
            ds = build(sc, nu)
            stack = torch.cat([torch.stack(ds), benign_deltas])
            _, info = fn(stack, torch.ones(len(stack)), ctx, params.get("evade_params", {}))
            flags = info.get("flags", {})
            kept = sum(k not in flags.get(g, []) for g in layers for k in range(n_mal)) / max(len(layers) * n_mal, 1)
            tried.append((sc, nu, kept))
            key = (sc * kept, sc, -nu)
            if best is None or key > best[0]:
                best = (key, sc, nu, kept, ds)
    params.setdefault("_evade_log", []).append({"s": best[1], "nu": best[2], "kept": best[3],
                                                "n_layers": len(layers)})
    return best[4]


# ----------------------------------------------------------------------------- ataques de 2026 (Fase 8A)
def _train_ext(model_ctor, start, global_vector, dataset, device, hp: dict, epochs=None, trainable=None,
               anchor_beta=0.0, epoch_cb=None):
    """Treino local como src/federated/client.py::local_train (SGD, momento 0,9), com extras dos ataques novos:
    parte de `start` (delta em relação ao global), treina só os tensores em `trainable` (os outros congelados),
    soma beta * sum_l ||theta_l - theta_l(servidor)||^2 sobre os treináveis (Eq. 7 da LaySelFL) e chama
    epoch_cb(vetor) ao fim de cada época. Retorna o delta em relação ao modelo global."""
    import torch.nn.functional as F
    from src.federated.utils import flatten
    model = model_ctor().to(device)
    g = global_vector.to(device)
    load_vector(model, g + start.to(device))
    names = dict(model.named_parameters())
    train_names = list(names) if trainable is None else [n for n in names if n in set(trainable)]
    for n, p in names.items():
        p.requires_grad_(n in train_names)
    groups = tensor_groups(model)
    anchor = {n: g[groups[n]].view_as(names[n]) for n in train_names}
    opt = torch.optim.SGD([names[n] for n in train_names], lr=hp["lr"], momentum=0.9)
    loader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True)
    model.train()
    for _ in range(epochs or hp["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if anchor_beta:
                loss = loss + anchor_beta * sum(((names[n] - anchor[n]) ** 2).sum() for n in train_names)
            loss.backward()
            opt.step()
        if epoch_cb:
            epoch_cb(flatten(model).cpu() - global_vector)
    return (flatten(model) - g).detach().cpu()


def lsa_attack(train, model_ctor, global_vector, attacker_data: list, benign_deltas: torch.Tensor,
               device, params: dict) -> torch.Tensor:
    """LSA — Layer Smoothing Attack (Foroughi, Rastegar, Sabokrou e Khonsari, IEEE ICC 2026, arXiv 2602.15161).
    Sem código público; implementado a partir do texto (§III-B, III-C):
    1. cada atacante treina limpo; u_a = média desses deltas (estimativa dos honestos, como no LP);
    2. um atacante treina com backdoor (w_malicious) e mede o BSR com gatilho nos dados dele;
    3. a cada `period` rodadas (5 no artigo), substituição de camadas: Delta BSR_b2m(l) = BSR_mal - BSR(mal com
       a camada l limpa), ordem decrescente; L* cresce até BSR(limpo com L* envenenadas) >= tau * BSR_mal
       (tau = 0,8, o melhor do artigo). Nas outras rodadas reaproveita L*;
    4. "suavização": w_m2b(L*) (limpo com L* do envenenado) é ajustado nos dados envenenados com as camadas fora
       de L* congeladas -> u_ft;
    5. w~ = lam v∘u_ft + ReLU(1 - lam) v∘u_a + (1 - v)∘u_a, v = indicadora de L*, lam = 1 (o melhor do artigo).
       O mesmo w~ é enviado por todos os atacantes (como o LP original).
    Escolhas onde o texto não diz (registradas antes de rodar): camada = tensor de parâmetro (como o código do
    LP); ajuste fino com `ft_epochs` = épocas locais do cenário; fração envenenada 0,5 e gatilho 4x4 do LP do
    laboratório; sem a simulação de defesa do LP (o artigo não a descreve)."""
    from src.data.datasets import BackdoorDataset
    groups = tensor_groups(model_ctor())
    hp, target = params["_hp"], params.get("target_label", 0)
    tau, lam, period = params.get("tau", 0.8), params.get("lambda", 1.0), params.get("period", 5)
    r = params["_round"] = params.get("_round", -1) + 1

    benign = [train(d)[0] for d in attacker_data]
    u_a = torch.stack(benign).mean(dim=0)
    data0 = attacker_data[0]
    pois_data = BackdoorDataset(data0, target, params.get("poison_fraction", 0.5), params.get("seed", 0) + r)
    poisoned, _ = train(pois_data)

    if r % period == 0 or "_lsa_layers" not in params:
        loader = DataLoader(_Triggered(_probe(data0, exclude_label=target)), batch_size=256)
        bsr = lambda d: _rate(model_ctor, global_vector + d, loader, device, target)
        bsr_mal = bsr(poisoned)

        def b2m(g):
            d = poisoned.clone()
            d[groups[g]] = benign[0][groups[g]]
            return d
        drop = {g: bsr_mal - bsr(b2m(g)) for g in groups}
        order = sorted(groups, key=lambda g: -drop[g])
        layers = []
        for g in order:
            layers.append(g)
            d = benign[0].clone()
            for h in layers:
                d[groups[h]] = poisoned[groups[h]]
            if bsr(d) >= tau * bsr_mal:
                break
        params["_lsa_layers"] = layers
        params.setdefault("_lp_log", []).append({"round": r, "bsr_local": bsr_mal, "layers": layers})
    layers = params["_lsa_layers"]

    hybrid = benign[0].clone()
    for g in layers:
        hybrid[groups[g]] = poisoned[groups[g]]
    u_ft = _train_ext(model_ctor, hybrid, global_vector, pois_data, device, hp,
                      epochs=params.get("ft_epochs"), trainable=layers)
    params["_bc_layers"] = layers
    return craft_lp(u_a, u_ft, layers, groups, lam)


def laysel_attack(train, model_ctor, global_vector, attacker_data: list, benign_deltas: torch.Tensor,
                  device, params: dict) -> torch.Tensor:
    """LaySelFL (Li et al., CVPR 2026, "Eliminate Distance Differences Induced by Backdoor Attacks: Layer-Selective
    Training and Clipping to Mask Backdoor Models"). Sem código público encontrado; implementado do texto (§3,
    Alg. 1). Cada atacante, sozinho, com os próprios dados:
    1. SLI: a partir do global, treina N épocas limpo e N com backdoor, anotando por época a variação L1 de cada
       camada; S_d(l) = média das variações do limpo - média das do backdoor (Eq. 4); S_c(l) = ||theta_c(l) -
       theta_atk(l)||_1 no fim (Eq. 3); S = S_d + S_c; camadas sensíveis = TopK(S, k);
    2. treino local normal (limpo) -> theta;
    3. CT: as sensíveis são divididas em P grupos; na rodada r treina só o grupo r mod P (o resto congelado),
       nos dados envenenados, com perda CE + beta * sum ||theta_l - theta_l(servidor)||^2 (Eq. 7; beta = 20);
    4. TC: nas camadas normais (não sensíveis), d = theta - theta_servidor é cortado elemento a elemento em
       [-0,05; 0,05].
    Valores do artigo: beta = 20, corte 0,05, fração envenenada 0,2, k = 60 na ResNet18.
    Escolhas onde o texto não diz (registradas antes de rodar): camada = tensor de parâmetro; k = (nº de
    tensores) - 2, que dá exatamente 60 na nossa ResNet18GN (62 tensores) e 8 na SmallCNN (10); P = 2; N = 2;
    o treino com backdoor parte do modelo já treinado limpo (passo 2) e a âncora da Eq. 7 é o modelo do
    servidor; gatilho 4x4 e classe-alvo do laboratório; ataca desde a 1ª rodada (o artigo começa na 80ª de 205)."""
    from src.data.datasets import BackdoorDataset
    groups = tensor_groups(model_ctor())
    names = list(groups)
    hp, target = params["_hp"], params.get("target_label", 0)
    beta, clip, frac = params.get("beta", 20.0), params.get("clip", 0.05), params.get("poison_fraction", 0.2)
    k = params.get("k") or len(names) - 2
    P, N = params.get("P", 2), params.get("N", 2)
    r = params["_round"] = params.get("_round", -1) + 1
    l1 = lambda v, g: v[groups[g]].abs().sum().item()

    if params.get("trigger") == "a3fl":            # A3FL: refina o gatilho adversarial (1x/rodada) antes de envenenar
        from src.attacks.a3fl import refine_a3fl_trigger
        refine_a3fl_trigger(model_ctor, global_vector, attacker_data, device, params, target)

    out, log = [], []
    for a, data in enumerate(attacker_data):
        pois = BackdoorDataset(data, target, frac, params.get("seed", 0) + 1000 * a + r)
        traj = {"c": [torch.zeros_like(global_vector)], "b": [torch.zeros_like(global_vector)]}
        zero = torch.zeros_like(global_vector)
        th_c = _train_ext(model_ctor, zero, global_vector, data, device, hp, epochs=N,
                          epoch_cb=lambda v: traj["c"].append(v))
        th_b = _train_ext(model_ctor, zero, global_vector, pois, device, hp, epochs=N,
                          epoch_cb=lambda v: traj["b"].append(v))
        mean_step = lambda t, g: sum(l1(t[n] - t[n - 1], g) for n in range(1, len(t))) / (len(t) - 1)
        S = {g: mean_step(traj["c"], g) - mean_step(traj["b"], g) + l1(th_c - th_b, g) for g in names}
        sens = sorted(names, key=lambda g: -S[g])[:k]
        subsets = [sens[i::P] for i in range(P)]
        sel = subsets[r % P]

        clean, _ = train(data)
        d = _train_ext(model_ctor, clean, global_vector, pois, device, hp, trainable=sel, anchor_beta=beta)
        for g in names:
            if g not in sens:
                d[groups[g]] = d[groups[g]].clamp(-clip, clip)
        out.append(d)
        log.append({"sensitive": sens, "trained": sel})
    params["_bc_layers"] = log[0]["trained"]
    params.setdefault("_lp_log", []).append({"round": r, "attackers": [{"n_sens": len(x["sensitive"]),
                                             "normal": [g for g in names if g not in x["sensitive"]],
                                             "trained": x["trained"]} for x in log]})
    params["_lp_individual"] = out
    return out[0]
