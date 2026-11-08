# Réplica fiel do artigo LP (ResNet18 + CIFAR-10) — guia de execução

Objetivo: rodar o ataque LP **exatamente na configuração-manchete do artigo original**
(Zhuang et al.) e verificar se a **nossa proposta** o segura nessa rede — coisa que o nosso
piloto anterior (fase 8C) NÃO testou de forma fiel.

## O que esta réplica tem de fiel (≠ do piloto fase 8C)

| item | artigo (quick-start do README) | esta réplica | piloto 8C (antigo) |
|---|---|---|---|
| modelo | ResNet18 **BatchNorm** (`--model resnet`) | **igual** | ResNet18 **GroupNorm** ✗ |
| dados | CIFAR-10, IID (`--iid 1`) | igual | CIFAR-10 |
| clientes | 100, 10 por rodada (`num_users=100 frac=0.1`) | igual | 20 / 2 ✗ |
| atacantes | 10% (`--malicious 0.1`) = 1/rodada | igual (+ opção 20%) | 10% |
| rodadas | 200 (`--epochs 200`) | igual | 40 ✗ |
| ataque | `lp_attack`, gatilho square (27,27), poison_frac 1.0, lr 0.1 | igual | igual |

O código dos autores **não é alterado**. A nossa defesa entra por injeção em tempo de
execução (`--inject-ours`): troca `utils.defense.flame` por um adaptador que chama a nossa
`layer_guard_sn` e roda com `--defence flame`. A linha de base usa `--defence avg` (FedAvg).

---

## Passo 1 — mover o diretório

Mova **todo o diretório do repositório** para a outra máquina:

```
Poisoning_Backdoor-critical_Layers_Attack-main/
```

Tudo que a réplica precisa está **dentro dele** (inclusive `replica_artigo/layer_guard.py`, uma
cópia da nossa defesa — não depende de nenhum outro diretório). Os índices de partição dos
clientes (`data/iid_cifar.npy`) já vão junto. As **imagens** do CIFAR-10 são baixadas
automaticamente na 1ª execução (precisa de internet uma vez; ~170 MB, gravadas em
`../data/cifar`, ou seja, na pasta *acima* do repositório).

## Passo 2 — instalar as dependências

Requer **Python 3.11** e uma **GPU NVIDIA com driver compatível com CUDA 12.1** (as versões
abaixo são as que funcionam na máquina atual: torch 2.5.1+cu121).

```bash
cd Poisoning_Backdoor-critical_Layers_Attack-main

# 1) criar o ambiente virtual (o mesmo nome que os scripts esperam: .venv-lp)
python3.11 -m venv .venv-lp
source .venv-lp/bin/activate
pip install --upgrade pip

# 2) instalar PyTorch (build CUDA 12.1) a partir do índice oficial do PyTorch
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) instalar o resto (já sem torch; usa o PyPI normal)
pip install -r replica_artigo/requirements.txt
```

Verifique a GPU:

```bash
.venv-lp/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Deve imprimir `True` e o nome da sua placa.

### Se a sua CUDA/driver for diferente
- Driver mais novo costuma rodar o build cu121 sem problema.
- Para outra versão de CUDA, troque o `--index-url` (ex. `.../whl/cu124`) e as tags
  (`+cu124`) em ambos os pacotes. As demais dependências (`requirements.txt`) não mudam.
- **Sem GPU** (só CPU): `pip install torch==2.5.1 torchvision==0.20.1` (sem `+cuXXX`, do PyPI).
  Funciona, mas 200 rodadas em CPU é muito lento — não recomendado.

## Passo 3 — rodar

A partir da **raiz do repositório**:

```bash
bash replica_artigo/rodar.sh
```

Por padrão roda **6 treinos**: `avg` (sem defesa) e `ours` (nossa proposta), para as sementes
1, 2 e 3, com 10% de atacantes. Cada treino = 200 rodadas; o tempo depende da GPU (pode ir de
~1 h a várias horas por treino). O script **pula** treinos já concluídos, então pode ser
reexecutado se for interrompido.

### Uso de GPU — garantia de usar SOMENTE as placas 3 e 4

Por padrão o runner usa **apenas as placas físicas 3 e 4**, e **nunca** nenhuma outra, mesmo
que a máquina tenha mais. Como isso é garantido:
- cada treino é lançado com `CUDA_VISIBLE_DEVICES=<id>` fixando **uma única** placa — o processo
  não enxerga nenhuma outra, então até as chamadas `.cuda()` fixas do código dos autores caem
  na placa certa;
- rodam no máximo **duas** ao mesmo tempo: um treino na placa 3 e outro na placa 4 (um
  "trabalhador" por placa, cada um processando sua fatia da fila em série).

Para mudar o conjunto de placas, use a variável `GPUS` (os scripts respeitam exatamente o que
estiver nela):

```bash
GPUS="3 4" bash replica_artigo/rodar.sh    # padrão: placas 3 e 4, 2 em paralelo
GPUS="3"   bash replica_artigo/rodar.sh    # só a placa 3 (serial)
GPUS="0 1" bash replica_artigo/rodar.sh    # outras placas, se um dia quiser
```

Dá para ajustar por variáveis de ambiente (sem editar nada):

```bash
# incluir também 20% de atacantes (o regime em que a proposta foi desenhada):
MALIC="0.1 0.2" bash replica_artigo/rodar.sh

# comparar também contra as defesas dos próprios autores:
DEFS="avg ours flame multikrum" bash replica_artigo/rodar.sh

# só uma semente, para um teste rápido de ponta a ponta:
SEEDS="1" DEFS="avg ours" bash replica_artigo/rodar.sh

# se o venv tiver outro caminho:
PY=/caminho/para/python bash replica_artigo/rodar.sh
```

Para rodar **um único** treino à mão (útil para depurar). Note o `CUDA_VISIBLE_DEVICES=3` no
início (fixa a placa 3; troque por `4` para a placa 4) e o `--gpu 0` fixo (dentro do processo a
placa escolhida é sempre vista como `cuda:0`):

```bash
# sem defesa (linha de base), na placa 3:
CUDA_VISIBLE_DEVICES=3 .venv-lp/bin/python replica_artigo/rodar_um.py \
  --dataset cifar --model resnet --attack lp_attack --lr 0.1 --poison_frac 1.0 \
  --local_ep 2 --local_bs 64 --attack_begin 0 --epochs 200 --attack_label 5 --attack_goal -1 \
  --trigger square --triggerX 27 --triggerY 27 --gpu 0 --iid 1 \
  --malicious 0.1 --defence avg --seed 1 --save save/rep_avg_m0.1_s1

# nossa proposta (note o --inject-ours e o --defence flame), na placa 4:
CUDA_VISIBLE_DEVICES=4 .venv-lp/bin/python replica_artigo/rodar_um.py --inject-ours \
  --dataset cifar --model resnet --attack lp_attack --lr 0.1 --poison_frac 1.0 \
  --local_ep 2 --local_bs 64 --attack_begin 0 --epochs 200 --attack_label 5 --attack_goal -1 \
  --trigger square --triggerX 27 --triggerY 27 --gpu 0 --iid 1 \
  --malicious 0.1 --defence flame --seed 1 --save save/rep_ours_m0.1_s1
```

## Passo 4 — ler os resultados

```bash
.venv-lp/bin/python replica_artigo/resumo.py
```

Imprime, por treino, **ASR** (sucesso do backdoor) e **acurácia**, média das últimas 10% das
rodadas. Leitura:
- **`avg` (sem defesa):** se o ataque é fiel ao artigo, o ASR deve ficar **alto**
  (o artigo reporta o LP pegando em ResNet/CIFAR a 10% de atacantes). Se o ASR do `avg` for
  baixo, o ataque **não pegou** e o resto é inconclusivo.
- **`ours` (nossa proposta):** ASR **baixo** (≤ ~10%) com acurácia perto da do `avg` = a nossa
  defesa segurou o LP nesta rede. ASR alto = não segurou (é um limite a declarar).

Os números brutos de cada treino ficam em `save/rep_*/accuracy_file*.txt`
(`main_task_accuracy=[...]` e `backdoor_accuracy=[...]`); o log de cada um em `save/rep_*.log`.

---

## Nota técnica importante (BatchNorm)

A nossa defesa agrega por tensor e **não toca nos buffers de BatchNorm**
(`running_mean`/`running_var`): eles ficam com o valor global anterior, não são mediados entre
clientes. Isso é a semântica da nossa proposta, mas na ResNet18-BatchNorm pode **custar
acurácia** (as estatísticas de BN não acompanham o treino federado). Se você observar a
acurácia do `ours` despencar em relação ao `avg`, provavelmente é isso — e não o ataque.
É justamente um dos pontos que esta réplica serve para medir. (O piloto 8C evitou o problema
usando GroupNorm, mas aí deixou de replicar o artigo.)
