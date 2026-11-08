#!/usr/bin/env bash
# Réplica FIEL do artigo LP: ResNet18 (BatchNorm) + CIFAR-10, 100 clientes, 10/rodada,
# 200 rodadas, LP (lp_attack), gatilho 'square' em (27,27), IID (--iid 1).
# Comando-base = quick-start do README dos autores. Compara:
#   - avg   : FedAvg dos autores, SEM defesa (linha de base: o ataque pega?)
#   - ours  : a NOSSA proposta injetada (--defence flame + --inject-ours)
#
# USO DE GPU — garantia: só as placas listadas em GPUS (padrão "3 4") são usadas, NUNCA outras.
#   Cada treino enxerga APENAS a sua placa (CUDA_VISIBLE_DEVICES=<id> + --gpu 0), então até as
#   chamadas .cuda() fixas do código dos autores caem na placa certa. Rodam no máximo tantos
#   treinos em paralelo quantas forem as GPUs em GPUS (2 por padrão: um na 3, outro na 4).
#
# Você pode trocar/estender via variáveis de ambiente, ex.:
#   GPUS="3 4" SEEDS="1 2 3" MALIC="0.1 0.2" DEFS="avg ours" bash replica_artigo/rodar.sh
#   GPUS="3"   ...                      # usar só a placa 3 (serial)
#
# Rode a PARTIR da raiz do repositório:   bash replica_artigo/rodar.sh
set -u
cd "$(dirname "$0")/.."                 # raiz do repositório
PY="${PY:-.venv-lp/bin/python}"         # ajuste se o venv tiver outro nome/caminho

# comando-base do artigo (SEM --gpu: a placa é fixada por treino, abaixo)
PAPER="--dataset cifar --model resnet --attack lp_attack --lr 0.1 --poison_frac 1.0 \
--local_ep 2 --local_bs 64 --attack_begin 0 --epochs 200 --attack_label 5 --attack_goal -1 \
--trigger square --triggerX 27 --triggerY 27 --iid 1"

GPUS="${GPUS:-3 4}"       # <<< placas permitidas. SOMENTE estas serão usadas.
SEEDS="${SEEDS:-1 2 3}"
MALIC="${MALIC:-0.1}"     # 0.1 = 10% (manchete do artigo). Acrescente 0.2 p/ o regime da nossa proposta.
DEFS="${DEFS:-avg ours}"

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
echo "Python: $PY | placas permitidas: [$GPUS] ($NGPU em paralelo)"
"$PY" -c "import torch;print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available())" || {
  echo "ERRO: venv/torch não encontrado. Veja o LEIA-ME.md (passos 1-2)."; exit 1; }

# monta a lista de treinos pendentes (pula os já concluídos)
JOBS=()
for m in $MALIC; do for s in $SEEDS; do for d in $DEFS; do
  tag="rep_${d}_m${m}_s${s}"
  if ls "save/$tag"/accuracy_file* >/dev/null 2>&1; then echo "[pulo ] $tag — já existe"; continue; fi
  JOBS+=("$d $m $s")
done; done; done
echo "treinos a rodar: ${#JOBS[@]}"
[ ${#JOBS[@]} -eq 0 ] && { echo "nada a fazer."; "$PY" replica_artigo/resumo.py; exit 0; }

# roda UM treino, fixado numa placa física específica (vista como cuda:0 dentro do processo)
run_one() {
  local phys="$1" d="$2" m="$3" s="$4"
  local tag="rep_${d}_m${m}_s${s}" out="save/rep_${d}_m${m}_s${s}"
  mkdir -p "$out"
  local inj="" defence
  case "$d" in
    ours) inj="--inject-ours"; defence="flame" ;;   # nossa proposta
    avg)  inj="";              defence="avg"   ;;   # sem defesa (FedAvg)
    *)    inj="";              defence="$d"    ;;   # defesa dos autores (flame, multikrum, ...)
  esac
  echo "[inicio] $tag  GPU física $phys  (malicious=$m seed=$s defence=$defence ${inj:-})  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$phys" "$PY" replica_artigo/rodar_um.py $inj $PAPER \
       --gpu 0 --malicious "$m" --defence "$defence" --seed "$s" --save "$out" > "$out.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[ok    ] $tag  (GPU $phys)  $(date '+%F %T')"
  else echo "[FALHOU] $tag (código $rc, GPU $phys) — veja $out.log"; fi
}

# um trabalhador por placa: o trabalhador g processa os treinos g, g+NGPU, g+2*NGPU, ... na sua placa
worker() {
  local wi="$1" phys="${GPU_ARR[$1]}"
  local i
  for ((i=wi; i<${#JOBS[@]}; i+=NGPU)); do
    read -r d m s <<< "${JOBS[$i]}"
    run_one "$phys" "$d" "$m" "$s"
  done
}

PIDS=()
for ((g=0; g<NGPU; g++)); do worker "$g" & PIDS+=($!); done
wait "${PIDS[@]}"

echo; echo "===== RESUMO ====="
"$PY" replica_artigo/resumo.py
