#!/usr/bin/env bash
# =====================================================================================
# Roda TODOS os testes restantes do ciclo, na máquina nova, usando SOMENTE as placas 0 e 1.
#
# O que roda, em ordem:
#   BLOCO 1 (lab)   — FASE B: 9B (fase9_sementes, 93 treinos) + 8A/LSA (fase8a, 120 treinos)
#   BLOCO 2 (lab)   — FASE C: overhead FedSurrogate (ResNet18GN/VGG19GN) + relatórios
#   BLOCO 3 (autores) — réplica fiel do artigo LP (ResNet18-BatchNorm/CIFAR-10): avg vs nossa proposta
#   BLOCO 4         — empacota todos os resultados num .tar.gz para você baixar
#
# GPUs: cada treino enxerga APENAS uma placa (CUDA_VISIBLE_DEVICES), nunca outras. Por padrão
#   usa as placas 3 e 4, com 1 treino por placa de cada vez (2 em paralelo). Ajuste com GPUS/PER_GPU.
#
# USO (a partir do diretório IC, onde este script está):
#   bash rodar_tudo.sh
# Em background, sobrevivendo ao logout:
#   nohup bash rodar_tudo.sh > rodar_tudo.log 2>&1 &
# Variáveis opcionais:
#   GPUS="3 4"   placas permitidas (padrão). Ex.: GPUS="3" só a placa 3.
#   PER_GPU=1    treinos simultâneos por placa (aumente p/ 2+ se as placas forem grandes).
#   SKIP_LAB=1   pula BLOCOS 1-2 (lab).   SKIP_REPLICA=1  pula BLOCO 3 (réplica).
# É resumível: relançar continua de onde parou (filas com lock + pulo de concluídos).
# =====================================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="$HERE/novo_ciclo/fl-robustness-lab"
REPO="$HERE/Poisoning_Backdoor-critical_Layers_Attack-main"
LABPY="${LABPY:-$LAB/.venv/bin/python}"
REPOPY="${REPOPY:-$REPO/.venv-lp/bin/python}"
GPUS="${GPUS:-3 4}"
PER_GPU="${PER_GPU:-1}"
read -r -a G <<< "$GPUS"; NG=${#G[@]}
ts() { date '+%F %T'; }

echo "== rodar_tudo.sh =="
echo "IC   : $HERE"
echo "placas: [$GPUS]  | por placa: $PER_GPU  | lab=$LABPY | autores=$REPOPY"
[ -x "$LABPY" ]  || { echo "ERRO: venv do lab não encontrado em $LABPY — crie-o (ver INSTRUÇÕES)."; exit 1; }
[ -x "$REPOPY" ] || echo "AVISO: venv dos autores não encontrado em $REPOPY — o BLOCO 3 (réplica) vai falhar até criá-lo."

# worker de fila do lab, preso a uma placa: roda os experimentos dados até esgotar (rc 3)
lab_worker() {
  local phys="$1"; shift
  local exp rc fails
  for exp in "$@"; do
    fails=0
    while :; do
      CUDA_VISIBLE_DEVICES="$phys" "$LABPY" -u experiments/$exp.py --one >> "results/_run_gpu${phys}.log" 2>&1
      rc=$?
      [ $rc -eq 3 ] && break                 # 3 = nada mais nessa fila
      if [ $rc -ne 0 ]; then
        fails=$((fails+1))
        echo "$(ts) [gpu$phys] $exp rc=$rc (falha $fails/5)" >> "results/_run_gpu${phys}.log"
        [ $fails -ge 5 ] && { echo "$(ts) [gpu$phys] $exp: 5 falhas seguidas, desisto dessa fila" >> "results/_run_gpu${phys}.log"; break; }
        sleep 10
      else
        fails=0                              # 0 = rodou um treino; segue pegando o próximo
      fi
    done
  done
}

# ---------------------------------------------------------------------------
if [ "${SKIP_LAB:-0}" != 1 ]; then
  cd "$LAB" || { echo "ERRO: não achei o lab em $LAB"; exit 1; }
  echo; echo "$(ts) === BLOCO 1: FASE B (9B + 8A) ==="
  # limpa bandeiras de parada herdadas e locks órfãos de execuções anteriores
  rm -f results/fase9/STOP results/fase9/STOP2 results/fase8a/STOP results/fase8d/NAO_MEDIR
  for d in results/fase9/raw results/fase9/sementes results/fase8a/raw; do
    [ -d "$d" ] || continue
    for L in "$d"/*.lock; do [ -e "$L" ] || continue
      p=$(cat "$L" 2>/dev/null); { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || rm -f "$L"; done
  done
  echo "pendentes: 9B=$("$LABPY" experiments/fase9_sementes.py --missing 2>/dev/null | head -1) | 8A=$("$LABPY" experiments/fase8a.py --missing 2>/dev/null | head -1)"

  PIDS=()
  for ((i=0; i<NG; i++)); do
    for ((k=0; k<PER_GPU; k++)); do
      lab_worker "${G[$i]}" fase9_sementes fase8a & PIDS+=($!)
    done
  done
  wait "${PIDS[@]}"
  echo "$(ts) FASE B concluída. Restantes: 9B=$("$LABPY" experiments/fase9_sementes.py --missing 2>/dev/null | head -1) | 8A=$("$LABPY" experiments/fase8a.py --missing 2>/dev/null | head -1)"

  echo; echo "$(ts) === BLOCO 2: FASE C (overhead + relatórios) ==="
  L=results/fase8/overhead_novamaquina.log
  for m in resnet18gn vgg19gn; do
    echo "$(ts) overhead $m" ; CUDA_VISIBLE_DEVICES="${G[0]}" "$LABPY" -u experiments/fase8_overhead.py --model $m --dataset cifar10 >> "$L" 2>&1
  done
  "$LABPY" experiments/fase8_overhead.py --report >> "$L" 2>&1
  for s in fase9_sementes fase9_autores fase8a fase8d; do
    "$LABPY" experiments/$s.py --report > results/fase8d/report_$s.log 2>&1 || true
  done
  echo "$(ts) FASE C concluída (relatórios em results/fase8d/report_*.log e $L)"
fi

# ---------------------------------------------------------------------------
if [ "${SKIP_REPLICA:-0}" != 1 ] && [ -x "$REPOPY" ]; then
  cd "$REPO" || { echo "ERRO: não achei o repositório dos autores em $REPO"; exit 1; }
  echo; echo "$(ts) === BLOCO 3: réplica fiel do artigo (ResNet18/CIFAR) ==="
  GPUS="$GPUS" PY="$REPOPY" bash replica_artigo/rodar.sh
  echo "$(ts) réplica concluída"
fi

# ---------------------------------------------------------------------------
echo; echo "$(ts) === BLOCO 4: empacotando resultados ==="
cd "$HERE"
OUT="resultados_$(date +%Y%m%d_%H%M).tar.gz"
LIST="$(mktemp)"
# só os artefatos leves: relatórios .md, resumos .json pequenos (exclui dumps lg_shadow de 4-25 MB),
# logs e accuracy_file. Nada de checkpoints nem dumps de diagnóstico.
LABR=novo_ciclo/fl-robustness-lab/results
{ find "$LABR" -type f -name '*.md'
  find "$LABR" -type f \( -name '*.json' -o -name '*.txt' \) -size -2M
  find "$LABR" -type f -name '*.log' -size -5M
  find Poisoning_Backdoor-critical_Layers_Attack-main/save -path '*rep_*' -type f \
      \( -name 'accuracy_file*' -o -name '*.log' \) -size -5M
} 2>/dev/null > "$LIST"
tar czf "$OUT" -T "$LIST" 2>/dev/null
rm -f "$LIST"
echo "$(ts) PRONTO. Baixe este arquivo e me entregue:"
echo "    $HERE/$OUT"
du -h "$OUT" 2>/dev/null
