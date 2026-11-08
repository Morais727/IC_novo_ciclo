#!/usr/bin/env bash
# =====================================================================================
# Sincroniza NOS DOIS SENTIDOS o diretório IC entre ESTA máquina e o dl-16,
# mantendo as duas cópias iguais. Regra: para cada arquivo, a versão MAIS RECENTE vence.
#
# Rode SEMPRE nesta máquina (não no dl-16):   bash sincronizar.sh
#
# - Usa apenas rsync (já instalado). Robusto à rede instável do dl-16: --partial retoma
#   cada arquivo de onde parou e, por padrão, repete até concluir.
# - NÃO apaga arquivos (deleção não se propaga, por segurança) e NÃO sincroniza binários
#   pesados/específicos de máquina: venvs, wheels, datasets, checkpoints, dumps, caches.
#   Para incluir tudo: FULL=1.
#
# Ajustes por variável de ambiente:
#   REMOTE="marcos.morais@dl-16"   usuário@host (troque por IP/FQDN se o nome não resolver)
#   RPATH="IC"                     pasta remota (relativa ao HOME do dl-16  => ~/IC)
#   LOCAL="$HOME/Documentos/IC"    pasta local (o seu IC real nesta máquina)
#   RETRY=0                        tenta uma vez só (padrão: repete até terminar)
#   FULL=1                         sincroniza TUDO (sem as exclusões de binários pesados)
#   ONLY=pull|push                 só um sentido (padrão: os dois)
# =====================================================================================
set -u

REMOTE="${REMOTE:-marcos.morais@dl-16}"
RPATH="${RPATH:-IC}"
LOCAL="${LOCAL:-$HOME/Documentos/IC}"; LOCAL="${LOCAL%/}"
RETRY="${RETRY:-1}"

EXC=(
  --exclude='.venv' --exclude='.venv-*'
  --exclude='wheels/'
  --exclude='__pycache__/' --exclude='*.pyc'
  --exclude='data/' --exclude='data_cache/' --exclude='cifar*/'
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
  --exclude='*lg_shadow*'
  --exclude='.git/'
)
[ "${FULL:-0}" = 1 ] && EXC=()

# --update = nunca sobrescreve um arquivo mais novo no destino (o recente vence);
# --partial = retoma; -a preserva mtime (evita reenvio em pingue-pongue).
COMMON=(-az --update --partial --human-readable --itemize-changes)

pull() { rsync "${COMMON[@]}" "${EXC[@]}" "$REMOTE:$RPATH/" "$LOCAL/"; }
push() { rsync "${COMMON[@]}" "${EXC[@]}" "$LOCAL/" "$REMOTE:$RPATH/"; }

retry() {   # repete a função até sair 0 (ou uma vez só, se RETRY!=1)
  local f="$1" n=0
  if [ "$RETRY" != 1 ]; then "$f"; return $?; fi
  until "$f"; do n=$((n+1)); echo ">> '$f' interrompido (tentativa $n) — repetindo em 10s... (Ctrl-C p/ parar)"; sleep 10; done
}

echo "== sincronização bidirecional =="
echo "local : $LOCAL"
echo "remoto: $REMOTE:$RPATH   (= ~/$RPATH no dl-16)"
echo "regra : o arquivo MAIS RECENTE vence; nada é apagado; completo(FULL)=${FULL:-0}; sentido=${ONLY:-ambos}"
echo
mkdir -p "$LOCAL"

case "${ONLY:-ambos}" in
  pull) echo "--- PUXANDO dl-16 -> local ---";  retry pull ;;
  push) echo "--- ENVIANDO local -> dl-16 ---"; retry push ;;
  *)
    echo "--- (1/2) PUXANDO dl-16 -> local (traz o que é mais novo lá) ---"; retry pull
    echo
    echo "--- (2/2) ENVIANDO local -> dl-16 (envia o que é mais novo aqui) ---"; retry push
    ;;
esac
echo
echo "== concluído: as duas cópias estão sincronizadas =="
