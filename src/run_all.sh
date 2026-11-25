#!/bin/bash
# Matriz de experimentos na receita v2 (Split CIFAR-100, 10 seeds, retomada automática).
#   bash run_all_v2.sh              -> todos os estágios, em ordem
#   bash run_all_v2.sh we fixed     -> só os estágios listados
# Variáveis de ambiente:
#   EXTRA="--epochs 60 --width_mult 1.5"   flags para TODOS os runs (o que o sweep do joint decidir);
#                         as flags de cada estágio (ex.: a largura do `matched`) têm prioridade
#   PAR=4                 execuções simultâneas na mesma GPU (cada uma = 1 seed de 1 configuração)
#   PAR_DER=2             idem para os estágios der/orth (DER é bem mais pesado; padrão: min(PAR,2))
#   DRY=1                 só imprime os comandos
#   SMOKE=1               dados sintéticos (teste do fluxo; pastas terminam em _smoke)
# Estágios: we fixed matched joint der orth abl_lr abl_guard abl_proto
DATASET=cifar100
STRATS=${STRATS:-"naive ewc si lwf lwm er agem"}   # pode ser restringido, ex.: STRATS="lwf"
EXTRA=${EXTRA:-""}
SFX=""
if [ -n "$SMOKE" ]; then EXTRA="$EXTRA --smoke"; SFX="_smoke"; fi

PAR=${PAR:-1}
PAR_DER=${PAR_DER:-$(( PAR < 2 ? PAR : 2 ))}

run() { echo "+ python $*"; if [ -z "$DRY" ]; then python "$@" || exit 1; fi; }
# $EXTRA vem ANTES dos argumentos do estágio: em argparse vale a última ocorrência
matrix() { run src/run_matrix.py --dataset $DATASET $EXTRA "$@" --parallel $CUR_PAR; }
CUR_PAR=$PAR

# largura da CNN fixa que iguala o nº FINAL de parâmetros do WE (receita v2: pool gap + emb 128)
width_for() {  # $1 = família (rcnn|rcnn_att)  $2 = diretório de resultados do WE
  if [ -n "$DRY" ]; then echo "<largura_$1>"; return; fi
  python src/match_width.py --results "$2" --family $1 --pool gap --emb_dim 128 \
    | sed -n 's/.*--width_mult \([0-9.]*\).*/\1/p'
}

stage_we()      { matrix --archs rwe rwe_att --strategies $STRATS; }
stage_fixed()   { matrix --archs rcnn rcnn_att --strategies $STRATS; }
stage_matched() {  # precisa do estágio we pronto (usa os parâmetros finais do LwF)
  W=$(width_for rcnn     results/$DATASET/rwe_lwf_v2$SFX)
  WA=$(width_for rcnn_att results/$DATASET/rwe_att_lwf_v2$SFX)
  echo "# larguras: rcnn=$W  rcnn_att=$WA"
  if [ -z "$DRY" ] && { [ -z "$W" ] || [ -z "$WA" ]; }; then
    echo "ERRO: não consegui calcular as larguras (rode o estágio 'we' antes)"; exit 1
  fi
  matrix --archs rcnn     --strategies $STRATS --width_mult $W  --tag_suffix _matched
  matrix --archs rcnn_att --strategies $STRATS --width_mult $WA --tag_suffix _matched
}
stage_joint()   { matrix --archs rcnn rcnn_att --strategies joint; }
stage_der() {      # referências COM replay (100 exemplares/classe, como ER/A-GEM)
  CUR_PAR=$PAR_DER
  matrix --strategies der orth_der --backbone res
  matrix --strategies der orth_der --backbone res --extra_prune
}
stage_orth() {     # checagens do Orth-DER: memória do artigo (20/classe) e variante reescalada
  CUR_PAR=$PAR_DER
  matrix --strategies der orth_der --backbone res --extra_prune --mem_per_class 20 --tag_suffix _mem20
  matrix --strategies orth_der --backbone res --extra_prune --orth_reduce norm
}
stage_abl_lr() {   # decaimento de LR por geração + controle sem expansão (por etapa)
  for f in 0.9 0.7 0.5; do
    matrix --archs rwe_att  --strategies lwf er --lr_decay $f
    matrix --archs rcnn_att --strategies lwf er --lr_decay $f --lr_decay_on step
  done
}
stage_abl_guard() {  # proteção da memória de slots
  for g in none importance projection; do
    matrix --archs rwe_att --strategies lwf er --mem_guard $g --tag_suffix _guard$g
  done
  matrix --archs rwe_att --strategies lwf er --att_mem_gain 4 --tag_suffix _gain4
  matrix --archs rwe_att --strategies lwf er --mem_mode append --mem_guard none --tag_suffix _append
}
stage_abl_proto() {  # protótipos sem compensação de deriva (as duas cabeças já saem em todo run)
  matrix --archs rwe_att --strategies naive lwf er --no_proto_sdc --tag_suffix _nosdc
}

STAGES=${@:-we fixed matched joint der orth abl_lr abl_guard abl_proto}
for s in $STAGES; do echo "##### estágio: $s"; CUR_PAR=$PAR; stage_$s; done
echo "Resumo: python src/summarize.py results/$DATASET/rwe_att_lwf_v2 results/$DATASET/rcnn_att_lwf_v2_matched --baseline results/$DATASET/rcnn_att_lwf_v2_matched"
