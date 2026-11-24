#!/bin/bash
# Varredura da receita no treino conjunto (limite superior). Uma mudança por vez; 3 seeds cada.
# Uso:  bash sweep_joint.sh [ARCH]      (padrão rcnn_att)    depois:  python report_joint.py
ARCH=${1:-rcnn_att}
SEEDS="1 2 3"
OUT=results/sweep_joint
run() {  # nome, args...
  name=$1; shift
  python src/run_experiment.py --dataset cifar100 --arch $ARCH --strategy joint --seeds $SEEDS \
      --quiet --out_dir $OUT/$name "$@"
}
run 0_legacy            --recipe legacy
run 1_cosine            --recipe legacy --lr_schedule cosine
run 2_cosine_60ep       --recipe legacy --lr_schedule cosine --epochs 60
run 3_cosine_gap        --recipe legacy --lr_schedule cosine --pool gap
run 4_cosine_gap_wd     --recipe legacy --lr_schedule cosine --pool gap --weight_decay 0.01
run 5_v2                --recipe v2
run 6_v2_60ep           --recipe v2 --epochs 60
run 7_v2_60ep_wide      --recipe v2 --epochs 60 --width_mult 1.5
run 8_v2_60ep_ls        --recipe v2 --epochs 60 --label_smoothing 0.1
