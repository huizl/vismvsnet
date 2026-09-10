#!/bin/bash
# Vis-MVSNet evaluation script (DTU test set)
# Usage: bash eval.sh

DATAPATH="/home/disk_10T/lzh_data/dtu_test"
GTPATH="/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu/Depths"
TESTLIST="lists/dtu/test.txt"

EVAL_SCRIPT="eval_geo.py"
OUTDIR="./outputs/vis_geo_view2_alpha03_best2mm"
CKPT="./checkpoints/dtu/vis_geo_view2/best_2mm.ckpt"
GPU=0
NVIEWS=2
GEO_ALPHA=0.3

GEO_ARGS=()
case "${EVAL_SCRIPT}" in
    *geo.py) GEO_ARGS+=(--geo_alpha=${GEO_ALPHA}) ;;
esac

CUDA_VISIBLE_DEVICES=${GPU} python ${EVAL_SCRIPT} \
    --dataset=dtu_yao_eval \
    --testpath=${DATAPATH} \
    --testlist=${TESTLIST} \
    --outdir=${OUTDIR} \
    --loadckpt=${CKPT} \
    --gtpath=${GTPATH} \
    --batch_size=1 \
    --nviews=${NVIEWS} \
    --numdepth=192 \
    --interval_scale=1.06 \
    --vismode=soft \
    --stage1_dnum=48 \
    --stage1_iscale=4 \
    --stage2_dnum=32 \
    --stage2_iscale=2 \
    --stage3_dnum=16 \
    --stage3_iscale=1 \
    "${GEO_ARGS[@]}" \
    --no_fusion
