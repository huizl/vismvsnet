#!/bin/bash
# Vis-MVSNet training script (DTU dataset)
# Usage: bash train.sh

DATAPATH="/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/test.txt"
LOGDIR="./checkpoints/dtu/vis_conv_pid_geo_view5_alpha02"
TRAIN_SCRIPT="train_conv_pid_geo.py"
GPU=0
GEO_ALPHA=0.2

CUDA_VISIBLE_DEVICES=${GPU} python ${TRAIN_SCRIPT} \
    --dataset=dtu_yao \
    --trainpath=${DATAPATH} \
    --testpath=${DATAPATH} \
    --trainlist=${TRAINLIST} \
    --testlist=${TESTLIST} \
    --logdir=${LOGDIR} \
    --batch_size=4 \
    --nviews=5 \
    --numdepth=192 \
    --interval_scale=1.06 \
    --epochs=16 \
    --lr=0.001 \
    --lrepochs="10,12,14:2" \
    --wd=0.0 \
    --vismode=soft \
    --geo_alpha=${GEO_ALPHA} \
    --stage1_dnum=48 \
    --stage1_iscale=4 \
    --stage2_dnum=32 \
    --stage2_iscale=2 \
    --stage3_dnum=16 \
    --stage3_iscale=1 \
    --summary_freq=20 \
    --save_freq=1 \
    --seed=1




























# #!/usr/bin/env bash
# set -euo pipefail

# # Vis-MVSNet single-model training script (DTU dataset)
# #
# # Run from the vismvsnet project root, for example:
# #   GPU=2 MODEL=conv bash train.sh
# #   GPU=2 MODEL=pid bash train.sh
# #   GPU=2 MODEL=conv_pid bash train.sh
# #
# # MODEL choices:
# #   conv      -> train_conv.py      -> ./checkpoints/dtu/vis_conv_view5
# #   pid       -> train_pid.py       -> ./checkpoints/dtu/vis_pid_view5
# #   conv_pid  -> train_conv_pid.py  -> ./checkpoints/dtu/vis_conv_pid_view5

# DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
# TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
# TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
# CKPT_ROOT="${CKPT_ROOT:-./checkpoints/dtu}"

# GPU="${GPU:-0}"
# MODEL="${MODEL:-conv}"
# BATCH_SIZE="${BATCH_SIZE:-4}"
# N_VIEWS=5
# EPOCHS="${EPOCHS:-16}"
# SEED="${SEED:-1}"

# case "${MODEL}" in
#     conv)
#         LOGDIR="${CKPT_ROOT}/vis_conv_view5"
#         TRAIN_SCRIPT="train_conv.py"
#         ;;
#     pid)
#         LOGDIR="${CKPT_ROOT}/vis_pid_view5"
#         TRAIN_SCRIPT="train_pid.py"
#         ;;
#     conv_pid)
#         LOGDIR="${CKPT_ROOT}/vis_conv_pid_view5"
#         TRAIN_SCRIPT="train_conv_pid.py"
#         ;;
#     *)
#         echo "ERROR: unknown MODEL='${MODEL}'. Use one of: conv, pid, conv_pid"
#         exit 1
#         ;;
# esac

# echo "======================================================================"
# echo "START: ${MODEL}"
# echo "script: ${TRAIN_SCRIPT}"
# echo "GPU: ${GPU}"
# echo "train views: ${N_VIEWS}; test views: 5"
# echo "logdir: ${LOGDIR}"
# echo "======================================================================"

# CUDA_VISIBLE_DEVICES="${GPU}" python "${TRAIN_SCRIPT}" \
#     --dataset=dtu_yao \
#     --trainpath="${DATAPATH}" \
#     --testpath="${DATAPATH}" \
#     --trainlist="${TRAINLIST}" \
#     --testlist="${TESTLIST}" \
#     --logdir="${LOGDIR}" \
#     --batch_size="${BATCH_SIZE}" \
#     --nviews="${N_VIEWS}" \
#     --numdepth=192 \
#     --interval_scale=1.06 \
#     --epochs="${EPOCHS}" \
#     --lr=0.001 \
#     --lrepochs="10,12,14:2" \
#     --wd=0.0 \
#     --vismode=soft \
#     --stage1_dnum=48 \
#     --stage1_iscale=4 \
#     --stage2_dnum=32 \
#     --stage2_iscale=2 \
#     --stage3_dnum=16 \
#     --stage3_iscale=1 \
#     --summary_freq=20 \
#     --save_freq=1 \
#     --seed="${SEED}"

# echo "DONE: ${MODEL}"
