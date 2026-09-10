#!/usr/bin/env bash
set -euo pipefail

# Trinocular training: one reference view + two source views.
# Override paths with environment variables, e.g. DATAPATH=/data/dtu GPU=1 bash train.sh
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
EXP_NAME="${EXP_NAME:-vis_full_view3}"
LOGDIR="${LOGDIR:-./checkpoints/dtu/${EXP_NAME}}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
  --dataset=dtu_yao \
  --trainpath="${DATAPATH}" \
  --testpath="${DATAPATH}" \
  --trainlist="${TRAINLIST}" \
  --testlist="${TESTLIST}" \
  --logdir="${LOGDIR}" \
  --batch_size="${BATCH_SIZE}" \
  --nviews=3 \
  --test_nviews=3 \
  --numdepth=192 \
  --interval_scale=1.06 \
  --epochs=16 \
  --lr=0.001 \
  --lrepochs="10,12,14:2" \
  --wd=0.0 \
  --vismode=soft \
  --stage1_dnum=48 --stage1_iscale=4 \
  --stage2_dnum=32 --stage2_iscale=2 \
  --stage3_dnum=16 --stage3_iscale=1 \
  --global_candidate_ratio=0.25 \
  --secondary_candidate_ratio=0.25 \
  --summary_freq=20 \
  --save_freq=1 \
  --seed=1 \
  "$@"
