#!/usr/bin/env bash
# Train the supervised stage-1 pair-wise visibility head on DTU.
#
# Phase 1 (recommended first):
#   GPU=1 PHASE=head bash tools/train_visibility_head_dtu_yao.sh
# Phase 2 (only after phase-1 regional metrics improve):
#   GPU=2 PHASE=finetune bash tools/train_visibility_head_dtu_yao.sh
# GPU=1 PHASE=head EPOCHS=16 LREPOCHS="6,10,14:2" LOGDIR=./checkpoints/dtu/vis_geo_visibility_view5_head_e16 bash tools/train_visibility_head_dtu_yao.sh

set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-python}"
PHASE="${PHASE:-head}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
# Match the existing 36-model checkpoint-selection protocol so val.txt (18
# scans) remains untouched for the held-out regional evaluation.
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
BASE_CKPT="${BASE_CKPT:-./checkpoints/dtu/vis_geo_view5_alpha02/best_2mm.ckpt}"
HEAD_LOGDIR="${HEAD_LOGDIR:-./checkpoints/dtu/vis_geo_visibility_view5_head}"
FINETUNE_LOGDIR="${FINETUNE_LOGDIR:-./checkpoints/dtu/vis_geo_visibility_view5_finetune}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-0}"

case "${PHASE}" in
  head)
    LOADCKPT="${LOADCKPT:-${BASE_CKPT}}"
    LOGDIR="${LOGDIR:-${HEAD_LOGDIR}}"
    EPOCHS="${EPOCHS:-4}"
    LR="${LR:-0.001}"
    LREPOCHS="${LREPOCHS:-3:2}"
    FREEZE_ARGS=(--freeze_backbone)
    ;;
  finetune)
    LOADCKPT="${LOADCKPT:-${HEAD_LOGDIR}/best_2mm.ckpt}"
    LOGDIR="${LOGDIR:-${FINETUNE_LOGDIR}}"
    EPOCHS="${EPOCHS:-8}"
    LR="${LR:-0.0001}"
    LREPOCHS="${LREPOCHS:-4,6:2}"
    FREEZE_ARGS=()
    ;;
  *)
    echo "ERROR: PHASE must be 'head' or 'finetune', got '${PHASE}'."
    exit 1
    ;;
esac

if [[ ! -f "${LOADCKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${LOADCKPT}"
  exit 1
fi

echo "======================================================================"
echo "Supervised pair-wise visibility training"
echo "phase: ${PHASE}"
echo "GPU: ${GPU}"
echo "load checkpoint: ${LOADCKPT}"
echo "train list: ${TRAINLIST}"
echo "validation list: ${TESTLIST}"
echo "output: ${LOGDIR}"
echo "epochs: ${EPOCHS}; lr: ${LR}; batch size: ${BATCH_SIZE}"
echo "batch limits: train=${MAX_TRAIN_BATCHES}; validation=${MAX_TEST_BATCHES} (0 means full)"
echo "visibility: stage1 only; beta=0.5; focal loss weight=0.2"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_geo.py \
  --dataset=dtu_yao \
  --trainpath="${DATAPATH}" \
  --testpath="${DATAPATH}" \
  --trainlist="${TRAINLIST}" \
  --testlist="${TESTLIST}" \
  --loadckpt="${LOADCKPT}" \
  --logdir="${LOGDIR}" \
  --batch_size="${BATCH_SIZE}" \
  --nviews=5 \
  --numdepth=192 \
  --interval_scale=1.06 \
  --epochs="${EPOCHS}" \
  --lr="${LR}" \
  --lrepochs="${LREPOCHS}" \
  --wd=0.0 \
  --vismode=soft \
  --geo_alpha=0.2 \
  --stage1_geo_alpha=0.0 \
  --stage2_geo_alpha=0.0 \
  --stage3_geo_alpha=0.0 \
  --use_visibility_head \
  --visibility_beta=0.5 \
  --visibility_loss_weight=0.2 \
  --visibility_focal_gamma=2.0 \
  --visibility_occ_abs_tol=2.0 \
  --visibility_occ_rel_tol=0.01 \
  --visibility_gt_downsample=8 \
  --stage1_dnum=48 \
  --stage1_iscale=4 \
  --stage2_dnum=32 \
  --stage2_iscale=2 \
  --stage3_dnum=16 \
  --stage3_iscale=1 \
  --summary_freq=20 \
  --save_freq=1 \
  --max_train_batches="${MAX_TRAIN_BATCHES}" \
  --max_test_batches="${MAX_TEST_BATCHES}" \
  --seed=1 \
  "${FREEZE_ARGS[@]}"
