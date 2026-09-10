#!/usr/bin/env bash
# Train stage-2/3 visibility residuals on top of the 4-epoch stage-1 head.
# Usage: GPU=2 bash tools/train_visibility_cascade_dtu_yao.sh

set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-python}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
BASE_CKPT="${BASE_CKPT:-./checkpoints/dtu/vis_geo_visibility_view5_head/best_2mm.ckpt}"
LOGDIR="${LOGDIR:-./checkpoints/dtu/vis_geo_visibility_cascade_view5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-0.001}"
LREPOCHS="${LREPOCHS:-3:2}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-0}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: stage-1 visibility checkpoint not found: ${BASE_CKPT}"
  exit 1
fi

echo "======================================================================"
echo "Cross-stage visibility propagation training"
echo "GPU: ${GPU}"
echo "stage-1 checkpoint: ${BASE_CKPT}"
echo "train list: ${TRAINLIST}"
echo "validation list: ${TESTLIST}"
echo "output: ${LOGDIR}"
echo "trainable: stage2/stage3 original occ_head only"
echo "visibility betas: 0.50, 0.25, 0.10"
echo "visibility residual scales: 0.00, 0.25, 0.10"
echo "visibility loss weights: 1.00, 0.50, 0.25"
echo "epochs: ${EPOCHS}; lr: ${LR}; batch size: ${BATCH_SIZE}"
echo "batch limits: train=${MAX_TRAIN_BATCHES}; validation=${MAX_TEST_BATCHES}"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_geo.py \
  --dataset=dtu_yao \
  --trainpath="${DATAPATH}" \
  --testpath="${DATAPATH}" \
  --trainlist="${TRAINLIST}" \
  --testlist="${TESTLIST}" \
  --loadckpt="${BASE_CKPT}" \
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
  --use_visibility_propagation \
  --visibility_beta=0.5 \
  --visibility_stage2_beta=0.25 \
  --visibility_stage3_beta=0.1 \
  --visibility_stage2_residual_scale=0.25 \
  --visibility_stage3_residual_scale=0.1 \
  --visibility_loss_weight=0.2 \
  --visibility_stage2_loss_weight=0.5 \
  --visibility_stage3_loss_weight=0.25 \
  --visibility_focal_gamma=2.0 \
  --visibility_occ_abs_tol=2.0 \
  --visibility_occ_rel_tol=0.01 \
  --visibility_gt_downsample=2 \
  --freeze_backbone \
  --freeze_stage1_visibility \
  --reset_visibility_residual_heads \
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
  --seed=1

