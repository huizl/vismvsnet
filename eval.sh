#!/usr/bin/env bash
set -euo pipefail

DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_test}"
GTPATH="${GTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu/Depths}"
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
if [[ -z "${DATASET:-}" ]]; then
  if [[ -f "${DATAPATH}/Cameras/pair.txt" ]]; then
    DATASET="dtu_yao"
  else
    DATASET="dtu_yao_eval"
  fi
fi
EXP_NAME="${EXP_NAME:-vis_full_view3}"
OUTDIR="${OUTDIR:-./outputs/${EXP_NAME}}"
CKPT="${CKPT:-./checkpoints/dtu/${EXP_NAME}/best_2mm.ckpt}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python eval.py \
  --dataset="${DATASET}" \
  --testpath="${DATAPATH}" \
  --testlist="${TESTLIST}" \
  --outdir="${OUTDIR}" \
  --loadckpt="${CKPT}" \
  --gtpath="${GTPATH}" \
  --batch_size=1 \
  --nviews=3 \
  --numdepth=192 \
  --interval_scale=1.06 \
  --vismode=soft \
  --stage1_dnum=48 --stage1_iscale=4 \
  --stage2_dnum=32 --stage2_iscale=2 \
  --stage3_dnum=16 --stage3_iscale=1 \
  --no_fusion \
  "$@"
