#!/usr/bin/env bash
set -euo pipefail

DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_test}"
GTPATH="${GTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu/Depths}"
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
OUTDIR="${OUTDIR:-./outputs/vis_research_view3}"
CKPT="${CKPT:-./checkpoints/dtu/vis_research_view3/best_2mm.ckpt}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python eval.py \
  --dataset=dtu_yao_eval \
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
  --no_fusion
