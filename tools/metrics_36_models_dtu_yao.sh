#!/usr/bin/env bash
set -euo pipefail

# Metrics-only evaluation for the current 36-model set on one DTU sample.
#
# Geo folders without an alpha suffix are evaluated with geo_alpha=0.3.
# Geo folders ending with alpha02 are evaluated with geo_alpha=0.2.

GPU="${GPU:-2}"
SCAN="${SCAN:-scan9}"
VIEW="${VIEW:-0}"
LIGHT="${LIGHT:-3}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
OUTDIR="${OUTDIR:-./outputs/18scan_metrics_36}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/vis_batch_region_models_dtu_yao.py" \
  --model vis_conv_geo_view2:conv_geo:./checkpoints/dtu/vis_conv_geo_view2/best_2mm.ckpt:0.3 \
  --model vis_conv_geo_view2_alpha02:conv_geo:./checkpoints/dtu/vis_conv_geo_view2_alpha02/best_2mm.ckpt:0.2 \
  --model vis_conv_geo_view3:conv_geo:./checkpoints/dtu/vis_conv_geo_view3/best_2mm.ckpt:0.3 \
  --model vis_conv_geo_view3_alpha02:conv_geo:./checkpoints/dtu/vis_conv_geo_view3_alpha02/best_2mm.ckpt:0.2 \
  --model vis_conv_geo_view5:conv_geo:./checkpoints/dtu/vis_conv_geo_view5/best_2mm.ckpt:0.3 \
  --model vis_conv_geo_view5_alpha02:conv_geo:./checkpoints/dtu/vis_conv_geo_view5_alpha02/best_2mm.ckpt:0.2 \
  --model vis_conv_pid_geo_view2:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view2/best_2mm.ckpt:0.3 \
  --model vis_conv_pid_geo_view2_alpha02:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view2_alpha02/best_2mm.ckpt:0.2 \
  --model vis_conv_pid_geo_view3:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view3/best_2mm.ckpt:0.3 \
  --model vis_conv_pid_geo_view3_alpha02:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view3_alpha02/best_2mm.ckpt:0.2 \
  --model vis_conv_pid_geo_view5:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view5/best_2mm.ckpt:0.3 \
  --model vis_conv_pid_geo_view5_alpha02:conv_pid_geo:./checkpoints/dtu/vis_conv_pid_geo_view5_alpha02/best_2mm.ckpt:0.2 \
  --model vis_geo_view2:geo:./checkpoints/dtu/vis_geo_view2/best_2mm.ckpt:0.3 \
  --model vis_geo_view2_alpha02:geo:./checkpoints/dtu/vis_geo_view2_alpha02/best_2mm.ckpt:0.2 \
  --model vis_geo_view3:geo:./checkpoints/dtu/vis_geo_view3/best_2mm.ckpt:0.3 \
  --model vis_geo_view3_alpha02:geo:./checkpoints/dtu/vis_geo_view3_alpha02/best_2mm.ckpt:0.2 \
  --model vis_geo_view5:geo:./checkpoints/dtu/vis_geo_view5/best_2mm.ckpt:0.3 \
  --model vis_geo_view5_alpha02:geo:./checkpoints/dtu/vis_geo_view5_alpha02/best_2mm.ckpt:0.2 \
  --model vis_pid_geo_view2:pid_geo:./checkpoints/dtu/vis_pid_geo_view2/best_2mm.ckpt:0.3 \
  --model vis_pid_geo_view2_alpha02:pid_geo:./checkpoints/dtu/vis_pid_geo_view2_alpha02/best_2mm.ckpt:0.2 \
  --model vis_pid_geo_view3:pid_geo:./checkpoints/dtu/vis_pid_geo_view3/best_2mm.ckpt:0.3 \
  --model vis_pid_geo_view3_alpha02:pid_geo:./checkpoints/dtu/vis_pid_geo_view3_alpha02/best_2mm.ckpt:0.2 \
  --model vis_pid_geo_view5:pid_geo:./checkpoints/dtu/vis_pid_geo_view5/best_2mm.ckpt:0.3 \
  --model vis_pid_geo_view5_alpha02:pid_geo:./checkpoints/dtu/vis_pid_geo_view5_alpha02/best_2mm.ckpt:0.2 \
  --model vis_view5:vis:./checkpoints/dtu/vis_view5/best_2mm.ckpt \
  --model vis_view2:vis:./checkpoints/dtu/d192/vis_view2/best_2mm.ckpt \
  --model vis_view3:vis:./checkpoints/dtu/d192/vis_view3/best_2mm.ckpt \
  --model vis_pid_view2:pid:./checkpoints/dtu/d192/vis_pid_view2/best_2mm.ckpt \
  --model vis_pid_view3:pid:./checkpoints/dtu/d192/vis_pid_view3/best_2mm.ckpt \
  --model vis_pid_view5:pid:./checkpoints/dtu/vis_pid_view5/best_2mm.ckpt \
  --model vis_conv_view2:conv:./checkpoints/dtu/d192/vis_conv_view2/best_2mm.ckpt \
  --model vis_conv_view3:conv:./checkpoints/dtu/d192/vis_conv_view3/best_2mm.ckpt \
  --model vis_conv_view5:conv:./checkpoints/dtu/vis_conv_view5/best_2mm.ckpt \
  --model vis_conv_pid_view2:conv_pid:./checkpoints/dtu/d192/vis_conv_pid_view2/best_2mm.ckpt \
  --model vis_conv_pid_view3:conv_pid:./checkpoints/dtu/d192/vis_conv_pid_view3/best_2mm.ckpt \
  --model vis_conv_pid_view5:conv_pid:./checkpoints/dtu/vis_conv_pid_view5/best_2mm.ckpt \
  --testpath "${TESTPATH}" \
  --testlist "${TESTLIST}" \
  --scan "${SCAN}" \
  --view "${VIEW}" \
  --light "${LIGHT}" \
  --eval_nviews 5 \
  --region_nviews 5 \
  --outdir "${OUTDIR}" \
  --vismode soft \
  --no_images
