#!/usr/bin/env bash
set -euo pipefail

# No-training check of the raw Geo confidence against pair-wise DTU GT
# visibility.  The default is the 18-scan held-out list at light 3.
#
# Run from the vismvsnet project root:
#   GPU=2 bash tools/check_geo_confidence_dtu_yao.sh
#
# Check all seven lights after the light-3 screening run:
#   LIGHT=all GPU=2 OUTDIR=./outputs/geo_confidence_check/geo_view5_alpha02_scan18_all_lights bash tools/check_geo_confidence_dtu_yao.sh
#
# Check another Geo variant:
#   MODEL_TYPE=pid_geo CKPT=./checkpoints/dtu/vis_pid_geo_view5_alpha02/best_2mm.ckpt LABEL=pid_geo_view5_alpha02 GEO_ALPHA=0.2 GPU=2 bash tools/check_geo_confidence_dtu_yao.sh

GPU="${GPU:-2}"
MODEL_TYPE="${MODEL_TYPE:-geo}"
CKPT="${CKPT:-./checkpoints/dtu/vis_geo_view5_alpha02/best_2mm.ckpt}"
LABEL="${LABEL:-geo_view5_alpha02}"
GEO_ALPHA="${GEO_ALPHA:-0.2}"

TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
LIGHT="${LIGHT:-3}"
OUTDIR="${OUTDIR:-./outputs/geo_confidence_check/${LABEL}_scan18_light${LIGHT}}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SCAN="${SCAN:-}"
VIEW="${VIEW:-}"
PRINT_FREQ="${PRINT_FREQ:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(
  --model_type "${MODEL_TYPE}"
  --loadckpt "${CKPT}"
  --label "${LABEL}"
  --geo_alpha "${GEO_ALPHA}"
  --testpath "${TESTPATH}"
  --testlist "${TESTLIST}"
  --outdir "${OUTDIR}"
  --eval_nviews 5
  --vismode soft
  --numdepth 192
  --interval_scale 1.06
  --stage1_dnum 48
  --stage1_iscale 4
  --stage2_dnum 32
  --stage2_iscale 2
  --stage3_dnum 16
  --stage3_iscale 1
  --large_disp_pct 80
  --occ_abs_tol 2.0
  --occ_rel_tol 0.01
  --confidence_floor 0.1
  --hist_bins 512
  --print_freq "${PRINT_FREQ}"
)

if [[ "${LIGHT}" != "all" ]]; then
  ARGS+=(--light "${LIGHT}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${SCAN}" ]]; then
  ARGS+=(--scan "${SCAN}")
fi
if [[ -n "${VIEW}" ]]; then
  ARGS+=(--view "${VIEW}")
fi

echo "============================================================"
echo "Geo confidence check: ${LABEL}"
echo "model: ${MODEL_TYPE}; alpha: ${GEO_ALPHA}; light: ${LIGHT}"
echo "list: ${TESTLIST}"
echo "output: ${OUTDIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/eval_geo_confidence_dtu_yao.py" "${ARGS[@]}"

echo "Geo confidence diagnostics saved to: ${OUTDIR}"
