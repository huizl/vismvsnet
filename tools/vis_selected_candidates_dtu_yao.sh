#!/usr/bin/env bash
set -euo pipefail

# Pairwise visualizations for selected high-gain large-disparity/occlusion samples.
# Run from the vismvsnet project root:
#   GPU=2 bash tools/vis_selected_candidates_dtu_yao.sh

GPU="${GPU:-0}"
LIGHT="${LIGHT:-3}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
OUTDIR="${OUTDIR:-./outputs/selected_candidate_visuals}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-vis}"
BASE_CKPT="${BASE_CKPT:-./checkpoints/dtu/vis_view5/best_2mm.ckpt}"
BASE_LABEL="${BASE_LABEL:-vis_view5}"

METHOD_MODEL_TYPE="${METHOD_MODEL_TYPE:-pid_geo}"
METHOD_CKPT="${METHOD_CKPT:-./checkpoints/dtu/vis_pid_geo_view5_alpha02/best_2mm.ckpt}"
METHOD_LABEL="${METHOD_LABEL:-vis_pid_geo_view5_alpha02}"
METHOD_GEO_ALPHA="${METHOD_GEO_ALPHA:-0.2}"

COMMON_ARGS=(
  --base_model_type "${BASE_MODEL_TYPE}"
  --base_ckpt "${BASE_CKPT}"
  --base_label "${BASE_LABEL}"
  --method_model_type "${METHOD_MODEL_TYPE}"
  --method_ckpt "${METHOD_CKPT}"
  --method_label "${METHOD_LABEL}"
  --method_geo_alpha "${METHOD_GEO_ALPHA}"
  --testpath "${TESTPATH}"
  --testlist "${TESTLIST}"
  --light "${LIGHT}"
  --eval_nviews 5
  --region_nviews 5
  --outdir "${OUTDIR}"
  --vismode soft
  --tile_width 320
  --crop_tile_width 720
  --auto_crop_width 60
  --auto_crop_height 48
  --error_max 20
  --improvement_max 5
)

# Candidates selected from all_metrics.csv on large_disp_and_occluded:
# scan view  notes
CANDIDATES=(
  "scan24 27"
  "scan34 6"
  "scan1 12"
  "scan4 1"
  "scan49 24"
  "scan49 25"
  "scan75 1"
  "scan13 4"
)

for item in "${CANDIDATES[@]}"; do
  read -r scan view <<< "${item}"
  echo "=== Visualizing ${scan} view ${view} light ${LIGHT} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/vis_region_diagnostics_dtu_yao.py" \
    "${COMMON_ARGS[@]}" \
    --scan "${scan}" \
    --view "${view}"
done

echo "Saved selected visualizations to: ${OUTDIR}"
