#!/usr/bin/env bash
set -euo pipefail

# Pairwise visualizations for selected 18-scan held-out candidates.
# Run from the vismvsnet project root:
#   GPU=2 bash tools/vis_selected_18scan_candidates_dtu_yao.sh
#
# Default comparison:
#   vis_view5 -> vis_pid_view5
#
# Optional examples:
#   METHOD_MODEL_TYPE=conv_pid METHOD_CKPT=./checkpoints/dtu/vis_conv_pid_view5/best_2mm.ckpt METHOD_LABEL=vis_conv_pid_view5 GPU=2 bash tools/vis_selected_18scan_candidates_dtu_yao.sh
# METHOD_MODEL_TYPE=conv_pid_geo METHOD_CKPT=./checkpoints/dtu/vis_conv_pid_geo_view5_alpha02/best_2mm.ckpt METHOD_LABEL=vis_conv_pid_geo_view5_alpha02 METHOD_GEO_ALPHA=0.2 GPU=2 bash tools/vis_selected_18scan_candidates_dtu_yao.sh

GPU="${GPU:-2}"
LIGHT="${LIGHT:-3}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
OUTDIR="${OUTDIR:-./outputs/selected_18scan_visuals}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-vis}"
BASE_CKPT="${BASE_CKPT:-./checkpoints/dtu/vis_view5/best_2mm.ckpt}"
BASE_LABEL="${BASE_LABEL:-vis_view5}"

METHOD_MODEL_TYPE="${METHOD_MODEL_TYPE:-pid}"
METHOD_CKPT="${METHOD_CKPT:-./checkpoints/dtu/vis_pid_view5/best_2mm.ckpt}"
METHOD_LABEL="${METHOD_LABEL:-vis_pid_view5}"
METHOD_GEO_ALPHA="${METHOD_GEO_ALPHA:-0.3}"

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

# Candidates selected from:
#   outputs/metrics_model36_scan18_light3/all_metrics.csv
# Main criteria:
#   vis_view5 vs vis_pid_view5 improvement on large_disp_and_occluded,
#   occluded_majority, and boundary_and_occluded, with enough region pixels.
# scan view  reason
CANDIDATES=(
  "scan28 29"   # strongest large_disp_and_occluded and occluded_majority gain
  "scan40 27"   # strong large disparity + boundary/occlusion gain
  "scan82 0"    # strong occlusion gain, good object-scale case
  "scan67 14"   # balanced large_disp_and_occluded and boundary_and_occluded gain
  "scan43 28"   # clear occluded_majority gain
  "scan28 24"   # large region, strong large_disparity/occlusion gain
  "scan40 29"   # compact but clear occlusion/boundary gain
  "scan3 9"     # boundary_and_occluded gain on a different scan
)

for item in "${CANDIDATES[@]}"; do
  read -r scan view _reason <<< "${item}"
  echo "=== Visualizing ${scan} view ${view} light ${LIGHT}: ${BASE_LABEL} vs ${METHOD_LABEL} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/vis_region_diagnostics_dtu_yao.py" \
    "${COMMON_ARGS[@]}" \
    --scan "${scan}" \
    --view "${view}"
done

echo "Saved selected 18-scan visualizations to: ${OUTDIR}"
