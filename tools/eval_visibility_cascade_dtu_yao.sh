#!/usr/bin/env bash
# Compare round-trip Geo, stage-1 visibility, and cross-stage propagation.
# Usage: GPU=2 bash tools/eval_visibility_cascade_dtu_yao.sh

set -u

GPU="${GPU:-2}"
PYTHON="${PYTHON:-python}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
ROUNDTRIP_CKPT="${ROUNDTRIP_CKPT:-./checkpoints/dtu/vis_geo_view5_alpha02/best_2mm.ckpt}"
STAGE1_CKPT="${STAGE1_CKPT:-./checkpoints/dtu/vis_geo_visibility_view5_head/best_2mm.ckpt}"
CASCADE_CKPT="${CASCADE_CKPT:-./checkpoints/dtu/vis_geo_visibility_cascade_view5/best_2mm.ckpt}"
OUTDIR="${OUTDIR:-./eval/18scan_geo_visibility_cascade_light3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-tools/eval_region_metrics_dtu_yao.py}"
LIGHT="${LIGHT:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRINT_FREQ="${PRINT_FREQ:-50}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MERGED_CSV="${MERGED_CSV:-${OUTDIR}/all_geo_visibility_cascade.csv}"
FAILED_LOG="${FAILED_LOG:-${OUTDIR}/failed.txt}"

mkdir -p "${OUTDIR}"
: > "${FAILED_LOG}"
successful_csvs=()
failed_cases=0

run_case() {
  local label="$1"
  local model_type="$2"
  local checkpoint="$3"
  local stage1_alpha="$4"
  local csv_path="${OUTDIR}/${label}.csv"

  if [[ "${SKIP_EXISTING}" -eq 1 && -s "${csv_path}" ]]; then
    echo "SKIP: ${label} (existing CSV: ${csv_path})"
    successful_csvs+=("${csv_path}")
    return
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "MISSING: ${label}: ${checkpoint}"
    echo "${label}: missing checkpoint ${checkpoint}" >> "${FAILED_LOG}"
    failed_cases=$((failed_cases + 1))
    return
  fi

  echo
  echo "======================================================================"
  echo "START: ${label}"
  echo "model type: ${model_type}"
  echo "checkpoint: ${checkpoint}"
  echo "test list: ${TESTLIST}; light: ${LIGHT}; test views: 5"
  echo "output: ${csv_path}"
  echo "======================================================================"

  local command=(
    "${PYTHON}" "${EVAL_SCRIPT}"
    --model_type "${model_type}"
    --loadckpt "${checkpoint}"
    --label "${label}"
    --testpath "${TESTPATH}"
    --testlist "${TESTLIST}"
    --out_csv "${csv_path}"
    --vismode soft
    --geo_alpha 0.2
    --stage1_geo_alpha "${stage1_alpha}"
    --stage2_geo_alpha 0.0
    --stage3_geo_alpha 0.0
    --geo_confidence_mode roundtrip
    --visibility_beta 0.5
    --visibility_stage2_beta 0.25
    --visibility_stage3_beta 0.1
    --visibility_stage2_residual_scale 0.25
    --visibility_stage3_residual_scale 0.1
    --eval_nviews 5
    --region_nviews 5
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --light "${LIGHT}"
    --numdepth 192
    --interval_scale 1.06
    --stage1_dnum 48
    --stage1_iscale 4
    --stage2_dnum 32
    --stage2_iscale 2
    --stage3_dnum 16
    --stage3_iscale 1
    --print_freq "${PRINT_FREQ}"
    --seed 1
  )

  if CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"; then
    successful_csvs+=("${csv_path}")
  else
    echo "${label}" >> "${FAILED_LOG}"
    failed_cases=$((failed_cases + 1))
  fi
}

run_case "geo_roundtrip_stage1_a05" "geo" "${ROUNDTRIP_CKPT}" "0.5"
run_case "geo_visibility_stage1_b05" "geo_visibility" "${STAGE1_CKPT}" "0.0"
run_case "geo_visibility_cascade_b0502510" "geo_visibility_cascade" "${CASCADE_CKPT}" "0.0"

: > "${MERGED_CSV}"
first=1
for csv_path in "${successful_csvs[@]}"; do
  if [[ "${first}" -eq 1 ]]; then
    cat "${csv_path}" >> "${MERGED_CSV}"
    first=0
  else
    tail -n +2 "${csv_path}" >> "${MERGED_CSV}"
  fi
done

echo
echo "======================================================================"
echo "Completed cases: ${#successful_csvs[@]}/3"
echo "Merged CSV: ${MERGED_CSV}"
echo "Failed cases: ${failed_cases}; log: ${FAILED_LOG}"
echo "======================================================================"

if [[ "${failed_cases}" -ne 0 ]]; then
  exit 1
fi
