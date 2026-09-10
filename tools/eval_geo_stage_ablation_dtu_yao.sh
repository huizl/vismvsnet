#!/bin/bash
# Inference-only stage-wise Geo ablation on the 18-scan DTU validation split.
# Usage: GPU=2 bash tools/eval_geo_stage_ablation_dtu_yao.sh
# Set SKIP_EXISTING=0 to force all cases to run again.

set -u

GPU="${GPU:-2}"
PYTHON="${PYTHON:-python}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/dtu/vis_geo_view5_alpha02/best_2mm.ckpt}"
OUTDIR="${OUTDIR:-./eval/18scan_geo_stage_ablation_light3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-tools/eval_region_metrics_dtu_yao.py}"
LIGHT="${LIGHT:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRINT_FREQ="${PRINT_FREQ:-50}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MERGED_CSV="${MERGED_CSV:-${OUTDIR}/all_geo_stage_ablation.csv}"
FAILED_LOG="${FAILED_LOG:-${OUTDIR}/failed.txt}"
TOTAL_CASES=6

mkdir -p "${OUTDIR}"
: > "${FAILED_LOG}"

successful_csvs=()
failed_cases=0

run_case() {
  local label="$1"
  local stage1_alpha="$2"
  local stage2_alpha="$3"
  local stage3_alpha="$4"
  local csv_path="${OUTDIR}/${label}.csv"

  if [[ "${SKIP_EXISTING}" -eq 1 && -s "${csv_path}" ]]; then
    echo "SKIP: ${label} (existing CSV: ${csv_path})"
    successful_csvs+=("${csv_path}")
    return
  fi

  echo
  echo "======================================================================"
  echo "START: ${label}"
  echo "checkpoint: ${CHECKPOINT}"
  echo "stage alphas: ${stage1_alpha}, ${stage2_alpha}, ${stage3_alpha}"
  echo "test list: ${TESTLIST}; light: ${LIGHT}; test views: 5"
  echo "output: ${csv_path}"
  echo "======================================================================"

  local command=(
    "${PYTHON}" "${EVAL_SCRIPT}"
    --model_type geo
    --loadckpt "${CHECKPOINT}"
    --label "${label}"
    --testpath "${TESTPATH}"
    --testlist "${TESTLIST}"
    --out_csv "${csv_path}"
    --vismode soft
    --geo_alpha 0.2
    --stage1_geo_alpha "${stage1_alpha}"
    --stage2_geo_alpha "${stage2_alpha}"
    --stage3_geo_alpha "${stage3_alpha}"
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

run_case "geo_current_020202" 0.2 0.2 0.2
run_case "geo_gate_off_000000" 0.0 0.0 0.0
run_case "geo_stage1_only_020000" 0.2 0.0 0.0
run_case "geo_tapered_030100" 0.3 0.1 0.0
run_case "geo_stage1_strong_050000" 0.5 0.0 0.0
run_case "geo_stage1_full_100000" 1.0 0.0 0.0

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
echo "Completed cases: ${#successful_csvs[@]}/${TOTAL_CASES}"
echo "Merged CSV: ${MERGED_CSV}"
echo "Failed cases: ${failed_cases}; log: ${FAILED_LOG}"
echo "======================================================================"

if [[ "${failed_cases}" -ne 0 ]]; then
  exit 1
fi
