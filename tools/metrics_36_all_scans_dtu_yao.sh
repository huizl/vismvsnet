#!/usr/bin/env bash
set -uo pipefail

# Metrics-only evaluation for the 36-model set over every scan/view in TESTLIST.
# Default: fixed light=3 and views 0..48.
# GPU=1 LIGHT=3 VIEW_START=0 VIEW_END=48 OUTDIR=./outputs/metrics_36_all_light3_full bash tools/metrics_36_all_scans_dtu_yao.sh
# GPU=1 TESTLIST=lists/dtu/val.txt VIEW_START=0 VIEW_END=48 LIGHT_START=3 LIGHT_END=3 OUTDIR=./outputs/metrics_model36_scan18_light3 bash tools/metrics_36_all_scans_dtu_yao.sh

GPU="${GPU:-2}"
LIGHT="${LIGHT:-3}"
VIEW_START="${VIEW_START:-0}"
VIEW_END="${VIEW_END:-48}"
TESTPATH="${TESTPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
OUTDIR="${OUTDIR:-./outputs/metrics_model36_scan18}"
MERGED_CSV="${MERGED_CSV:-${OUTDIR}/all_metrics.csv}"
FAILED_LOG="${FAILED_LOG:-${OUTDIR}/failed.txt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTDIR}"
: > "${FAILED_LOG}"

while IFS= read -r scan || [[ -n "${scan}" ]]; do
  scan="${scan//$'\r'/}"
  [[ -z "${scan}" ]] && continue
  [[ "${scan}" =~ ^# ]] && continue

  for view in $(seq "${VIEW_START}" "${VIEW_END}"); do
    echo "=== ${scan} view ${view} light ${LIGHT} ==="
    if ! SCAN="${scan}" VIEW="${view}" LIGHT="${LIGHT}" GPU="${GPU}" \
        TESTPATH="${TESTPATH}" TESTLIST="${TESTLIST}" OUTDIR="${OUTDIR}" \
        bash "${SCRIPT_DIR}/metrics_36_models_dtu_yao.sh"; then
      echo "${scan},${view},${LIGHT}" | tee -a "${FAILED_LOG}"
    fi
  done
done < "${TESTLIST}"

first=1
: > "${MERGED_CSV}"
find "${OUTDIR}" -name "*_metrics.csv" ! -name "$(basename "${MERGED_CSV}")" | sort | while IFS= read -r csv; do
  if [[ "${first}" -eq 1 ]]; then
    cat "${csv}" >> "${MERGED_CSV}"
    first=0
  else
    tail -n +2 "${csv}" >> "${MERGED_CSV}"
  fi
done

echo "Merged metrics: ${MERGED_CSV}"
echo "Failed samples: ${FAILED_LOG}"
