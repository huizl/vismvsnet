#!/usr/bin/env bash
# Run the 27 completed experiments in the result table sequentially.
# Each job starts immediately after the previous one finishes.
# Usage: bash eval_region_queue.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GPU=2
DATAPATH="/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu"
TESTLIST="lists/dtu/val.txt"
CKPT_ROOT="./checkpoints/dtu"
OUTDIR="./eval/18scan_region_metrics"
EVAL_SCRIPT="tools/eval_region_metrics_dtu_yao.py"

mkdir -p "${OUTDIR}"

run_eval() {
    local label="$1"
    local model_type="$2"
    local checkpoint="$3"
    local geo_alpha="$4"
    local csv_path="${OUTDIR}/${label}.csv"

    if [[ -s "${csv_path}" ]]; then
        echo "SKIP: ${label} (${csv_path} already exists)"
        return 0
    fi

    echo
    echo "======================================================================"
    echo "START: ${label}"
    echo "checkpoint: ${checkpoint}"
    echo "======================================================================"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi

    local -a command=(
        python "${EVAL_SCRIPT}"
        --model_type "${model_type}"
        --label "${label}"
        --loadckpt "${checkpoint}"
        --testpath "${DATAPATH}"
        --testlist "${TESTLIST}"
        --eval_nviews 5
        --region_nviews 5
        --out_csv "${csv_path}"
    )

    case "${model_type}" in
        geo|conv_geo|pid_geo|conv_pid_geo)
            command+=(--geo_alpha "${geo_alpha}")
            ;;
    esac

    CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}"
    echo "DONE: ${label}"
}

# Vis baseline: train total views = 2, 3, 5
run_eval "vis_view2" \
    "vis" "${CKPT_ROOT}/d192/vis_view2/best_2mm.ckpt" "-"

run_eval "vis_view3" \
    "vis" "${CKPT_ROOT}/d192/vis_view3/best_2mm.ckpt" "-"

run_eval "vis_view5" \
    "vis" "${CKPT_ROOT}/vis_view5/best_2mm.ckpt" "-"

# Conv (without Geo): train total views = 2, 3
run_eval "conv_view2" \
    "conv" "${CKPT_ROOT}/d192/vis_conv_view2/best_2mm.ckpt" "-"

run_eval "conv_view3" \
    "conv" "${CKPT_ROOT}/d192/vis_conv_view3/best_2mm.ckpt" "-"

run_eval "conv_view5" \
    "conv" "${CKPT_ROOT}/vis_conv_view5/best_2mm.ckpt" "-"

# PID (without Geo): train total views = 2, 3
run_eval "pid_view2" \
    "pid" "${CKPT_ROOT}/d192/vis_pid_view2/best_2mm.ckpt" "-"

run_eval "pid_view3" \
    "pid" "${CKPT_ROOT}/d192/vis_pid_view3/best_2mm.ckpt" "-"
    
run_eval "pid_view5" \
    "pid" "${CKPT_ROOT}/vis_pid_view5/best_2mm.ckpt" "-"

# Conv + PID (without Geo): train total views = 2, 3
run_eval "conv_pid_view2" \
    "conv_pid" "${CKPT_ROOT}/d192/vis_conv_pid_view2/best_2mm.ckpt" "-"

run_eval "conv_pid_view3" \
    "conv_pid" "${CKPT_ROOT}/d192/vis_conv_pid_view3/best_2mm.ckpt" "-"

run_eval "conv_pid_view5" \
    "conv_pid" "${CKPT_ROOT}/vis_conv_pid_view5/best_2mm.ckpt" "-"

# Geo: train total views = 2, 3, 5
run_eval "geo_view2_alpha03" \
    "geo" "${CKPT_ROOT}/vis_geo_view2/best_2mm.ckpt" "0.3"

run_eval "geo_view3_alpha03" \
    "geo" "${CKPT_ROOT}/vis_geo_view3/best_2mm.ckpt" "0.3"

run_eval "geo_view5_alpha03" \
    "geo" "${CKPT_ROOT}/vis_geo_view5/best_2mm.ckpt" "0.3"

# Conv + Geo: train total views = 2, 3, 5
run_eval "conv_geo_view2_alpha03" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view2/best_2mm.ckpt" "0.3"

run_eval "conv_geo_view3_alpha03" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view3/best_2mm.ckpt" "0.3"

run_eval "conv_geo_view5_alpha03" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view5/best_2mm.ckpt" "0.3"

# PID + Geo: train total views = 2, 3, 5
run_eval "pid_geo_view2_alpha03" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view2/best_2mm.ckpt" "0.3"

run_eval "pid_geo_view3_alpha03" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view3/best_2mm.ckpt" "0.3"

run_eval "pid_geo_view5_alpha03" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view5/best_2mm.ckpt" "0.3"

# Conv + PID + Geo: train total views = 2, 3, 5
run_eval "conv_pid_geo_view2_alpha03" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view2/best_2mm.ckpt" "0.3"

run_eval "conv_pid_geo_view3_alpha03" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view3/best_2mm.ckpt" "0.3"

run_eval "conv_pid_geo_view5_alpha03" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view5/best_2mm.ckpt" "0.3"

# Geo: train total views = 2
run_eval "geo_view2_alpha02" \
    "geo" "${CKPT_ROOT}/vis_geo_view2_alpha02/best_2mm.ckpt" "0.2"

run_eval "geo_view3_alpha02" \
    "geo" "${CKPT_ROOT}/vis_geo_view3_alpha02/best_2mm.ckpt" "0.2"

run_eval "geo_view5_alpha02" \
    "geo" "${CKPT_ROOT}/vis_geo_view5_alpha02/best_2mm.ckpt" "0.2"

# Conv + Geo: 
run_eval "conv_geo_view2_alpha02" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view2_alpha02/best_2mm.ckpt" "0.2"

run_eval "conv_geo_view3_alpha02" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view3_alpha02/best_2mm.ckpt" "0.2"

run_eval "conv_geo_view5_alpha02" \
    "conv_geo" "${CKPT_ROOT}/vis_conv_geo_view5_alpha02/best_2mm.ckpt" "0.2"

# PID + Geo: 
run_eval "pid_geo_view2_alpha02" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view2_alpha02/best_2mm.ckpt" "0.2"

run_eval "pid_geo_view3_alpha02" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view3_alpha02/best_2mm.ckpt" "0.2"

run_eval "pid_geo_view5_alpha02" \
    "pid_geo" "${CKPT_ROOT}/vis_pid_geo_view5_alpha02/best_2mm.ckpt" "0.2"

# Conv + PID + Geo: 
run_eval "conv_pid_geo_view2_alpha02" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view2_alpha02/best_2mm.ckpt" "0.2"

run_eval "conv_pid_geo_view3_alpha02" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view3_alpha02/best_2mm.ckpt" "0.2"

run_eval "conv_pid_geo_view5_alpha02" \
    "conv_pid_geo" "${CKPT_ROOT}/vis_conv_pid_geo_view5_alpha02/best_2mm.ckpt" "0.2"

echo
echo "All 36 region evaluations completed."
echo "CSV directory: ${OUTDIR}"
