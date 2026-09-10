"""Evaluate whether a trained Geo confidence separates GT-visible/occluded pixels.

The model is not changed or trained.  During a normal Geo-model forward pass,
this tool temporarily wraps the model module's ``geometric_confidence`` symbol
and records the exact confidence maps used by the fusion code.  DTU source-view
GT depths then provide pair-wise visibility labels using the same occlusion
definition as ``eval_region_metrics_dtu_yao.py``.
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from datasets import find_dataset_def  # noqa: E402
from eval_region_metrics_dtu_yao import (  # noqa: E402
    MODEL_SPECS,
    build_model,
    load_checkpoint,
    percentile_mask,
    read_dtu_depth,
    read_dtu_mask,
    read_train_camera,
    to_2d,
)


GEO_MODEL_TYPES = tuple(
    name for name, (_module, kwargs) in MODEL_SPECS.items() if kwargs.get("use_geo")
)
STAGE_NAMES = ("stage1", "stage2", "stage3")
REGION_NAMES = ("all", "large_disparity")


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-training DTU check of Geo confidence versus GT source visibility."
    )
    parser.add_argument("--model_type", required=True, choices=sorted(GEO_MODEL_TYPES))
    parser.add_argument("--loadckpt", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--testpath", required=True)
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--outdir", default="./outputs/geo_confidence_check")

    parser.add_argument("--vismode", default="soft", choices=["soft"])
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--stage1_dnum", type=int, default=48)
    parser.add_argument("--stage1_iscale", type=int, default=4)
    parser.add_argument("--stage2_dnum", type=int, default=32)
    parser.add_argument("--stage2_iscale", type=int, default=2)
    parser.add_argument("--stage3_dnum", type=int, default=16)
    parser.add_argument("--stage3_iscale", type=int, default=1)
    parser.add_argument("--geo_alpha", type=float, default=0.3)
    parser.add_argument("--eval_nviews", type=int, default=5)

    parser.add_argument(
        "--light", action="append", type=int, default=None,
        help="Light id to include. Repeat for multiple lights; omit for all seven.",
    )
    parser.add_argument(
        "--scan", action="append", default=None,
        help="Optional scan filter, e.g. --scan scan3. Repeat as needed.",
    )
    parser.add_argument(
        "--view", action="append", type=int, default=None,
        help="Optional reference-view filter. Repeat as needed.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=20)

    parser.add_argument("--large_disp_pct", type=float, default=80.0)
    parser.add_argument("--occ_abs_tol", type=float, default=2.0)
    parser.add_argument("--occ_rel_tol", type=float, default=0.01)
    parser.add_argument("--confidence_floor", type=float, default=0.1)
    parser.add_argument("--hist_bins", type=int, default=512)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def validate_args(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA because it runs the trained Geo model.")
    if args.eval_nviews < 2:
        raise ValueError("--eval_nviews must contain one reference and at least one source view.")
    if not 0.0 < args.large_disp_pct < 100.0:
        raise ValueError("--large_disp_pct must be in (0, 100).")
    if args.occ_abs_tol < 0.0 or args.occ_rel_tol < 0.0:
        raise ValueError("Occlusion tolerances must be non-negative.")
    if not 0.0 <= args.confidence_floor <= 1.0:
        raise ValueError("--confidence_floor must be in [0, 1].")
    if not 16 <= args.hist_bins <= 8192:
        raise ValueError("--hist_bins must be between 16 and 8192.")
    if args.light is not None and any(light < 0 or light > 6 for light in args.light):
        raise ValueError("Every --light must be in [0, 6].")


def _histogram(values, bins):
    if values.size == 0:
        return np.zeros(bins, dtype=np.int64)
    indices = np.minimum((np.clip(values, 0.0, 1.0) * bins).astype(np.int64), bins - 1)
    return np.bincount(indices, minlength=bins).astype(np.int64, copy=False)


def _hist_quantile(histogram, quantile):
    total = int(histogram.sum())
    if total == 0:
        return float("nan")
    target = quantile * max(total - 1, 0)
    index = int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))
    return (index + 0.5) / len(histogram)


def _binary_hist_metrics(visible_hist, occluded_hist):
    visible_count = int(visible_hist.sum())
    occluded_count = int(occluded_hist.sum())
    if visible_count == 0 or occluded_count == 0:
        return {
            "roc_auc_visible": float("nan"),
            "average_precision_visible": float("nan"),
            "best_balanced_accuracy": float("nan"),
            "best_threshold": float("nan"),
        }

    occluded_below = np.cumsum(occluded_hist) - occluded_hist
    concordant = np.sum(visible_hist * (occluded_below + 0.5 * occluded_hist))
    auc = float(concordant / (visible_count * occluded_count))

    visible_desc = visible_hist[::-1].astype(np.float64)
    occluded_desc = occluded_hist[::-1].astype(np.float64)
    true_positive = np.cumsum(visible_desc)
    false_positive = np.cumsum(occluded_desc)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    ap = float(np.sum(precision * (visible_desc / visible_count)))

    true_positive_rate = true_positive / visible_count
    false_positive_rate = false_positive / occluded_count
    true_negative_rate = 1.0 - false_positive_rate
    balanced = 0.5 * (true_positive_rate + true_negative_rate)
    best_index = int(np.argmax(balanced))
    bins = len(visible_hist)
    best_threshold = (bins - 1 - best_index) / bins
    return {
        "roc_auc_visible": auc,
        "average_precision_visible": ap,
        "best_balanced_accuracy": float(balanced[best_index]),
        "best_threshold": float(best_threshold),
    }


class ScoreAccumulator:
    def __init__(self, bins, geo_alpha, confidence_floor):
        self.bins = bins
        self.geo_alpha = geo_alpha
        self.confidence_floor = confidence_floor
        self.visible_hist = np.zeros(bins, dtype=np.int64)
        self.occluded_hist = np.zeros(bins, dtype=np.int64)
        self.visible_sum = 0.0
        self.visible_sq_sum = 0.0
        self.occluded_sum = 0.0
        self.occluded_sq_sum = 0.0
        self.mod_visible_sum = 0.0
        self.mod_occluded_sum = 0.0
        self.pair_count = 0

    def add(self, confidence, visible, occluded):
        finite = np.isfinite(confidence)
        visible_values = np.asarray(confidence[visible & finite], dtype=np.float64)
        occluded_values = np.asarray(confidence[occluded & finite], dtype=np.float64)
        visible_values = np.clip(visible_values, 0.0, 1.0)
        occluded_values = np.clip(occluded_values, 0.0, 1.0)

        self.visible_hist += _histogram(visible_values, self.bins)
        self.occluded_hist += _histogram(occluded_values, self.bins)
        self.visible_sum += float(visible_values.sum())
        self.visible_sq_sum += float(np.square(visible_values).sum())
        self.occluded_sum += float(occluded_values.sum())
        self.occluded_sq_sum += float(np.square(occluded_values).sum())

        visible_mod = 1.0 - self.geo_alpha + self.geo_alpha * np.maximum(
            visible_values, self.confidence_floor
        )
        occluded_mod = 1.0 - self.geo_alpha + self.geo_alpha * np.maximum(
            occluded_values, self.confidence_floor
        )
        self.mod_visible_sum += float(visible_mod.sum())
        self.mod_occluded_sum += float(occluded_mod.sum())
        self.pair_count += 1

    @staticmethod
    def _mean_std(total, square_total, count):
        if count == 0:
            return float("nan"), float("nan")
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        return float(mean), float(math.sqrt(variance))

    def summary(self):
        visible_count = int(self.visible_hist.sum())
        occluded_count = int(self.occluded_hist.sum())
        visible_mean, visible_std = self._mean_std(
            self.visible_sum, self.visible_sq_sum, visible_count
        )
        occluded_mean, occluded_std = self._mean_std(
            self.occluded_sum, self.occluded_sq_sum, occluded_count
        )
        pooled_std = math.sqrt(
            max(0.5 * (visible_std ** 2 + occluded_std ** 2), 0.0)
        ) if np.isfinite(visible_std) and np.isfinite(occluded_std) else float("nan")
        mean_gap = visible_mean - occluded_mean
        metrics = _binary_hist_metrics(self.visible_hist, self.occluded_hist)
        metrics.update({
            "pair_count": self.pair_count,
            "visible_pixels": visible_count,
            "occluded_pixels": occluded_count,
            "visible_mean": visible_mean,
            "visible_std": visible_std,
            "visible_p10": _hist_quantile(self.visible_hist, 0.10),
            "visible_p50": _hist_quantile(self.visible_hist, 0.50),
            "visible_p90": _hist_quantile(self.visible_hist, 0.90),
            "occluded_mean": occluded_mean,
            "occluded_std": occluded_std,
            "occluded_p10": _hist_quantile(self.occluded_hist, 0.10),
            "occluded_p50": _hist_quantile(self.occluded_hist, 0.50),
            "occluded_p90": _hist_quantile(self.occluded_hist, 0.90),
            "mean_gap": mean_gap,
            "standardized_gap": mean_gap / pooled_std if pooled_std > 0.0 else float("nan"),
            "mod_visible_mean": (
                self.mod_visible_sum / visible_count if visible_count else float("nan")
            ),
            "mod_occluded_mean": (
                self.mod_occluded_sum / occluded_count if occluded_count else float("nan")
            ),
        })
        metrics["mod_mean_gap"] = metrics["mod_visible_mean"] - metrics["mod_occluded_mean"]
        return metrics

    def curve_rows(self):
        visible_count = int(self.visible_hist.sum())
        occluded_count = int(self.occluded_hist.sum())
        if visible_count == 0 or occluded_count == 0:
            return []
        visible_desc = self.visible_hist[::-1].astype(np.float64)
        occluded_desc = self.occluded_hist[::-1].astype(np.float64)
        true_positive = np.cumsum(visible_desc)
        false_positive = np.cumsum(occluded_desc)
        precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
        rows = []
        for index in range(self.bins):
            rows.append({
                "threshold": (self.bins - 1 - index) / self.bins,
                "tpr_visible": true_positive[index] / visible_count,
                "fpr_visible": false_positive[index] / occluded_count,
                "precision_visible": precision[index],
                "recall_visible": true_positive[index] / visible_count,
            })
        return rows


def build_pair_gt_maps(datapath, scan, ref_view, src_views, depth, ref_valid,
                       occ_abs_tol, occ_rel_tol):
    """Return per-source visible/occluded maps and max reference-source disparity."""
    height, width = depth.shape
    ref_k, ref_ext = read_train_camera(datapath, ref_view)
    y, x = np.indices((height, width), dtype=np.float32)
    homogeneous = np.stack((
        x.reshape(-1), y.reshape(-1), np.ones(height * width, dtype=np.float32)
    ))
    rays_ref = np.linalg.inv(ref_k).astype(np.float32) @ homogeneous
    xyz_ref = rays_ref * depth.reshape(1, -1)
    xyz_ref_h = np.vstack((xyz_ref, np.ones((1, xyz_ref.shape[1]), dtype=np.float32)))

    pair_maps = []
    max_disparity = np.zeros(height * width, dtype=np.float32)
    ref_ext_inv = np.linalg.inv(ref_ext)
    for src_view in src_views:
        src_k, src_ext = read_train_camera(datapath, src_view)
        xyz_src = (src_ext @ ref_ext_inv)[:3, :] @ xyz_ref_h
        projected = src_k @ xyz_src
        z = projected[2]
        safe_z = np.where(np.abs(z) > 1e-6, z, 1e-6)
        src_x = projected[0] / safe_z
        src_y = projected[1] / safe_z
        disparity = np.sqrt((src_x - homogeneous[0]) ** 2 + (src_y - homogeneous[1]) ** 2)
        max_disparity = np.maximum(max_disparity, disparity.astype(np.float32))

        map_x = src_x.reshape(height, width).astype(np.float32)
        map_y = src_y.reshape(height, width).astype(np.float32)
        projected_depth = z.reshape(height, width).astype(np.float32)
        in_bounds = (
            np.isfinite(map_x) & np.isfinite(map_y) & np.isfinite(projected_depth)
            & (projected_depth > 0.0)
            & (map_x >= 0.0) & (map_x <= width - 1.0)
            & (map_y >= 0.0) & (map_y <= height - 1.0)
        )

        src_depth = read_dtu_depth(datapath, scan, src_view)
        if src_depth.shape != depth.shape:
            src_depth = cv2.resize(src_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        sampled_depth = cv2.remap(
            src_depth, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        src_mask = read_dtu_mask(datapath, scan, src_view, depth.shape)
        if src_mask is None:
            src_mask = np.isfinite(src_depth) & (src_depth > 0.0)
        sampled_mask = cv2.remap(
            src_mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)

        comparable = (
            ref_valid & in_bounds & sampled_mask
            & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        )
        tolerance = np.maximum(occ_abs_tol, occ_rel_tol * sampled_depth)
        occluded = comparable & (projected_depth > sampled_depth + tolerance)
        visible = comparable & ~occluded
        pair_maps.append({
            "source_view": src_view,
            "visible": visible,
            "occluded": occluded,
            "comparable": comparable,
            "disparity": disparity.reshape(height, width).astype(np.float32),
        })

    return pair_maps, max_disparity.reshape(height, width)


def resize_mask(mask, target_shape):
    if mask.shape == target_shape:
        return mask
    return cv2.resize(
        mask.astype(np.uint8), (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def tensorize_model_inputs(sample):
    result = {}
    for key in ("imgs", "proj_matrices", "depth_values"):
        value = np.ascontiguousarray(sample[key])
        result[key] = torch.from_numpy(value).unsqueeze(0).cuda(non_blocking=True)
    return result


def selected_indices(dataset, args):
    lights = set(args.light) if args.light is not None else None
    scans = set(args.scan) if args.scan is not None else None
    views = set(args.view) if args.view is not None else None
    indices = []
    for index, (scan, light, ref_view, _src_views) in enumerate(dataset.metas):
        if lights is not None and light not in lights:
            continue
        if scans is not None and scan not in scans:
            continue
        if views is not None and ref_view not in views:
            continue
        indices.append(index)
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    return indices


def write_csv(filename, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(filename, accumulators, summary_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping summary PNG.")
        return

    stage_colors = {"stage1": "#0072B2", "stage2": "#009E73", "stage3": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    centers = (np.arange(next(iter(accumulators.values())).bins) + 0.5) / next(
        iter(accumulators.values())
    ).bins
    for stage in STAGE_NAMES:
        accumulator = accumulators.get((stage, 0, "all"))
        if accumulator is None:
            continue
        visible_total = max(int(accumulator.visible_hist.sum()), 1)
        occluded_total = max(int(accumulator.occluded_hist.sum()), 1)
        axes[0].plot(
            centers, accumulator.visible_hist / visible_total,
            color=stage_colors[stage], label=stage + " visible",
        )
        axes[0].plot(
            centers, accumulator.occluded_hist / occluded_total,
            color=stage_colors[stage], linestyle="--", label=stage + " occluded",
        )
    axes[0].set_title("Geo confidence distributions")
    axes[0].set_xlabel("raw geo confidence")
    axes[0].set_ylabel("class-normalized frequency")
    axes[0].legend(fontsize=8)

    all_source_rows = {
        (row["stage"], row["region"]): row
        for row in summary_rows if row["source_rank"] == 0
    }
    x = np.arange(len(STAGE_NAMES))
    width = 0.36
    all_auc = [all_source_rows[(stage, "all")]["roc_auc_visible"] for stage in STAGE_NAMES]
    large_auc = [
        all_source_rows[(stage, "large_disparity")]["roc_auc_visible"]
        for stage in STAGE_NAMES
    ]
    axes[1].bar(x - width / 2, all_auc, width, label="all")
    axes[1].bar(x + width / 2, large_auc, width, label="large disparity")
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle=":")
    axes[1].set_xticks(x, STAGE_NAMES)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Visible-vs-occluded ROC-AUC")
    axes[1].legend()

    visible_means = [all_source_rows[(stage, "all")]["visible_mean"] for stage in STAGE_NAMES]
    occluded_means = [all_source_rows[(stage, "all")]["occluded_mean"] for stage in STAGE_NAMES]
    axes[2].bar(x - width / 2, visible_means, width, label="visible")
    axes[2].bar(x + width / 2, occluded_means, width, label="occluded")
    axes[2].set_xticks(x, STAGE_NAMES)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Mean raw Geo confidence")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def diagnostic_grade(auc, gap):
    if not np.isfinite(auc) or not np.isfinite(gap):
        return "insufficient comparable visible/occluded pixels"
    if auc >= 0.70 and gap > 0.05:
        return "informative"
    if auc >= 0.60 and gap > 0.02:
        return "weakly informative"
    if auc < 0.50:
        return "reversed (occluded pixels receive higher confidence)"
    return "very weak / ineffective"


def main():
    args = parse_args()
    validate_args(args)
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    dataset_class = find_dataset_def("dtu_yao")
    dataset = dataset_class(
        args.testpath, args.testlist, "test", args.eval_nviews,
        args.numdepth, args.interval_scale,
    )
    indices = selected_indices(dataset, args)
    if not indices:
        raise RuntimeError("No samples match the requested scan/view/light filters.")

    label = args.label or args.model_type
    module_name, _kwargs = MODEL_SPECS[args.model_type]
    model_module = importlib.import_module(module_name)
    original_geometric_confidence = model_module.geometric_confidence
    captured_confidences = []

    def capture_geometric_confidence(*function_args, **function_kwargs):
        confidence = original_geometric_confidence(*function_args, **function_kwargs)
        captured_confidences.append(confidence.detach())
        return confidence

    model_module.geometric_confidence = capture_geometric_confidence
    try:
        model = build_model(args).cuda()
        load_checkpoint(model, args.loadckpt)
        model.eval()

        pair_rows = []
        accumulators = {}
        previous_geometry_key = None
        cached_pair_maps = None
        cached_large_disparity = None
        start_time = time.time()
        source_count = args.eval_nviews - 1
        expected_captures = len(STAGE_NAMES) * source_count

        print(
            "Checking {}: {} samples, {} total views, lights={}".format(
                label, len(indices), args.eval_nviews,
                sorted(set(args.light)) if args.light is not None else "all",
            )
        )
        with torch.no_grad():
            for position, sample_index in enumerate(indices, start=1):
                sample = dataset[sample_index]
                scan, light, ref_view, all_src_views = dataset.metas[sample_index]
                src_views = all_src_views[:source_count]

                depth_gt = to_2d(sample["depth"])
                ref_mask = to_2d(sample["mask"]) > 0.5
                ref_valid = ref_mask & np.isfinite(depth_gt) & (depth_gt > 0.0)
                geometry_key = (
                    scan, ref_view, tuple(src_views), depth_gt.shape,
                    args.occ_abs_tol, args.occ_rel_tol,
                )
                if geometry_key != previous_geometry_key:
                    cached_pair_maps, max_disparity = build_pair_gt_maps(
                        args.testpath, scan, ref_view, src_views, depth_gt, ref_valid,
                        args.occ_abs_tol, args.occ_rel_tol,
                    )
                    cached_large_disparity = percentile_mask(
                        max_disparity, ref_valid, args.large_disp_pct
                    )
                    previous_geometry_key = geometry_key

                captured_confidences.clear()
                sample_cuda = tensorize_model_inputs(sample)
                model_output = model(
                    sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_values"],
                )
                if len(captured_confidences) != expected_captures:
                    raise RuntimeError(
                        "Expected {} Geo maps (3 stages x {} sources), captured {}. "
                        "The model forward structure may have changed.".format(
                            expected_captures, source_count, len(captured_confidences)
                        )
                    )

                confidence_maps = [
                    to_2d(confidence.float().cpu().numpy())
                    for confidence in captured_confidences
                ]
                del model_output, sample_cuda

                for capture_index, confidence in enumerate(confidence_maps):
                    stage_index = capture_index // source_count
                    source_index = capture_index % source_count
                    stage = STAGE_NAMES[stage_index]
                    source_rank = source_index + 1
                    pair_gt = cached_pair_maps[source_index]
                    target_shape = confidence.shape
                    visible = resize_mask(pair_gt["visible"], target_shape)
                    occluded = resize_mask(pair_gt["occluded"], target_shape)
                    large_disparity = resize_mask(cached_large_disparity, target_shape)

                    for region in REGION_NAMES:
                        region_mask = (
                            np.ones(target_shape, dtype=bool)
                            if region == "all" else large_disparity
                        )
                        region_visible = visible & region_mask
                        region_occluded = occluded & region_mask

                        pair_accumulator = ScoreAccumulator(
                            args.hist_bins, args.geo_alpha, args.confidence_floor
                        )
                        pair_accumulator.add(confidence, region_visible, region_occluded)
                        pair_summary = pair_accumulator.summary()
                        pair_rows.append({
                            "scan": scan,
                            "view": ref_view,
                            "light": light,
                            "label": label,
                            "model_type": args.model_type,
                            "geo_alpha": args.geo_alpha,
                            "stage": stage,
                            "source_rank": source_rank,
                            "source_view": pair_gt["source_view"],
                            "region": region,
                            **pair_summary,
                        })

                        for key in (
                            (stage, source_rank, region),
                            (stage, 0, region),
                        ):
                            if key not in accumulators:
                                accumulators[key] = ScoreAccumulator(
                                    args.hist_bins, args.geo_alpha, args.confidence_floor
                                )
                            accumulators[key].add(
                                confidence, region_visible, region_occluded
                            )

                if position % args.print_freq == 0 or position == len(indices):
                    elapsed = time.time() - start_time
                    print(
                        "[{}/{}] {} view {} light {} elapsed={:.1f}s".format(
                            position, len(indices), scan, ref_view, light, elapsed
                        )
                    )

        summary_rows = []
        for (stage, source_rank, region), accumulator in sorted(accumulators.items()):
            summary_rows.append({
                "label": label,
                "model_type": args.model_type,
                "geo_alpha": args.geo_alpha,
                "eval_nviews": args.eval_nviews,
                "stage": stage,
                "source_rank": source_rank,
                "source_group": "all_sources" if source_rank == 0 else "source_rank_{}".format(source_rank),
                "region": region,
                **accumulator.summary(),
            })

        curve_rows = []
        histogram_rows = []
        for (stage, source_rank, region), accumulator in sorted(accumulators.items()):
            if source_rank != 0:
                continue
            for row in accumulator.curve_rows():
                curve_rows.append({"stage": stage, "region": region, **row})
            for bin_index in range(args.hist_bins):
                histogram_rows.append({
                    "stage": stage,
                    "region": region,
                    "bin_index": bin_index,
                    "confidence_center": (bin_index + 0.5) / args.hist_bins,
                    "visible_pixels": int(accumulator.visible_hist[bin_index]),
                    "occluded_pixels": int(accumulator.occluded_hist[bin_index]),
                })

        pair_csv = os.path.join(args.outdir, "pair_metrics.csv")
        summary_csv = os.path.join(args.outdir, "summary.csv")
        curves_csv = os.path.join(args.outdir, "curves.csv")
        histograms_csv = os.path.join(args.outdir, "histograms.csv")
        write_csv(pair_csv, pair_rows)
        write_csv(summary_csv, summary_rows)
        write_csv(curves_csv, curve_rows)
        write_csv(histograms_csv, histogram_rows)

        config = vars(args).copy()
        config.update({
            "label": label,
            "selected_samples": len(indices),
            "positive_class": "GT visible",
            "negative_class": "GT occluded (projected behind valid source GT surface)",
            "out_of_view_policy": "excluded from visible-vs-occluded classification",
            "auc_method": "fixed-bin histogram with tie correction",
        })
        with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=True, indent=2)

        if not args.no_plot:
            save_summary_plot(
                os.path.join(args.outdir, "summary.png"), accumulators, summary_rows
            )

        print("\nStage-level Geo confidence check (all source ranks):")
        for row in summary_rows:
            if row["source_rank"] != 0 or row["region"] != "all":
                continue
            print(
                "  {stage}: visible={visible_mean:.4f} occluded={occluded_mean:.4f} "
                "gap={mean_gap:.4f} AUC={roc_auc_visible:.4f} AP={average_precision_visible:.4f} "
                "mod_gap={mod_mean_gap:.4f} -> {grade}".format(
                    grade=diagnostic_grade(row["roc_auc_visible"], row["mean_gap"]),
                    **row,
                )
            )
        print("\nSaved:")
        print(" ", summary_csv)
        print(" ", pair_csv)
        print(" ", curves_csv)
        print(" ", histograms_csv)
        if not args.no_plot:
            print(" ", os.path.join(args.outdir, "summary.png"))
    finally:
        model_module.geometric_confidence = original_geometric_confidence


if __name__ == "__main__":
    main()
