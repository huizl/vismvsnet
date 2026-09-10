"""Region-wise DTU metrics on the *training* dtu_yao evaluation pipeline.

This script deliberately runs the model again instead of reading ``dtu_test``
fusion outputs.  It therefore uses exactly the same Rectified images,
``Cameras/train`` files, GT depth maps and valid masks as train*.py.

The ``full_train_mean`` diagnostic is a per-image mean and should reproduce
the ``avg_test_scalars`` depth metrics printed by the corresponding training
script (up to insignificant floating-point differences).  Only interpret the
other region rows after that diagnostic agrees with the training log.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import find_dataset_def  # noqa: E402
from datasets.data_io import read_pfm  # noqa: E402


MODEL_SPECS = {
    "vis": ("models.vismvsnet", {}),
    "geo": ("models.vismvsnet_geo", {"use_geo": True}),
    "conv": ("models.vismvsnet_convnext", {}),
    "conv_geo": ("models.vismvsnet_conv_geo", {"use_geo": True}),
    "pid": ("models.vismvsnet_pid", {"use_pid": True}),
    "pid_geo": ("models.vismvsnet_pid_geo", {"use_pid": True, "use_geo": True}),
    "conv_pid": ("models.vismvsnet_conv_pid", {"use_pid": True}),
    "conv_pid_geo": (
        "models.vismvsnet_conv_pid_geo",
        {"use_pid": True, "use_geo": True},
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Region metrics using the GT-aligned dtu_yao test pipeline."
    )
    parser.add_argument("--model_type", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--loadckpt", required=True, help="Checkpoint saved by the matching train*.py.")
    parser.add_argument("--label", default=None, help="Printed model label; defaults to --model_type.")
    parser.add_argument("--testpath", required=True,
                        help="DTU training root containing Rectified/, Depths/ and Cameras/.")
    parser.add_argument("--testlist", required=True, help="e.g. lists/dtu/val.txt")
    parser.add_argument("--out_csv", default=None, help="Optional CSV destination.")

    # Keep model construction identical to the training commands.
    parser.add_argument("--vismode", default="soft",
                        choices=["soft", "hard", "average", "uwta", "maxpool"])
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--stage1_dnum", type=int, default=48)
    parser.add_argument("--stage1_iscale", type=int, default=4)
    parser.add_argument("--stage2_dnum", type=int, default=32)
    parser.add_argument("--stage2_iscale", type=int, default=2)
    parser.add_argument("--stage3_dnum", type=int, default=16)
    parser.add_argument("--stage3_iscale", type=int, default=1)
    parser.add_argument("--geo_alpha", type=float, default=0.3,
                        help="Base Geo weight. Keep the training value unless running a test-time ablation.")
    parser.add_argument("--stage1_geo_alpha", type=float, default=None,
                        help="Optional inference-only stage1 Geo weight; defaults to --geo_alpha.")
    parser.add_argument("--stage2_geo_alpha", type=float, default=None,
                        help="Optional inference-only stage2 Geo weight; defaults to --geo_alpha.")
    parser.add_argument("--stage3_geo_alpha", type=float, default=None,
                        help="Optional inference-only stage3 Geo weight; defaults to --geo_alpha.")
    parser.add_argument("--geo_confidence_mode", default="roundtrip",
                        choices=["roundtrip", "occlusion"],
                        help="Geo confidence used by the base geo model during inference.")
    parser.add_argument("--geo_occ_abs_tol", type=float, default=2.0,
                        help="Absolute predicted source-depth tolerance for the occlusion gate.")
    parser.add_argument("--geo_occ_rel_tol", type=float, default=0.01,
                        help="Relative predicted source-depth tolerance for the occlusion gate.")
    parser.add_argument("--geo_occ_temperature", type=float, default=1.0,
                        help="Positive transition temperature for the soft occlusion gate.")

    # train*.py currently creates its test set with 5 views irrespective of
    # training --nviews.  Retain that as the default to reproduce its logs.
    parser.add_argument("--eval_nviews", type=int, default=5,
                        help="Views at test time; default 5 matches the current train*.py scripts.")
    parser.add_argument("--region_nviews", type=int, default=5,
                        help="Total views used only to define disparity/occlusion masks. Keep fixed when comparing eval_nviews.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size; default 4 matches the current train*.py scripts.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--light", type=int, default=None,
                        help="Optional DTU light index in [0, 6]. Omit to evaluate all seven lights.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional debug limit. One sample is one scan/ref/light tuple.")
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--boundary_pct", type=float, default=10.0,
                        help="Top percentage of GT depth-gradient pixels.")
    parser.add_argument("--near_pct", type=float, default=20.0,
                        help="Nearest percentage of valid GT-depth pixels.")
    parser.add_argument("--large_disp_pct", type=float, default=80.0,
                        help="Pixels at or above this percentile of real ref-to-src displacement.")
    parser.add_argument("--tail_pct", type=float, default=10.0,
                        help="Top percentage of absolute-depth-error pixels.")
    parser.add_argument("--conf_drop_pct", type=float, default=0.0,
                        help="Also report results after dropping this lowest confidence percentage.")
    parser.add_argument("--occ_abs_tol", type=float, default=2.0,
                        help="Absolute source-depth tolerance in mm for GT occlusion classification.")
    parser.add_argument("--occ_rel_tol", type=float, default=0.01,
                        help="Relative source-depth tolerance for GT occlusion classification.")
    return parser.parse_args()


def build_model(args, model_type=None, geo_alpha=None):
    model_type = model_type or args.model_type
    module_name, extra_kwargs = MODEL_SPECS[model_type]
    module = __import__(module_name, fromlist=["VisMVSModel"])
    kwargs = dict(
        mode=args.vismode,
        stage1_depth_num=args.stage1_dnum,
        stage1_interval_scale=args.stage1_iscale,
        stage2_depth_num=args.stage2_dnum,
        stage2_interval_scale=args.stage2_iscale,
        stage3_depth_num=args.stage3_dnum,
        stage3_interval_scale=args.stage3_iscale,
    )
    kwargs.update(extra_kwargs)
    if "use_geo" in kwargs:
        kwargs["geo_alpha"] = args.geo_alpha if geo_alpha is None else geo_alpha
    if model_type == "geo":
        kwargs.update(
            geo_confidence_mode=args.geo_confidence_mode,
            geo_occ_abs_tol=args.geo_occ_abs_tol,
            geo_occ_rel_tol=args.geo_occ_rel_tol,
            geo_occ_temperature=args.geo_occ_temperature,
        )
    return module.VisMVSModel(**kwargs)


def resolve_stage_geo_alphas(args):
    overrides = (
        args.stage1_geo_alpha,
        args.stage2_geo_alpha,
        args.stage3_geo_alpha,
    )
    is_geo_model = "geo" in args.model_type
    if not is_geo_model:
        if any(value is not None for value in overrides):
            raise ValueError("Stage-specific Geo alpha overrides require a geo model_type.")
        return None

    values = tuple(args.geo_alpha if value is None else value for value in overrides)
    for stage_name, value in zip(("stage1", "stage2", "stage3"), values):
        if not 0.0 <= value <= 1.0:
            raise ValueError("{} Geo alpha must be in [0, 1], got {}.".format(stage_name, value))
    return values


def apply_stage_geo_alphas(model, stage_geo_alphas):
    if stage_geo_alphas is None:
        return
    for stage_name, value in zip(("stage1", "stage2", "stage3"), stage_geo_alphas):
        stage = getattr(model, stage_name, None)
        if stage is None or not hasattr(stage, "geo_alpha"):
            raise AttributeError("Model does not expose {}.geo_alpha for the requested ablation.".format(
                stage_name
            ))
        stage.geo_alpha = value


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Allow a checkpoint saved without DataParallel to be used as well.
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        else:
            state_dict = {"module." + key: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)


def to_cuda(sample):
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
            for key, value in sample.items()}


def to_2d(array):
    """Convert one DataLoader item to a 2-D HxW numpy array."""
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 3:
        # The official DTU depth mask is normally one-channel.  This keeps the
        # script usable if a copy was saved as RGB without affecting that case.
        array = array.mean(axis=-1)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D depth/mask/confidence array, got {}".format(array.shape))
    return array.astype(np.float32, copy=False)


def boundary_mask(depth_gt, valid, pct):
    if not valid.any():
        return valid.copy()
    filled = depth_gt.copy()
    filled[~valid] = np.median(depth_gt[valid])
    grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    threshold = np.percentile(gradient[valid], 100.0 - pct)
    return valid & (gradient >= threshold)


def near_depth_mask(depth_gt, valid, pct):
    if not valid.any():
        return valid.copy()
    threshold = np.percentile(depth_gt[valid], pct)
    return valid & (depth_gt <= threshold)


def percentile_mask(values, valid, percentile):
    if not valid.any():
        return valid.copy()
    threshold = np.percentile(values[valid], percentile)
    return valid & (values >= threshold)


def top_error_mask(error, valid, pct):
    return percentile_mask(error, valid, 100.0 - pct)


def conf_keep_mask(confidence, valid, drop_pct):
    if drop_pct <= 0 or not valid.any():
        return valid.copy()
    threshold = np.percentile(confidence[valid], drop_pct)
    return valid & (confidence >= threshold)


def read_train_camera(datapath, view_id):
    filename = os.path.join(datapath, "Cameras", "train", "{:08d}_cam.txt".format(view_id))
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(4, 4)
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape(3, 3)
    return intrinsics, extrinsics


def read_dtu_depth(datapath, scan, view_id):
    filename = os.path.join(
        datapath, "Depths", scan + "_train", "depth_map_{:04d}.pfm".format(view_id)
    )
    if not os.path.exists(filename):
        raise FileNotFoundError("DTU GT depth not found: {}".format(filename))
    depth, _ = read_pfm(filename)
    return np.asarray(depth, dtype=np.float32)


def read_dtu_mask(datapath, scan, view_id, target_shape=None):
    filename = os.path.join(
        datapath, "Depths", scan + "_train", "depth_visual_{:04d}.png".format(view_id)
    )
    if not os.path.exists(filename):
        return None
    mask = np.asarray(Image.open(filename), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.mean(axis=2)
    mask = mask > 10.0
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def geometry_region_maps(datapath, scan, ref_view, src_views, depth,
                         occ_abs_tol=2.0, occ_rel_tol=0.01):
    """Build fixed GT/camera disparity and source-occlusion maps.

    Occlusion is defined only for source pixels with valid source GT.  A
    reference 3-D point is occluded in a source when its projected source Z is
    behind the sampled visible source surface by more than
    ``max(occ_abs_tol, occ_rel_tol * source_depth)``.  Out-of-image points are
    deliberately not counted as occlusion.
    """
    if not src_views:
        raise ValueError("At least one source view is required to define geometry regions.")

    height, width = depth.shape
    ref_k, ref_ext = read_train_camera(datapath, ref_view)
    y, x = np.indices((height, width), dtype=np.float32)
    homogeneous = np.stack((x.reshape(-1), y.reshape(-1), np.ones(height * width, dtype=np.float32)))
    rays_ref = np.linalg.inv(ref_k).astype(np.float32) @ homogeneous
    xyz_ref = rays_ref * depth.reshape(1, -1)
    xyz_ref_h = np.vstack((xyz_ref, np.ones((1, xyz_ref.shape[1]), dtype=np.float32)))

    max_disparity = np.zeros(height * width, dtype=np.float32)
    occluded_count = np.zeros(height * width, dtype=np.uint16)
    comparable_count = np.zeros(height * width, dtype=np.uint16)
    for src_view in src_views:
        src_k, src_ext = read_train_camera(datapath, src_view)
        transform = src_ext @ np.linalg.inv(ref_ext)
        xyz_src = transform[:3, :] @ xyz_ref_h
        projected = src_k @ xyz_src
        z = projected[2]
        safe_z = np.where(np.abs(z) > 1e-6, z, 1e-6)
        src_x = projected[0] / safe_z
        src_y = projected[1] / safe_z
        displacement = np.sqrt((src_x - homogeneous[0]) ** 2 + (src_y - homogeneous[1]) ** 2)
        max_disparity = np.maximum(max_disparity, displacement.astype(np.float32))

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
        comparable = in_bounds & sampled_mask & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        tolerance = np.maximum(occ_abs_tol, occ_rel_tol * sampled_depth)
        occluded = comparable & (projected_depth > sampled_depth + tolerance)
        comparable_count += comparable.reshape(-1).astype(np.uint16)
        occluded_count += occluded.reshape(-1).astype(np.uint16)

    comparable_count = comparable_count.reshape(height, width)
    occluded_count = occluded_count.reshape(height, width)
    occlusion_ratio = np.zeros((height, width), dtype=np.float32)
    comparable = comparable_count > 0
    occlusion_ratio[comparable] = (
        occluded_count[comparable].astype(np.float32)
        / comparable_count[comparable].astype(np.float32)
    )
    occluded_any = comparable & (occluded_count > 0)
    occluded_majority = comparable & (2 * occluded_count >= comparable_count)
    return {
        "disparity": max_disparity.reshape(height, width),
        "occlusion_ratio": occlusion_ratio,
        "occluded_any": occluded_any,
        "occluded_majority": occluded_majority,
        "comparable_count": comparable_count,
    }


def reprojection_disparity(datapath, ref_view, src_views, depth, scan=None):
    """Backward-compatible disparity helper used by earlier local tools."""
    if scan is None:
        # Disparity itself does not need source GT; retain the old lightweight
        # behavior for callers that do not have a scan identifier.
        height, width = depth.shape
        ref_k, ref_ext = read_train_camera(datapath, ref_view)
        y, x = np.indices((height, width), dtype=np.float32)
        homogeneous = np.stack((x.reshape(-1), y.reshape(-1), np.ones(height * width, dtype=np.float32)))
        rays_ref = np.linalg.inv(ref_k).astype(np.float32) @ homogeneous
        xyz_ref = rays_ref * depth.reshape(1, -1)
        xyz_ref_h = np.vstack((xyz_ref, np.ones((1, xyz_ref.shape[1]), dtype=np.float32)))
        max_disparity = np.zeros(height * width, dtype=np.float32)
        for src_view in src_views:
            src_k, src_ext = read_train_camera(datapath, src_view)
            xyz_src = (src_ext @ np.linalg.inv(ref_ext))[:3, :] @ xyz_ref_h
            projected = src_k @ xyz_src
            safe_z = np.where(np.abs(projected[2]) > 1e-6, projected[2], 1e-6)
            src_x = projected[0] / safe_z
            src_y = projected[1] / safe_z
            displacement = np.sqrt((src_x - homogeneous[0]) ** 2 + (src_y - homogeneous[1]) ** 2)
            max_disparity = np.maximum(max_disparity, displacement.astype(np.float32))
        return max_disparity.reshape(height, width)
    return geometry_region_maps(datapath, scan, ref_view, src_views, depth)["disparity"]


def metrics(error, mask):
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return {"pixels": 0, "abs": np.nan, "acc2": np.nan, "acc4": np.nan, "acc8": np.nan}
    values = error[mask]
    return {
        "pixels": pixel_count,
        "abs": float(values.mean()),
        "acc2": float((values < 2.0).mean()),
        "acc4": float((values < 4.0).mean()),
        "acc8": float((values < 8.0).mean()),
    }


def add_metrics(accumulator, region, result):
    if result["pixels"] == 0:
        return
    if region not in accumulator:
        accumulator[region] = {"pixels": 0, "abs_sum": 0.0, "lt2": 0, "lt4": 0, "lt8": 0}
    item = accumulator[region]
    count = result["pixels"]
    item["pixels"] += count
    item["abs_sum"] += result["abs"] * count
    item["lt2"] += result["acc2"] * count
    item["lt4"] += result["acc4"] * count
    item["lt8"] += result["acc8"] * count


def summarize(accumulator):
    rows = []
    for region, values in accumulator.items():
        count = values["pixels"]
        rows.append({
            "region": region,
            "pixels": count,
            "abs": values["abs_sum"] / count,
            "acc2": values["lt2"] / count,
            "acc4": values["lt4"] / count,
            "acc8": values["lt8"] / count,
        })
    return rows


def print_rows(label, rows):
    print("{:<24} {:<24} {:>12} {:>10} {:>10} {:>10} {:>10}".format(
        "label", "region", "pixels", "abs", "acc2", "acc4", "acc8"))
    for row in rows:
        print("{:<24} {:<24} {:>12d} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            label, row["region"], row["pixels"], row["abs"], row["acc2"], row["acc4"], row["acc8"]
        ))


def save_csv(label, rows, filename, args):
    parent = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent, exist_ok=True)
    with open(filename, "w", newline="") as f:
        fieldnames = [
            "label", "model_type", "geo_alpha", "stage1_geo_alpha", "stage2_geo_alpha",
            "stage3_geo_alpha", "geo_confidence_mode", "geo_occ_abs_tol",
            "geo_occ_rel_tol", "geo_occ_temperature", "eval_nviews", "region_nviews", "light",
            "region", "pixels", "abs", "acc2", "acc4", "acc8",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(
                label=label,
                model_type=args.model_type,
                geo_alpha=args.geo_alpha if "geo" in args.model_type else "",
                stage1_geo_alpha=args.effective_stage_geo_alphas[0]
                if args.effective_stage_geo_alphas is not None else "",
                stage2_geo_alpha=args.effective_stage_geo_alphas[1]
                if args.effective_stage_geo_alphas is not None else "",
                stage3_geo_alpha=args.effective_stage_geo_alphas[2]
                if args.effective_stage_geo_alphas is not None else "",
                geo_confidence_mode=args.geo_confidence_mode
                if "geo" in args.model_type else "",
                geo_occ_abs_tol=args.geo_occ_abs_tol
                if args.model_type == "geo" else "",
                geo_occ_rel_tol=args.geo_occ_rel_tol
                if args.model_type == "geo" else "",
                geo_occ_temperature=args.geo_occ_temperature
                if args.model_type == "geo" else "",
                eval_nviews=args.eval_nviews,
                region_nviews=args.region_nviews,
                light=args.light if args.light is not None else "all",
                **row
            ))


def main():
    args = parse_args()
    args.effective_stage_geo_alphas = resolve_stage_geo_alphas(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires CUDA, matching the project training/evaluation setup.")
    if args.eval_nviews < 2:
        raise ValueError("--eval_nviews must be at least 2 (one reference and one source).")
    if args.region_nviews < 2:
        raise ValueError("--region_nviews must be at least 2 (one reference and one source).")
    if not 0.0 < args.large_disp_pct < 100.0:
        raise ValueError("--large_disp_pct must be in (0, 100).")
    if args.occ_abs_tol < 0.0 or args.occ_rel_tol < 0.0:
        raise ValueError("Occlusion tolerances must be non-negative.")
    if args.geo_occ_abs_tol < 0.0 or args.geo_occ_rel_tol < 0.0:
        raise ValueError("Geo occlusion tolerances must be non-negative.")
    if args.geo_occ_temperature <= 0.0:
        raise ValueError("--geo_occ_temperature must be positive.")
    if args.geo_confidence_mode != "roundtrip" and args.model_type != "geo":
        raise ValueError("The occlusion confidence mode is currently implemented only for model_type=geo.")
    if args.light is not None and not 0 <= args.light <= 6:
        raise ValueError("--light must be in [0, 6].")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    dataset_class = find_dataset_def("dtu_yao")
    dataset = dataset_class(args.testpath, args.testlist, "test", args.eval_nviews,
                            args.numdepth, args.interval_scale)
    if args.light is not None:
        dataset.metas = [meta for meta in dataset.metas if int(meta[1]) == args.light]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False)

    model = nn.DataParallel(build_model(args)).cuda()
    load_checkpoint(model, args.loadckpt)
    apply_stage_geo_alphas(model.module, args.effective_stage_geo_alphas)
    model.eval()
    label = args.label or args.model_type

    region_accumulator = {}
    train_mean_sum = {"abs": 0.0, "acc2": 0.0, "acc4": 0.0, "acc8": 0.0}
    evaluated_samples = 0
    evaluated_batches = 0
    previous_geometry_key = None
    cached_geometry = None

    print("Evaluating {} with {} test views; region masks use {} total views; dataset samples: {}".format(
        label, args.eval_nviews, args.region_nviews, len(dataset)))
    if args.effective_stage_geo_alphas is not None:
        print("Geo inference alphas: stage1={:.3f}, stage2={:.3f}, stage3={:.3f}".format(
            *args.effective_stage_geo_alphas
        ))
        print("Geo confidence: mode={}, abs_tol={}, rel_tol={}, temperature={}".format(
            args.geo_confidence_mode, args.geo_occ_abs_tol,
            args.geo_occ_rel_tol, args.geo_occ_temperature
        ))
    print("DTU light:", args.light if args.light is not None else "all")
    with torch.no_grad():
        for batch_index, sample in enumerate(loader):
            first_sample_index = batch_index * args.batch_size
            if args.max_samples is not None and first_sample_index >= args.max_samples:
                break

            sample_cuda = to_cuda(sample)
            model_output = model(
                sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]
            )
            outputs, _, confidence_maps = model_output[:3]

            # This is deliberately the same tensor and interpolation used by
            # train*.py:test_sample(), rather than a saved eval.py depth map.
            depth_gt_tensor = sample_cuda["depth"]
            stage3_depth = outputs[-1][0]
            depth_est_full = F.interpolate(
                stage3_depth.unsqueeze(1), size=depth_gt_tensor.shape[-2:],
                mode="bilinear", align_corners=False
            ).squeeze(1)
            confidence_full = F.interpolate(
                confidence_maps[-1], size=depth_gt_tensor.shape[-2:],
                mode="bilinear", align_corners=False
            ).squeeze(1)

            current_batch_metrics = []
            for batch_item in range(depth_gt_tensor.shape[0]):
                sample_index = first_sample_index + batch_item
                if args.max_samples is not None and sample_index >= args.max_samples:
                    break

                depth_gt = to_2d(depth_gt_tensor[batch_item].cpu().numpy())
                depth_est = to_2d(depth_est_full[batch_item].cpu().numpy())
                confidence = to_2d(confidence_full[batch_item].cpu().numpy())
                gt_mask = to_2d(sample_cuda["mask"][batch_item].cpu().numpy()) > 0.5
                valid = gt_mask & np.isfinite(depth_gt) & np.isfinite(depth_est)
                error = np.abs(depth_est - depth_gt)

                # dtu_yao orders metadata as scan -> reference -> seven lights.
                # The geometry is invariant across the seven lights, so reuse the
                # expensive true-displacement map for those consecutive samples.
                scan, _light, ref_view, all_src_views = dataset.metas[sample_index]
                region_src_views = all_src_views[:args.region_nviews - 1]
                geometry_key = (scan, ref_view, tuple(region_src_views), depth_gt.shape)
                if geometry_key != previous_geometry_key:
                    cached_geometry = geometry_region_maps(
                        args.testpath, scan, ref_view, region_src_views, depth_gt,
                        occ_abs_tol=args.occ_abs_tol,
                        occ_rel_tol=args.occ_rel_tol,
                    )
                    previous_geometry_key = geometry_key

                full_result = metrics(error, valid)
                if full_result["pixels"] == 0:
                    continue
                current_batch_metrics.append(full_result)
                evaluated_samples += 1

                boundary = boundary_mask(depth_gt, valid, args.boundary_pct)
                large_disparity = percentile_mask(
                    cached_geometry["disparity"], valid, args.large_disp_pct
                )
                occluded_any = valid & cached_geometry["occluded_any"]
                occluded_majority = valid & cached_geometry["occluded_majority"]
                region_masks = {
                    "full": valid,
                    "boundary_top{:g}%".format(args.boundary_pct): boundary,
                    "near_depth_{:g}%".format(args.near_pct): near_depth_mask(
                        depth_gt, valid, args.near_pct
                    ),
                    "large_disp_top{:g}%".format(100.0 - args.large_disp_pct): large_disparity,
                    "occluded_any_src": occluded_any,
                    "occluded_majority": occluded_majority,
                    "large_disp_and_occ_any": large_disparity & occluded_any,
                    "large_disp_and_occ_majority": large_disparity & occluded_majority,
                    "boundary_and_occ_any": boundary & occluded_any,
                    "top_error_{:g}%".format(args.tail_pct): top_error_mask(
                        error, valid, args.tail_pct
                    ),
                }
                if args.conf_drop_pct > 0:
                    region_masks["conf_keep_{:g}%".format(100.0 - args.conf_drop_pct)] = conf_keep_mask(
                        confidence, valid, args.conf_drop_pct
                    )
                for region_name, region_mask in region_masks.items():
                    add_metrics(region_accumulator, region_name, metrics(error, region_mask))

            # train*.py averages each batch's per-image metric, then its
            # DictAverageMeter averages those batches.  Preserve that behavior
            # here so this sanity-check line is directly comparable.
            if current_batch_metrics:
                for name in train_mean_sum:
                    train_mean_sum[name] += float(np.mean([item[name] for item in current_batch_metrics]))
                evaluated_batches += 1

            if (batch_index + 1) % args.print_freq == 0 and current_batch_metrics:
                last_metric = current_batch_metrics[-1]
                print("[batch {}/{}; sample {}] full_abs={:.4f}".format(
                    batch_index + 1, len(loader), min(first_sample_index + args.batch_size, len(dataset)),
                    last_metric["abs"]
                ))

    if evaluated_samples == 0:
        raise RuntimeError("No samples were evaluated. Check --testpath and --testlist.")

    rows = summarize(region_accumulator)
    diagnostic = {
        "region": "full_train_mean",
        "pixels": evaluated_samples,
        "abs": train_mean_sum["abs"] / evaluated_batches,
        "acc2": train_mean_sum["acc2"] / evaluated_batches,
        "acc4": train_mean_sum["acc4"] / evaluated_batches,
        "acc8": train_mean_sum["acc8"] / evaluated_batches,
    }
    # Put the diagnostic first: it is the validity check for every other row.
    rows.insert(0, diagnostic)
    print_rows(label, rows)
    print("\nfull_train_mean is the checkpoint/data-pipeline sanity check. "
          "Compare it with avg_test_scalars before interpreting region rows.")
    if args.out_csv:
        save_csv(label, rows, args.out_csv, args)
        print("Saved CSV:", args.out_csv)


if __name__ == "__main__":
    main()
