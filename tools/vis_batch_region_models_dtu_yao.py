"""Batch GT-aligned region visualization for multiple DTU checkpoints.

This script evaluates one DTU training-format sample with multiple models.  It
uses one shared set of large-disparity/occlusion/boundary masks, saves one PNG
overview and one crop PNG per model, and writes all per-region metrics into a
single CSV.

Example model spec:
    --model vis_view5:vis:./checkpoints/dtu/vis_view5/best_2mm.ckpt
    --model geo_a03:geo:./checkpoints/dtu/vis_geo_view5/best_2mm.ckpt:0.3
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
from PIL import Image, ImageDraw


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from datasets import find_dataset_def  # noqa: E402
from eval_region_metrics_dtu_yao import (  # noqa: E402
    MODEL_SPECS,
    boundary_mask,
    build_model,
    geometry_region_maps,
    load_checkpoint,
    percentile_mask,
    to_2d,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch per-model GT-aligned DTU region visualization and CSV metrics."
    )
    parser.add_argument(
        "--model", action="append", required=True,
        help=(
            "Model spec label:model_type:checkpoint[:geo_alpha]. Repeat for each model. "
            "model_type must be one of {}.".format(",".join(sorted(MODEL_SPECS)))
        ),
    )
    parser.add_argument(
        "--baseline_label", default=None,
        help="Optional label used to add delta_abs and rel_improve_pct columns in CSV.",
    )
    parser.add_argument("--testpath", required=True,
                        help="DTU training root containing Rectified/, Depths/ and Cameras/.")
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--scan", required=True, help="e.g. scan9")
    parser.add_argument("--view", required=True, type=int, help="Reference view id, e.g. 0")
    parser.add_argument("--light", type=int, default=3, choices=range(7))
    parser.add_argument("--include_light_in_name", action="store_true",
                        help="Include light id in output directories and filenames.")
    parser.add_argument("--eval_nviews", type=int, default=5)
    parser.add_argument("--region_nviews", type=int, default=5)
    parser.add_argument("--outdir", default="./outputs/diagnostics_batch_dtu_yao")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--no_images", action="store_true",
                        help="Only write the metrics CSV; skip all PNG visualization outputs.")

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

    parser.add_argument("--boundary_pct", type=float, default=10.0)
    parser.add_argument("--large_disp_pct", type=float, default=80.0)
    parser.add_argument("--occ_abs_tol", type=float, default=2.0)
    parser.add_argument("--occ_rel_tol", type=float, default=0.01)
    parser.add_argument("--depth_min", type=float, default=425.0)
    parser.add_argument("--depth_max", type=float, default=935.0)
    parser.add_argument("--error_max", type=float, default=20.0)
    parser.add_argument("--tile_width", type=int, default=320)
    parser.add_argument("--crop_tile_width", type=int, default=640,
                        help="Display width for crop tiles. Larger than tile_width to make local details visible.")
    parser.add_argument("--crop", action="append", default=None,
                        help="Optional x,y,width,height crop. Repeat for multiple crops.")
    parser.add_argument("--auto_crop_width", type=int, default=80)
    parser.add_argument("--auto_crop_height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def parse_model_spec(text):
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            "--model must be label:model_type:checkpoint[:geo_alpha], got {!r}".format(text)
        )
    label, model_type, checkpoint = parts[:3]
    if not label:
        raise ValueError("Model label cannot be empty: {!r}".format(text))
    if model_type not in MODEL_SPECS:
        raise ValueError("Unknown model_type {!r}; choices are {}".format(
            model_type, sorted(MODEL_SPECS)
        ))
    geo_alpha = float(parts[3]) if len(parts) == 4 else 0.3
    return {
        "label": label,
        "model_type": model_type,
        "checkpoint": checkpoint,
        "geo_alpha": geo_alpha,
    }


def find_sample(dataset, scan, view, light):
    for index, meta in enumerate(dataset.metas):
        item_scan, item_light, ref_view, _src_views = meta
        if item_scan == scan and item_light == light and ref_view == view:
            return index, meta
    raise ValueError(
        "Sample scan={} view={} light={} is not present in {}".format(
            scan, view, light, dataset.listfile
        )
    )


def tensorize_sample(sample):
    result = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).cuda()
        elif torch.is_tensor(value):
            result[key] = value.unsqueeze(0).cuda()
        else:
            result[key] = value
    return result


def infer_checkpoint(args, sample, spec):
    print("Running inference:", spec["label"])
    model = nn.DataParallel(
        build_model(args, model_type=spec["model_type"], geo_alpha=spec["geo_alpha"])
    ).cuda()
    load_checkpoint(model, spec["checkpoint"])
    model.eval()
    sample_cuda = tensorize_sample(sample)
    with torch.no_grad():
        result = model(
            sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]
        )
        outputs, _, confidence_maps = result[:3]
        target_size = sample_cuda["depth"].shape[-2:]
        depth = F.interpolate(
            outputs[-1][0].unsqueeze(1), target_size,
            mode="bilinear", align_corners=False,
        ).squeeze(1)[0]
        confidence = F.interpolate(
            confidence_maps[-1], target_size,
            mode="bilinear", align_corners=False,
        ).squeeze(1)[0]
        depth = to_2d(depth.cpu().numpy())
        confidence = to_2d(confidence.cpu().numpy())
    del result, outputs, confidence_maps, sample_cuda, model
    torch.cuda.empty_cache()
    return depth, confidence


def read_reference_rgb(args):
    filename = os.path.join(
        args.testpath,
        "Rectified",
        args.scan + "_train",
        "rect_{:03d}_{}_r5000.png".format(args.view + 1, args.light),
    )
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    return np.asarray(Image.open(filename).convert("RGB"))


def color_map(values, valid, vmin, vmax, colormap):
    scale = max(vmax - vmin, 1e-6)
    normalized = np.clip((values - vmin) / scale, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def color_depth(depth, valid, args):
    return color_map(depth, valid, args.depth_min, args.depth_max, cv2.COLORMAP_JET)


def color_error(error, valid, args):
    return color_map(error, valid, 0.0, args.error_max, cv2.COLORMAP_INFERNO)


def color_confidence(confidence, valid):
    finite = np.isfinite(confidence) & valid
    if not finite.any():
        return np.zeros((*confidence.shape, 3), dtype=np.uint8)
    vmax = max(float(np.percentile(confidence[finite], 99.0)), 1e-6)
    return color_map(confidence, finite, 0.0, vmax, cv2.COLORMAP_VIRIDIS)


def color_binary(mask, color=(255, 255, 255)):
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = np.asarray(color, dtype=np.uint8)
    return image


def color_occlusion(ratio, valid):
    return color_map(ratio, valid, 0.0, 1.0, cv2.COLORMAP_TURBO)


def overlay_region(rgb, mask, color=(255, 40, 40), alpha=0.55):
    output = rgb.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    output[mask] = output[mask] * (1.0 - alpha) + tint * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_region_contour(image, mask, color=(255, 255, 0), thickness=2):
    result = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (color[2], color[1], color[0]), thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_tile(image, width):
    height, original_width = image.shape[:2]
    target_height = max(1, int(round(height * width / float(original_width))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def add_title(image, title):
    title_height = 30
    pil_image = Image.fromarray(image.astype(np.uint8))
    canvas = Image.new("RGB", (pil_image.width, pil_image.height + title_height), (20, 20, 20))
    canvas.paste(pil_image, (0, title_height))
    ImageDraw.Draw(canvas).text((7, 8), title, fill=(245, 245, 245))
    return np.asarray(canvas)


def tile(image, title, width):
    return add_title(resize_tile(image, width), title)


def row(images):
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] < max_height:
            pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
            image = np.concatenate((image, pad), axis=0)
        padded.append(image)
    return np.concatenate(padded, axis=1)


def grid(rows):
    max_width = max(image.shape[1] for image in rows)
    padded = []
    for image in rows:
        if image.shape[1] < max_width:
            pad = np.zeros((image.shape[0], max_width - image.shape[1], 3), dtype=np.uint8)
            image = np.concatenate((image, pad), axis=1)
        padded.append(image)
    return np.concatenate(padded, axis=0)


def parse_crop(text, image_shape):
    try:
        x, y, width, height = [int(value.strip()) for value in text.split(",")]
    except (TypeError, ValueError):
        raise ValueError("--crop must be x,y,width,height, got {!r}".format(text))
    image_height, image_width = image_shape
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))
    return x, y, width, height


def automatic_crop(mask, width, height):
    image_height, image_width = mask.shape
    width = max(1, min(width, image_width))
    height = max(1, min(height, image_height))
    score = cv2.boxFilter(
        mask.astype(np.float32), -1, (width, height), normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    _, _, _, max_location = cv2.minMaxLoc(score)
    center_x, center_y = max_location
    x = min(max(center_x - width // 2, 0), image_width - width)
    y = min(max(center_y - height // 2, 0), image_height - height)
    return x, y, width, height


def crop_array(array, crop):
    x, y, width, height = crop
    return array[y:y + height, x:x + width]


def sample_prefix(args):
    prefix = "{}_view{:04d}".format(args.scan, args.view)
    if args.include_light_in_name:
        prefix += "_light{}".format(args.light)
    return prefix


def metric_row(args, spec, region_name, mask, depth, depth_gt, baseline_abs=None):
    pixels = int(mask.sum())
    if pixels == 0:
        abs_error = float("nan")
        acc2 = float("nan")
        acc4 = float("nan")
        acc8 = float("nan")
    else:
        error = np.abs(depth - depth_gt)
        region_error = error[mask]
        abs_error = float(region_error.mean())
        acc2 = float((region_error <= 2.0).mean())
        acc4 = float((region_error <= 4.0).mean())
        acc8 = float((region_error <= 8.0).mean())
    delta_abs = ""
    rel_improve_pct = ""
    if baseline_abs is not None and np.isfinite(abs_error):
        delta_abs = baseline_abs - abs_error
        rel_improve_pct = 100.0 * delta_abs / max(baseline_abs, 1e-6)
    return {
        "scan": args.scan,
        "view": args.view,
        "light": args.light,
        "label": spec["label"],
        "model_type": spec["model_type"],
        "geo_alpha": spec["geo_alpha"],
        "eval_nviews": args.eval_nviews,
        "region_nviews": args.region_nviews,
        "region": region_name,
        "pixels": pixels,
        "abs": abs_error,
        "acc2": acc2,
        "acc4": acc4,
        "acc8": acc8,
        "delta_abs_vs_baseline": delta_abs,
        "rel_improve_pct_vs_baseline": rel_improve_pct,
    }


def write_model_images(args, output_dir, spec, rgb, depth_gt, depth, confidence,
                       valid, masks, crops):
    label = spec["label"]
    error = np.abs(depth - depth_gt)
    target = masks["large_disp_and_occluded"]
    target_overlay = overlay_region(rgb, target)
    target_overlay = draw_region_contour(
        target_overlay, masks["occluded_majority"], color=(255, 255, 0), thickness=2
    )

    first_row = row([
        tile(rgb, "Reference RGB", args.tile_width),
        tile(color_depth(depth_gt, valid, args), "GT depth", args.tile_width),
        tile(color_binary(masks["large_disparity"]), "Large disparity top {:g}%".format(
            100.0 - args.large_disp_pct
        ), args.tile_width),
        tile(color_occlusion(masks["occlusion_ratio"], valid & masks["comparable"]),
             "Source occlusion ratio", args.tile_width),
        tile(target_overlay, "Red: large-disp & occ; yellow: occ-majority", args.tile_width),
    ])
    second_row = row([
        tile(color_depth(depth, valid, args), "{} depth".format(label), args.tile_width),
        tile(draw_region_contour(color_error(error, valid, args), target),
             "{} abs error".format(label), args.tile_width),
        tile(color_confidence(confidence, valid), "{} confidence".format(label), args.tile_width),
        tile(draw_region_contour(color_depth(depth, target, args), target),
             "{} target depth".format(label), args.tile_width),
        tile(draw_region_contour(color_error(error, target, args), target),
             "{} target error".format(label), args.tile_width),
    ])

    prefix = "{}_{}".format(sample_prefix(args), label)
    overview_path = os.path.join(output_dir, prefix + "_overview.png")
    Image.fromarray(grid([first_row, second_row])).save(overview_path)
    print("Saved:", overview_path)

    for crop_index, crop in enumerate(crops, start=1):
        crop_valid = crop_array(valid, crop)
        crop_target = crop_array(target, crop)
        crop_row = row([
            tile(draw_region_contour(crop_array(rgb, crop), crop_target), "RGB crop", args.crop_tile_width),
            tile(color_depth(crop_array(depth_gt, crop), crop_valid, args), "GT depth", args.crop_tile_width),
            tile(color_depth(crop_array(depth, crop), crop_valid, args), "{} depth".format(label), args.crop_tile_width),
            tile(draw_region_contour(color_error(crop_array(error, crop), crop_valid, args), crop_target),
                 "{} error".format(label), args.crop_tile_width),
            tile(color_confidence(crop_array(confidence, crop), crop_valid), "{} confidence".format(label), args.crop_tile_width),
        ])
        crop_path = os.path.join(output_dir, prefix + "_crop{:02d}.png".format(crop_index))
        Image.fromarray(crop_row).save(crop_path)
        print("Saved:", crop_path, "crop=", crop)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This visualization script requires CUDA.")
    if args.eval_nviews < 2 or args.region_nviews < 2:
        raise ValueError("eval_nviews and region_nviews must both be at least 2.")

    specs = [parse_model_spec(text) for text in args.model]
    labels = [spec["label"] for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Model labels must be unique: {}".format(labels))
    if args.baseline_label is not None and args.baseline_label not in labels:
        raise ValueError("--baseline_label {!r} is not among model labels {}".format(
            args.baseline_label, labels
        ))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    dataset_class = find_dataset_def("dtu_yao")
    dataset = dataset_class(
        args.testpath, args.testlist, "test", args.eval_nviews,
        args.numdepth, args.interval_scale,
    )
    sample_index, meta = find_sample(dataset, args.scan, args.view, args.light)
    sample = dataset[sample_index]
    _scan, _light, ref_view, all_src_views = meta

    depth_gt = to_2d(sample["depth"])
    gt_mask = to_2d(sample["mask"]) > 0.5
    rgb = None
    if not args.no_images:
        rgb = read_reference_rgb(args)
        if rgb.shape[:2] != depth_gt.shape:
            rgb = cv2.resize(rgb, (depth_gt.shape[1], depth_gt.shape[0]), interpolation=cv2.INTER_AREA)

    region_src_views = all_src_views[:args.region_nviews - 1]
    geometry = geometry_region_maps(
        args.testpath, args.scan, ref_view, region_src_views, depth_gt,
        occ_abs_tol=args.occ_abs_tol, occ_rel_tol=args.occ_rel_tol,
    )

    predictions = {}
    for spec in specs:
        depth, confidence = infer_checkpoint(args, sample, spec)
        predictions[spec["label"]] = {
            "depth": depth,
            "confidence": confidence,
        }

    finite_predictions = np.ones_like(depth_gt, dtype=bool)
    for prediction in predictions.values():
        finite_predictions &= np.isfinite(prediction["depth"])
    valid = gt_mask & np.isfinite(depth_gt) & finite_predictions

    masks = {
        "full": valid,
        "boundary": boundary_mask(depth_gt, valid, args.boundary_pct),
        "large_disparity": percentile_mask(geometry["disparity"], valid, args.large_disp_pct),
        "occluded_any": valid & geometry["occluded_any"],
        "occluded_majority": valid & geometry["occluded_majority"],
        "occlusion_ratio": geometry["occlusion_ratio"],
        "comparable": geometry["comparable_count"] > 0,
    }
    masks["large_disp_and_occluded"] = masks["large_disparity"] & masks["occluded_any"]
    masks["boundary_and_occluded"] = masks["boundary"] & masks["occluded_any"]

    crops = []
    if not args.no_images:
        if args.crop:
            crops = [parse_crop(text, depth_gt.shape) for text in args.crop]
        else:
            auto_mask = masks["large_disp_and_occluded"]
            if not auto_mask.any():
                auto_mask = masks["large_disparity"] & masks["boundary"]
            crops = [automatic_crop(auto_mask, args.auto_crop_width, args.auto_crop_height)]

    sample_dir = os.path.join(args.outdir, sample_prefix(args))
    os.makedirs(sample_dir, exist_ok=True)

    baseline_abs_by_region = {}
    if args.baseline_label is not None:
        baseline_depth = predictions[args.baseline_label]["depth"]
        for region_name in (
            "full", "boundary", "large_disparity", "occluded_any",
            "occluded_majority", "large_disp_and_occluded", "boundary_and_occluded",
        ):
            mask = masks[region_name]
            if mask.any():
                baseline_abs_by_region[region_name] = float(np.abs(
                    baseline_depth - depth_gt
                )[mask].mean())

    rows_out = []
    for spec in specs:
        prediction = predictions[spec["label"]]
        if not args.no_images:
            write_model_images(
                args, sample_dir, spec, rgb, depth_gt, prediction["depth"],
                prediction["confidence"], valid, masks, crops,
            )
        for region_name in (
            "full", "boundary", "large_disparity", "occluded_any",
            "occluded_majority", "large_disp_and_occluded", "boundary_and_occluded",
        ):
            rows_out.append(metric_row(
                args, spec, region_name, masks[region_name], prediction["depth"], depth_gt,
                baseline_abs=baseline_abs_by_region.get(region_name),
            ))

    csv_path = args.out_csv or os.path.join(
        sample_dir, "{}_metrics.csv".format(sample_prefix(args))
    )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "scan", "view", "light", "label", "model_type", "geo_alpha",
        "eval_nviews", "region_nviews", "region", "pixels", "abs",
        "acc2", "acc4", "acc8",
    ]
    if args.baseline_label is not None:
        fieldnames.extend([
            "delta_abs_vs_baseline",
            "rel_improve_pct_vs_baseline",
        ])
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    print("Saved metrics:", csv_path)

    print("Region summary:")
    for item in rows_out:
        if args.baseline_label is None:
            print(
                "  {label:32s} {region:24s} pixels={pixels:8d} abs={abs:8.4f} "
                "acc2={acc2:6.4f} acc4={acc4:6.4f} acc8={acc8:6.4f}".format(**item)
            )
        else:
            print(
                "  {label:24s} {region:24s} pixels={pixels:8d} abs={abs:8.4f} "
                "acc2={acc2:6.4f} acc4={acc4:6.4f} acc8={acc8:6.4f} "
                "delta={delta_abs_vs_baseline}".format(**item)
            )


if __name__ == "__main__":
    main()
