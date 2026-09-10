"""GT-aligned large-disparity/occlusion visualization for two checkpoints.

The script evaluates only one requested DTU scan/reference/light sample.  It
does not consume or overwrite dtu_test fusion outputs.  Region masks come from
the training-format GT depths and Cameras/train calibration, so the displayed
absolute errors and improvement map are pixel aligned.
"""

import argparse
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
        description="Compare two checkpoints on one GT-aligned DTU large-disparity/occlusion sample."
    )
    parser.add_argument("--base_model_type", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--base_label", default="baseline")
    parser.add_argument("--base_geo_alpha", type=float, default=0.3)
    parser.add_argument("--method_model_type", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--method_ckpt", required=True)
    parser.add_argument("--method_label", default="proposed")
    parser.add_argument("--method_geo_alpha", type=float, default=0.3)

    parser.add_argument("--testpath", required=True,
                        help="DTU training root containing Rectified/, Depths/ and Cameras/.")
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--scan", required=True, help="e.g. scan9")
    parser.add_argument("--view", required=True, type=int, help="Reference view id, e.g. 0")
    parser.add_argument("--light", type=int, default=3, choices=range(7))
    parser.add_argument("--eval_nviews", type=int, default=5,
                        help="Total model input views, including the reference.")
    parser.add_argument("--region_nviews", type=int, default=5,
                        help="Fixed total views used to define disparity/occlusion masks.")
    parser.add_argument("--outdir", default="./outputs/diagnostics_dtu_yao")

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
    parser.add_argument("--error_max", type=float, default=20.0,
                        help="Upper color limit for absolute error in mm.")
    parser.add_argument("--improvement_max", type=float, default=10.0,
                        help="Symmetric color limit for baseline_error-method_error in mm.")
    parser.add_argument("--tile_width", type=int, default=300)
    parser.add_argument("--crop_tile_width", type=int, default=640,
                        help="Display width for crop tiles. Larger than tile_width to make local details visible.")
    parser.add_argument("--crop", action="append", default=None,
                        help="Optional x,y,width,height crop. Repeat for multiple crops.")
    parser.add_argument("--save_crops", action="store_true",
                        help="Save automatic/manual crop visualizations.")
    parser.add_argument("--save_region_crops", action="store_true",
                        help="Save one automatic crop visualization for each region mask.")
    parser.add_argument("--auto_crop_width", type=int, default=80)
    parser.add_argument("--auto_crop_height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


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


def infer_checkpoint(args, sample, model_type, checkpoint, geo_alpha):
    model = nn.DataParallel(build_model(args, model_type=model_type, geo_alpha=geo_alpha)).cuda()
    load_checkpoint(model, checkpoint)
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


def color_occlusion(ratio, valid):
    return color_map(ratio, valid, 0.0, 1.0, cv2.COLORMAP_TURBO)


def color_binary(mask, color=(255, 255, 255)):
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = np.asarray(color, dtype=np.uint8)
    return image


def color_improvement(improvement, valid, limit):
    """Diverging map: red means proposed is better, blue means worse."""
    normalized = np.clip(improvement / max(limit, 1e-6), -1.0, 1.0)
    magnitude = np.abs(normalized)[..., None]
    white = np.full((*improvement.shape, 3), 255.0, dtype=np.float32)
    positive = np.zeros_like(white)
    positive[..., 0] = 255.0  # RGB red
    negative = np.zeros_like(white)
    negative[..., 2] = 255.0  # RGB blue
    endpoint = np.where((normalized >= 0.0)[..., None], positive, negative)
    image = white * (1.0 - magnitude) + endpoint * magnitude
    image = np.clip(image, 0, 255).astype(np.uint8)
    image[~valid] = 0
    return image


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


def hard_region_overlay(rgb, large_disp_occ, boundary_occ, occ_majority):
    """Consistent overlay colors for hard regions."""
    image = overlay_region(rgb, large_disp_occ, color=(255, 40, 40), alpha=0.50)
    image = overlay_region(image, boundary_occ, color=(40, 220, 80), alpha=0.45)
    image = draw_region_contour(image, large_disp_occ, color=(255, 40, 40), thickness=2)
    image = draw_region_contour(image, boundary_occ, color=(40, 220, 80), thickness=2)
    image = draw_region_contour(image, occ_majority, color=(255, 230, 0), thickness=2)
    return image


def large_disp_occ_overlay(rgb, large_disp_occ, occ_majority):
    """Overlay only the main large-disparity/occlusion target and majority occlusion."""
    image = overlay_region(rgb, large_disp_occ, color=(255, 40, 40), alpha=0.50)
    image = draw_region_contour(image, large_disp_occ, color=(255, 40, 40), thickness=2)
    image = draw_region_contour(image, occ_majority, color=(255, 230, 0), thickness=2)
    return image


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


def save_clean_panel(output_dir, prefix, name, image):
    panel_path = os.path.join(output_dir, "{}_{:s}.png".format(prefix, name))
    Image.fromarray(image.astype(np.uint8)).save(panel_path)
    return panel_path


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


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else float("nan")


def save_region_crop(args, output_dir, prefix, region_name, region_mask, valid,
                     rgb, depth_gt, base_error, method_error, improvement):
    if not region_mask.any():
        return
    crop = automatic_crop(region_mask, args.auto_crop_width, args.auto_crop_height)
    crop_valid = crop_array(valid, crop)
    crop_region = crop_array(region_mask, crop)
    crop_row = row([
        tile(draw_region_contour(crop_array(rgb, crop), crop_region),
             "{} RGB".format(region_name), args.crop_tile_width),
        tile(color_depth(crop_array(depth_gt, crop), crop_valid, args),
             "GT depth", args.crop_tile_width),
        tile(draw_region_contour(color_error(crop_array(base_error, crop), crop_valid, args), crop_region),
             "{} error".format(args.base_label), args.crop_tile_width),
        tile(draw_region_contour(color_error(crop_array(method_error, crop), crop_valid, args), crop_region),
             "{} error".format(args.method_label), args.crop_tile_width),
        tile(draw_region_contour(
            color_improvement(crop_array(improvement, crop), crop_valid, args.improvement_max),
            crop_region),
            "Delta: red proposed better", args.crop_tile_width),
    ])
    crop_path = os.path.join(output_dir, "{}_{}_crop01.png".format(prefix, region_name))
    Image.fromarray(crop_row).save(crop_path)
    print("Saved:", crop_path, "crop=", crop)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This visualization script requires CUDA.")
    if args.eval_nviews < 2 or args.region_nviews < 2:
        raise ValueError("eval_nviews and region_nviews must both be at least 2.")

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

    print("Running baseline inference:", args.base_label)
    base_depth, _base_conf = infer_checkpoint(
        args, sample, args.base_model_type, args.base_ckpt, args.base_geo_alpha
    )
    print("Running proposed inference:", args.method_label)
    method_depth, _method_conf = infer_checkpoint(
        args, sample, args.method_model_type, args.method_ckpt, args.method_geo_alpha
    )

    depth_gt = to_2d(sample["depth"])
    gt_mask = to_2d(sample["mask"]) > 0.5
    rgb = read_reference_rgb(args)
    if rgb.shape[:2] != depth_gt.shape:
        rgb = cv2.resize(rgb, (depth_gt.shape[1], depth_gt.shape[0]), interpolation=cv2.INTER_AREA)
    valid = (
        gt_mask & np.isfinite(depth_gt)
        & np.isfinite(base_depth) & np.isfinite(method_depth)
    )
    base_error = np.abs(base_depth - depth_gt)
    method_error = np.abs(method_depth - depth_gt)
    improvement = base_error - method_error

    region_src_views = all_src_views[:args.region_nviews - 1]
    geometry = geometry_region_maps(
        args.testpath, args.scan, ref_view, region_src_views, depth_gt,
        occ_abs_tol=args.occ_abs_tol, occ_rel_tol=args.occ_rel_tol,
    )
    boundary = boundary_mask(depth_gt, valid, args.boundary_pct)
    large_disparity = percentile_mask(geometry["disparity"], valid, args.large_disp_pct)
    occluded_any = valid & geometry["occluded_any"]
    occluded_majority = valid & geometry["occluded_majority"]
    target = large_disparity & occluded_any
    boundary_and_occluded = boundary & occluded_any
    region_masks = {
        "full": valid,
        "boundary": boundary,
        "large_disparity": large_disparity,
        "occluded_any": occluded_any,
        "occluded_majority": occluded_majority,
        "large_disp_and_occluded": target,
        "boundary_and_occluded": boundary_and_occluded,
    }

    target_overlay = hard_region_overlay(rgb, target, boundary_and_occluded, occluded_majority)
    target_overlay_no_boundary = large_disp_occ_overlay(rgb, target, occluded_majority)
    base_error_plain = color_error(base_error, valid, args)
    method_error_plain = color_error(method_error, valid, args)
    improvement_plain = color_improvement(improvement, valid, args.improvement_max)
    base_error_image = draw_region_contour(color_error(base_error, valid, args), target, color=(255, 255, 0))
    method_error_image = draw_region_contour(color_error(method_error, valid, args), target, color=(255, 255, 0))
    improvement_image = draw_region_contour(
        color_improvement(improvement, valid, args.improvement_max), target, color=(255, 255, 0)
    )
    occlusion_ratio_image = color_occlusion(
        geometry["occlusion_ratio"], valid & (geometry["comparable_count"] > 0)
    )

    panels = [
        ("01_reference_rgb", rgb, "Reference RGB"),
        ("02_gt_depth", color_depth(depth_gt, valid, args), "GT depth"),
        ("03_large_disp_mask", color_binary(large_disparity),
         "Large disparity top {:g}%".format(100 - args.large_disp_pct)),
        ("04_occlusion_ratio", occlusion_ratio_image, "Source occlusion ratio"),
        ("05_hard_region_overlay", target_overlay,
         "Red: large-disp&occ; green: boundary&occ; yellow: occ-majority"),
        ("06_{}_depth".format(args.base_label), color_depth(base_depth, valid, args),
         "{} depth".format(args.base_label)),
        ("07_{}_depth".format(args.method_label), color_depth(method_depth, valid, args),
         "{} depth".format(args.method_label)),
        ("08_{}_abs_error".format(args.base_label), base_error_image,
         "{} abs error; yellow: large-disp&occ".format(args.base_label)),
        ("09_{}_abs_error".format(args.method_label), method_error_image,
         "{} abs error; yellow: large-disp&occ".format(args.method_label)),
        ("10_error_delta", improvement_image,
         "Error delta: red better; yellow: large-disp&occ"),
        ("11_boundary_mask", color_binary(boundary),
         "Boundary top {:g}%".format(args.boundary_pct)),
        ("12_boundary_and_occ_mask", color_binary(boundary_and_occluded),
         "Boundary & occ-any"),
        ("13_occluded_any_mask", color_binary(occluded_any), "Occluded any src"),
        ("14_occluded_majority_mask", color_binary(occluded_majority), "Occluded majority"),
        ("15_large_disp_and_occ_mask", color_binary(target), "Large-disp & occ-any"),
        ("16_hard_region_overlay_no_boundary", target_overlay_no_boundary,
         "Red: large-disp&occ; yellow: occ-majority"),
        ("17_{}_abs_error_no_contour".format(args.base_label), base_error_plain,
         "{} abs error".format(args.base_label)),
        ("18_{}_abs_error_no_contour".format(args.method_label), method_error_plain,
         "{} abs error".format(args.method_label)),
        ("19_error_delta_no_contour", improvement_plain, "Error delta: red proposed better"),
    ]
    first_row = row([tile(image, title, args.tile_width) for _name, image, title in panels[:5]])
    second_row = row([tile(image, title, args.tile_width) for _name, image, title in panels[5:10]])
    third_row = row([tile(image, title, args.tile_width) for _name, image, title in panels[10:15]])
    fourth_row = row([tile(image, title, args.tile_width) for _name, image, title in panels[15:19]])

    output_dir = os.path.join(
        args.outdir,
        "{}_vs_{}".format(args.base_label, args.method_label),
    )
    os.makedirs(output_dir, exist_ok=True)
    prefix = "{}_view{:04d}_light{}".format(args.scan, args.view, args.light)
    overview_path = os.path.join(output_dir, prefix + "_overview.png")
    Image.fromarray(grid([first_row, second_row, third_row, fourth_row])).save(overview_path)
    for panel_name, panel_image, panel_title in panels:
        save_clean_panel(output_dir, prefix, panel_name, panel_image)

    if args.save_crops:
        if args.crop:
            crops = [parse_crop(text, depth_gt.shape) for text in args.crop]
        else:
            auto_mask = target if target.any() else (large_disparity & boundary)
            crops = [automatic_crop(auto_mask, args.auto_crop_width, args.auto_crop_height)]

        for crop_index, crop in enumerate(crops, start=1):
            crop_valid = crop_array(valid, crop)
            crop_target = crop_array(target, crop)
            crop_row = row([
                tile(draw_region_contour(crop_array(rgb, crop), crop_target), "RGB crop", args.crop_tile_width),
                tile(color_depth(crop_array(depth_gt, crop), crop_valid, args), "GT depth", args.crop_tile_width),
                tile(draw_region_contour(color_error(crop_array(base_error, crop), crop_valid, args), crop_target),
                     "{} error".format(args.base_label), args.crop_tile_width),
                tile(draw_region_contour(color_error(crop_array(method_error, crop), crop_valid, args), crop_target),
                     "{} error".format(args.method_label), args.crop_tile_width),
                tile(draw_region_contour(
                    color_improvement(crop_array(improvement, crop), crop_valid, args.improvement_max), crop_target),
                    "Delta: red proposed better", args.crop_tile_width),
            ])
            crop_path = os.path.join(output_dir, prefix + "_crop{:02d}.png".format(crop_index))
            Image.fromarray(crop_row).save(crop_path)
            print("Saved:", crop_path, "crop=", crop)

    if args.save_region_crops:
        for region_name in (
            "boundary",
            "large_disparity",
            "occluded_any",
            "occluded_majority",
            "large_disp_and_occluded",
            "boundary_and_occluded",
        ):
            save_region_crop(
                args, output_dir, prefix, region_name, region_masks[region_name], valid,
                rgb, depth_gt, base_error, method_error, improvement,
            )

    print("Saved:", overview_path)
    print("Region summary (same pixels for both models):")
    for name in (
        "full",
        "boundary",
        "large_disparity",
        "occluded_any",
        "occluded_majority",
        "large_disp_and_occluded",
        "boundary_and_occluded",
    ):
        mask = region_masks[name]
        print("  {:26s} pixels={:8d} {}={:8.4f} {}={:8.4f} delta={:8.4f}".format(
            name, int(mask.sum()), args.base_label, masked_mean(base_error, mask),
            args.method_label, masked_mean(method_error, mask), masked_mean(improvement, mask),
        ))


if __name__ == "__main__":
    main()
