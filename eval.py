"""
Vis-MVSNet evaluation script.

Two-step pipeline:
  1. save_depth:  run inference -> save depth maps (PFM), confidence maps (PFM),
                  error maps (PFM+PNG, if GT available)
  2. filter_depth: photometric + geometric consistency -> fused point cloud (PLY)

Confidence uses Vis-MVSNet's prob_map: sum of probabilities within +-2 depth bins
of the soft-argmin, NOT max probability.

Usage:
    python eval.py --dataset=dtu_yao_eval --testpath <path> --testlist <list>
                   --loadckpt <ckpt> --outdir <dir> --display
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
import sys
from datasets import find_dataset_def
from models import VisMVSModel
from utils import *
from datasets.data_io import read_pfm, save_pfm
import cv2
from PIL import Image

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Vis-MVSNet evaluation')
parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--nviews', type=int, default=3,
                    help='total views (default: trinocular, one reference plus two sources)')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--gtpath', default=None, help='path to GT depth maps (e.g. .../Depths)')
parser.add_argument('--no_fusion', action='store_true', help='skip point cloud fusion (depth-only eval)')
parser.add_argument('--metrics_only', action='store_true',
                    help='only print depth metrics; do not save PFM, PNG, masks, or point clouds')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

# Vis-MVSNet specific
parser.add_argument('--vismode', type=str, default='soft',
                    choices=['soft', 'hard', 'average', 'uwta', 'maxpool'])
parser.add_argument('--stage1_dnum', type=int, default=48)
parser.add_argument('--stage1_iscale', type=int, default=4)
parser.add_argument('--stage2_dnum', type=int, default=32)
parser.add_argument('--stage2_iscale', type=int, default=2)
parser.add_argument('--stage3_dnum', type=int, default=16)
parser.add_argument('--stage3_iscale', type=int, default=1)
parser.add_argument('--disable_adaptive_search', action='store_true')
parser.add_argument('--disable_hypothesis_visibility', action='store_true')
parser.add_argument('--disable_boundary_refine', action='store_true')
parser.add_argument('--global_candidate_ratio', type=float, default=0.25)
parser.add_argument('--secondary_candidate_ratio', type=float, default=0.25)

args = parser.parse_args()
if args.nviews != 3:
    parser.error("the current experiment phase is fixed to three total views")
if args.global_candidate_ratio + args.secondary_candidate_ratio >= 1.0:
    parser.error("global and secondary candidate ratios must sum to less than 1")
if args.dataset == 'dtu_yao' and not args.metrics_only:
    print("DTU training layout detected: enabling --metrics_only automatically.")
    args.metrics_only = True
print("argv:", sys.argv[1:])
print_args(args)


# =============================================================================
# Helpers for camera I/O
# =============================================================================
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    return np.array(img, dtype=np.float32) / 255.


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


def _save_depth_pfm_and_png(outdir, fname, depth, depth_min=425., depth_max=935.):
    """Save depth as PFM and colour-mapped PNG."""
    pfm_path = os.path.join(outdir, fname.format('.pfm'))
    png_path = os.path.join(outdir, fname.format('.png'))
    os.makedirs(os.path.dirname(pfm_path), exist_ok=True)
    save_pfm(pfm_path, depth)
    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0., 1.)
    depth_col = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(png_path, depth_col)


def _save_error_pfm_and_png(outdir, fname, error):
    """Save error map as PFM and colour-mapped PNG (hot colormap)."""
    pfm_path = os.path.join(outdir, fname.format('.pfm'))
    png_path = os.path.join(outdir, fname.format('.png'))
    os.makedirs(os.path.dirname(pfm_path), exist_ok=True)
    save_pfm(pfm_path, error)
    error_clip = np.clip(error, 0., 20.) / 20.
    error_col = cv2.applyColorMap((error_clip * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(png_path, error_col)




# =============================================================================
# Depth map saving (step 1)
# =============================================================================
def save_depth():
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test",
                              args.nviews, args.numdepth, args.interval_scale)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                               num_workers=4, drop_last=False)

    model = VisMVSModel(
        mode=args.vismode,
        stage1_depth_num=args.stage1_dnum,
        stage1_interval_scale=args.stage1_iscale,
        stage2_depth_num=args.stage2_dnum,
        stage2_interval_scale=args.stage2_iscale,
        stage3_depth_num=args.stage3_dnum,
        stage3_interval_scale=args.stage3_iscale,
        use_adaptive_search=not args.disable_adaptive_search,
        use_hypothesis_visibility=not args.disable_hypothesis_visibility,
        use_boundary_refine=not args.disable_boundary_refine,
        global_candidate_ratio=args.global_candidate_ratio,
        secondary_candidate_ratio=args.secondary_candidate_ratio,
    )
    model = nn.DataParallel(model)
    model.cuda()

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, weights_only=True)
    model.load_state_dict(state_dict['model'])
    model.eval()

    H_orig, W_orig = 1184, 1600  # DTU eval crop size

    # metric accumulators (only on GT-available views)
    metrics = {"abs_sum": 0.0, "pixels": 0, "lt2": 0, "lt4": 0, "lt8": 0}

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            outputs, final_depth, conf_maps, auxiliary = model(
                sample_cuda["imgs"], sample_cuda["proj_matrices"],
                sample_cuda["depth_values"])

            filenames = sample.get("filename")
            if filenames is not None:
                scan_name = filenames[0].split('/')[0] if len(filenames) > 0 else '?'
            else:
                if not args.metrics_only:
                    raise ValueError(
                        "datasets without output filenames can only be used with --metrics_only")
                scan_name = 'training-layout sample'
            print('Iter {}/{}  scan {}'.format(
                batch_idx, len(TestImgLoader), scan_name))

            B = final_depth.shape[0]

            # Native stage-3 output is resized to GT resolution by the metric helper.
            depth_s3_native = final_depth.squeeze(1).cpu().numpy()  # [B, H_native, W_native]

            if args.metrics_only:
                del sample_cuda, outputs, final_depth, conf_maps, auxiliary
                if "depth" in sample and "mask" in sample:
                    depth_gt = sample["depth"].cpu().numpy()
                    valid_mask = sample["mask"].cpu().numpy() > 0.5
                    for b in range(B):
                        ret = _metrics_from_arrays(
                            depth_s3_native[b], depth_gt[b], valid_mask[b])
                        for key in metrics:
                            metrics[key] += ret[key]
                elif filenames is not None:
                    for b in range(B):
                        ret = _compute_depth_metrics(
                            filenames[b].format('', ''), depth_s3_native[b])
                        if ret is not None:
                            for key in metrics:
                                metrics[key] += ret[key]
                continue

            # ---- upsample stage depths to original resolution for saving ----
            depth_s1 = F.interpolate(
                outputs[0][0].unsqueeze(1),
                size=(H_orig, W_orig),
                mode='bilinear', align_corners=False).squeeze(1)  # [B, H, W]
            depth_s2 = F.interpolate(
                outputs[1][0].unsqueeze(1),
                size=(H_orig, W_orig),
                mode='bilinear', align_corners=False).squeeze(1)
            depth_s3 = F.interpolate(
                final_depth,
                size=(H_orig, W_orig),
                mode='bilinear', align_corners=False).squeeze(1)

            # ---- upsample stage-3 confidence only ----
            conf_s3 = F.interpolate(
                conf_maps[2],
                size=(H_orig, W_orig),
                mode='bilinear', align_corners=False).squeeze(1)

            depth_s1 = depth_s1.cpu().numpy()
            depth_s2 = depth_s2.cpu().numpy()
            depth_s3 = depth_s3.cpu().numpy()
            conf_s3 = conf_s3.cpu().numpy()

            del sample_cuda, outputs, final_depth, conf_maps, auxiliary

            # ---- save per-sample ----
            for b in range(B):
                fname = filenames[b]

                # --- stage depths: PFM + colour PNG ---
                _save_depth_pfm_and_png(args.outdir,
                                        fname.format('depth_stage1', '{}'),
                                        depth_s1[b])
                _save_depth_pfm_and_png(args.outdir,
                                        fname.format('depth_stage2', '{}'),
                                        depth_s2[b])
                _save_depth_pfm_and_png(args.outdir,
                                        fname.format('depth_est', '{}'),
                                        depth_s3[b])

                # --- confidence: stage 3 PFM only ---
                cfd = fname.format('confidence', '{}')
                os.makedirs(os.path.dirname(os.path.join(args.outdir, cfd.format('.pfm'))), exist_ok=True)
                save_pfm(os.path.join(args.outdir, cfd.format('.pfm')), conf_s3[b])

                # --- error maps (if GT available) ---
                ret = _save_error_maps(fname.format('', ''), depth_s1[b], depth_s2[b], depth_s3[b],
                                     depth_s3_native[b])
                if ret is not None:
                    metrics["abs_sum"] += ret["abs_sum"]
                    metrics["pixels"] += ret["pixels"]
                    metrics["lt2"] += ret["lt2"]
                    metrics["lt4"] += ret["lt4"]
                    metrics["lt8"] += ret["lt8"]

    # ---- print aggregated metrics ----
    if metrics["pixels"] > 0:
        total = metrics["pixels"]
        abs_err = metrics["abs_sum"] / total
        lt2 = metrics["lt2"] / total * 100
        lt4 = metrics["lt4"] / total * 100
        lt8 = metrics["lt8"] / total * 100
        print("\n" + "=" * 60)
        print("Stage 3 (final) evaluation on {} valid pixels:".format(total))
        print("  abs_depth_error : {:.4f} mm".format(abs_err))
        print("  <2mm  (accuracy): {:.2f}%".format(lt2))
        print("  <4mm  (accuracy): {:.2f}%".format(lt4))
        print("  <8mm  (accuracy): {:.2f}%".format(lt8))
        print("=" * 60 + "\n")
    else:
        print("No valid GT pixels were found; check DATASET, DATAPATH, TESTLIST, and GTPATH.")


def _load_gt_depth(fname_blank):
    """Load the GT depth corresponding to a dataset output filename."""
    parts = fname_blank.split('/')
    scan = parts[0]
    try:
        view_id = int(parts[-1])
    except ValueError:
        return None

    gt_root = args.gtpath if args.gtpath else os.path.join(args.testpath, 'Depths')
    gt_path = os.path.join(gt_root, scan + '_train',
                           'depth_map_{:04d}.pfm'.format(view_id))
    if not os.path.exists(gt_path):
        return None

    depth_gt, _ = read_pfm(gt_path)
    return scan, view_id, depth_gt


def _compute_depth_metrics(fname_blank, prediction):
    """Compute metrics at the prediction's native resolution without saving files."""
    gt_data = _load_gt_depth(fname_blank)
    if gt_data is None:
        return None

    _, _, depth_gt = gt_data
    depth_min, depth_max = 425., 935.
    gt_mask = (depth_gt > depth_min) & (depth_gt < depth_max)

    return _metrics_from_arrays(prediction, depth_gt, gt_mask)


def _metrics_from_arrays(prediction, depth_gt, valid_mask):
    """Compute metrics after resizing a prediction to the GT resolution."""
    while valid_mask.ndim > 2:
        valid_mask = valid_mask.mean(axis=-1) > 0.5
    if prediction.shape != depth_gt.shape:
        prediction = cv2.resize(prediction, (depth_gt.shape[1], depth_gt.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    valid_err = np.abs(prediction - depth_gt)[valid_mask]
    return {
        "abs_sum": float(valid_err.sum()),
        "pixels": valid_err.size,
        "lt2": int((valid_err < 2).sum()),
        "lt4": int((valid_err < 4).sum()),
        "lt8": int((valid_err < 8).sum()),
    }


def _save_error_maps(fname_blank, d1, d2, d3, d3_native=None):
    """Try to load GT depth and save error maps for each stage.

    Args:
        d1, d2, d3: depths upsampled to 1184×1600 (for visualization)
        d3_native:  native stage3 output before upsampling (for accurate metrics)

    Returns:
        dict with keys abs_sum, pixels, lt2, lt4, lt8  (or None if no GT).
    """
    gt_data = _load_gt_depth(fname_blank)
    if gt_data is None:
        return None
    scan, view_id, depth_gt = gt_data
    depth_min, depth_max = 425., 935.

    # ---- metrics at native pred size ----
    pred_for_metric = d3_native if d3_native is not None else d3
    metric_values = _compute_depth_metrics(fname_blank, pred_for_metric)

    # ---- error maps for visualization (upsample GT to pred size) ----
    gt_up = cv2.resize(depth_gt, (d1.shape[1], d1.shape[0]),
                       interpolation=cv2.INTER_NEAREST) if depth_gt.shape != d1.shape else depth_gt
    gt_mask_up = (gt_up > depth_min) & (gt_up < depth_max)

    err1 = np.abs(d1 - gt_up); err1[~gt_mask_up] = 0.
    fname_err1 = scan + '/error_stage1/' + '{:0>8}'.format(view_id) + '{}'
    _save_error_pfm_and_png(args.outdir, fname_err1, err1)

    err2 = np.abs(d2 - gt_up); err2[~gt_mask_up] = 0.
    fname_err2 = scan + '/error_stage2/' + '{:0>8}'.format(view_id) + '{}'
    _save_error_pfm_and_png(args.outdir, fname_err2, err2)

    err3 = np.abs(d3 - gt_up); err3[~gt_mask_up] = 0.
    fname_err3 = scan + '/error_stage3/' + '{:0>8}'.format(view_id) + '{}'
    _save_error_pfm_and_png(args.outdir, fname_err3, err3)

    return metric_values


# =============================================================================
# Geometric consistency filtering (step 2)
# =============================================================================
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                         depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref,
        depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename):
    pair_file = os.path.join(scan_folder, "pair.txt")
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder,
                                              'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder,
                                           'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > 0.8

        all_srcview_depth_ests = []

        geo_mask_sum = 0
        for src_view in src_views:
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_depth_est = read_pfm(os.path.join(out_folder,
                                                  'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            if src_depth_est.shape != ref_depth_est.shape:
                continue

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics,
                src_depth_est, src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)),
                    (photo_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
                    (geo_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)),
                    (final_mask.astype(np.uint8) * 255))

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{:.3f}/{:.3f}/{:.3f}".format(
            scan_folder, ref_view, photo_mask.mean(), geo_mask.mean(), final_mask.mean()))

        if args.display:
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 935)
            cv2.imshow('ref_depth * final_mask',
                       ref_depth_est * final_mask.astype(np.float32) / 935)
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[1:-16:4, 1::4, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)

    # write PLY
    vertexs_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_colors_dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertexs_arr = np.array([tuple(v) for v in vertexs], dtype=vertexs_dtype)
    vertex_colors_arr = np.array([tuple(v) for v in vertex_colors], dtype=vertex_colors_dtype)

    from plyfile import PlyData, PlyElement
    vertex_all = np.empty(len(vertexs_arr), vertexs_arr.dtype.descr + vertex_colors_arr.dtype.descr)
    for prop in vertexs_arr.dtype.names:
        vertex_all[prop] = vertexs_arr[prop]
    for prop in vertex_colors_arr.dtype.names:
        vertex_all[prop] = vertex_colors_arr[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving point cloud to", plyfilename)


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    if args.metrics_only:
        print("=" * 60)
        print("Running metrics-only evaluation (no files will be saved) ...")
        print("=" * 60)
        save_depth()
        print("Done (--metrics_only).")
        sys.exit(0)

    # Step 1: save all depth maps and confidence maps
    print("=" * 60)
    print("Step 1: saving depth maps and confidence maps ...")
    print("=" * 60)
    save_depth()

    if args.no_fusion:
        print("Done (--no_fusion, skipping point cloud fusion).")
        sys.exit(0)

    # Step 2: filter and fuse into point cloud
    print("=" * 60)
    print("Step 2: filtering and fusing ...")
    print("=" * 60)
    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    for scan in scans:
        scan_id = int(scan[4:])
        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        plyfilename = os.path.join(args.outdir,
                                   'vismvsnet_{:0>3}_l3.ply'.format(scan_id))
        filter_depth(scan_folder, out_folder, plyfilename)

    print("Done!")
