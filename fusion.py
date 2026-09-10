"""
Depth map fusion + mask application for Vis-MVSNet.

1. photometric + geometric consistency filtering → masks
2. PLY point cloud export
3. Apply photo/geo/final masks to depth maps → filtered depth .pfm/.png

Usage:
    python fusion.py
    python fusion.py --outdir ./outputs/other_path
"""
import argparse
import os
import numpy as np
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement
from datasets.data_io import read_pfm, save_pfm


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = np_img[:-16, :]  # DTU eval crop: remove bottom 16 pixels
    return np_img


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


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

    return depth_reprojected, xy_reprojected[0].reshape([height, width]).astype(np.float32), \
           xy_reprojected[1].reshape([height, width]).astype(np.float32), x_src, y_src


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


def _save_depth_png(path, depth, dmin=425., dmax=935.):
    depth_norm = np.clip((depth - dmin) / (dmax - dmin), 0., 1.)
    depth_col = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_col)


def filter_depth(scan_folder, out_folder, plyfilename, conf_threshold=0.8, geo_threshold=3):
    pair_file = os.path.join(scan_folder, "pair.txt")
    vertexs = []
    vertex_colors = []
    depth_min, depth_max = 425., 935.

    pair_data = read_pair_file(pair_file)

    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder,
                                              'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder,
                                           'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > conf_threshold

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
        geo_mask = geo_mask_sum >= geo_threshold
        final_mask = np.logical_and(photo_mask, geo_mask)

        # ---- save masks ----
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)),
                    (photo_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
                    (geo_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)),
                    (final_mask.astype(np.uint8) * 255))

        # ---- apply masks to depth map ----
        for suffix, mask in [("photo", photo_mask), ("geo", geo_mask), ("final", final_mask)]:
            d_out = os.path.join(out_folder, f"depth_{suffix}")
            os.makedirs(d_out, exist_ok=True)
            d_masked = ref_depth_est * mask.astype(np.float32)
            fname = '{:0>8}'.format(ref_view)
            save_pfm(os.path.join(d_out, fname + '.pfm'), d_masked)
            _save_depth_png(os.path.join(d_out, fname + '.png'), d_masked, depth_min, depth_max)

        print("processing {}, ref-view{:0>2}, photo/geo/final:{:.3f}/{:.3f}/{:.3f}".format(
            scan_folder, ref_view, photo_mask.mean(), geo_mask.mean(), final_mask.mean()))

        # ---- point cloud (subsampled) ----
        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        x_sub = x[1:-16:4, 1::4]
        y_sub = y[1:-16:4, 1::4]
        depth_sub = depth_est_averaged[1:-16:4, 1::4]
        mask_sub = final_mask[1:-16:4, 1::4]
        color_sub = ref_img[1:-16:4, 1::4, :]

        valid = mask_sub
        x, y, depth = x_sub[valid], y_sub[valid], depth_sub[valid]
        color = color_sub[valid]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)

    vertexs_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_colors_dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertexs_arr = np.array([tuple(v) for v in vertexs], dtype=vertexs_dtype)
    vertex_colors_arr = np.array([tuple(v) for v in vertex_colors], dtype=vertex_colors_dtype)

    vertex_all = np.empty(len(vertexs_arr), vertexs_arr.dtype.descr + vertex_colors_arr.dtype.descr)
    for prop in vertexs_arr.dtype.names:
        vertex_all[prop] = vertexs_arr[prop]
    for prop in vertex_colors_arr.dtype.names:
        vertex_all[prop] = vertex_colors_arr[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving point cloud to", plyfilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis-MVSNet depth fusion + mask')
    parser.add_argument('--testpath', default='/home/disk_10T/lzh_data/dtu_test')
    parser.add_argument('--testlist', default='lists/dtu/test.txt')
    parser.add_argument('--outdir', default='./outputs/dtu/d192_convnext_view2/best_2mm_view2')
    parser.add_argument('--conf_threshold', type=float, default=0.2)
    parser.add_argument('--geo_threshold', type=int, default=2)
    args = parser.parse_args()

    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    for scan in scans:
        scan_id = int(scan[4:])
        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        plyfilename = os.path.join(args.outdir,
                                   'vismvsnet_{:0>3}_l3.ply'.format(scan_id))
        filter_depth(scan_folder, out_folder, plyfilename,
                     args.conf_threshold, args.geo_threshold)
