from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.return_visibility_gt = bool(kwargs.get("return_visibility_gt", False))
        self.visibility_downsample = int(kwargs.get("visibility_downsample", 8))
        if self.visibility_downsample < 1:
            raise ValueError("visibility_downsample must be at least 1")

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "Cameras/pair.txt"
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    # ImageNet stats in RGB order (PIL opens RGB)
    _IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMGNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    @classmethod
    def normalize(cls, img):
        """ImageNet centre — same preprocessing as original Vis-MVSNet."""
        return (img - cls._IMGNET_MEAN.reshape(3, 1, 1)) / (cls._IMGNET_STD.reshape(3, 1, 1) + 1e-8)

    def read_depth(self, filename):
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    @staticmethod
    def _mask_to_2d(mask):
        if mask.ndim == 3:
            mask = mask.mean(axis=2)
        return (mask > 0.05).astype(np.float32)

    def _downsample_visibility_map(self, array, image_shape, is_mask=False):
        image_height, image_width = image_shape
        out_height = max(1, image_height // self.visibility_downsample)
        out_width = max(1, image_width // self.visibility_downsample)
        pil_mode = 'L' if is_mask else 'F'
        source = (array > 0.5).astype(np.uint8) * 255 if is_mask else array.astype(np.float32)
        nearest = getattr(Image, 'Resampling', Image).NEAREST
        resized = Image.fromarray(source, mode=pil_mode).resize(
            (out_width, out_height), resample=nearest)
        result = np.asarray(resized, dtype=np.float32)
        if is_mask:
            result = (result > 127).astype(np.float32)
        return result

    @staticmethod
    def _augment_color(img):
        """Random brightness, contrast, saturation jitter (numpy, in-place-ish).

        Args:
            img: [H, W, 3] float32 in [0, 1].
        Returns:
            augmented image, same shape and range.
        """
        # brightness
        b = np.random.uniform(-0.1, 0.1)
        img = img + b
        # contrast
        c = np.random.uniform(0.9, 1.1)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c + mean
        # saturation
        s = np.random.uniform(0.9, 1.1)
        gray = img.mean(axis=2, keepdims=True)
        img = gray + s * (img - gray)
        return np.clip(img, 0., 1.)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        visibility_depths = []
        visibility_masks = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img_np = self.read_img(img_filename)
            image_shape = img_np.shape[:2]
            if self.mode == 'train':
                img_np = self._augment_color(img_np)
            imgs.append(img_np)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

            if self.return_visibility_gt:
                view_depth = depth if i == 0 else self.read_depth(depth_filename)
                view_mask = mask if i == 0 else self.read_img(mask_filename)
                view_mask = self._mask_to_2d(view_mask)
                visibility_depths.append(self._downsample_visibility_map(
                    view_depth, image_shape, is_mask=False))
                visibility_masks.append(self._downsample_visibility_map(
                    view_mask, image_shape, is_mask=True))

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        imgs = self.normalize(imgs)
        proj_matrices = np.stack(proj_matrices)

        sample = {"imgs": imgs,
                  "proj_matrices": proj_matrices,
                  "depth": depth,
                  "depth_values": depth_values,
                  "mask": mask}
        if self.return_visibility_gt:
            sample["visibility_depths"] = np.stack(visibility_depths)
            sample["visibility_masks"] = np.stack(visibility_masks)
        return sample
