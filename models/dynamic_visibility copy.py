import torch
import torch.nn.functional as F


def _pixel_grid(batch, height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(0, height, dtype=dtype, device=device),
        torch.arange(0, width, dtype=dtype, device=device),
        indexing='ij')
    xy1 = torch.stack((x.reshape(-1), y.reshape(-1), torch.ones(height * width, dtype=dtype, device=device)), dim=0)
    return xy1.unsqueeze(0).repeat(batch, 1, 1)


def _normalize_grid(xy, height, width):
    x = xy[:, 0]
    y = xy[:, 1]
    x_norm = x / max((width - 1) / 2, 1e-6) - 1
    y_norm = y / max((height - 1) / 2, 1e-6) - 1
    return torch.stack((x_norm, y_norm), dim=-1)


def project_depth(ref_depth, ref_proj, src_proj, height, width):
    """Project a reference depth map into a source image.

    Args:
        ref_depth: [B, H, W]
        ref_proj/src_proj: [B, 4, 4], same scale as the depth map.

    Returns:
        src_xy: [B, 2, H*W] source pixel coordinates.
        src_z: [B, 1, H*W] source camera depth.
        ref_xy1: [B, 3, H*W] reference homogeneous pixels.
    """
    B = ref_depth.shape[0]
    device, dtype = ref_depth.device, ref_depth.dtype
    ref_xy1 = _pixel_grid(B, height, width, device, dtype)

    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]
    trans = proj[:, :3, 3:4]

    depth_flat = ref_depth.reshape(B, 1, height * width)
    src_xyz = torch.matmul(rot, ref_xy1) * depth_flat + trans
    src_z = src_xyz[:, 2:3, :].clamp(min=1e-6)
    src_xy = src_xyz[:, :2, :] / src_z
    return src_xy, src_z, ref_xy1


def projection_visibility_confidence(ref_depth, ref_proj, src_proj, height, width, border_margin=2.0):
    """Geometry-guided visibility without a source depth map.

    This is intentionally a visibility prior, not a strict depth-consistency
    score: pixels projected outside the source view or behind the source camera
    receive low confidence, while safely visible pixels stay near 1.
    """
    with torch.no_grad():
        B = ref_depth.shape[0]
        src_xy, src_z, _ = project_depth(ref_depth, ref_proj, src_proj, height, width)
        x = src_xy[:, 0, :]
        y = src_xy[:, 1, :]

        dx = torch.minimum(x, (width - 1) - x)
        dy = torch.minimum(y, (height - 1) - y)
        border_dist = torch.minimum(dx, dy)
        border_conf = torch.sigmoid(border_dist / max(border_margin, 1e-6))
        z_conf = (src_z[:, 0, :] > 1e-6).to(ref_depth.dtype)
        return (border_conf * z_conf).reshape(B, 1, height, width).clamp(0.0, 1.0)


def roundtrip_reprojection_confidence(ref_depth, src_depth, ref_proj, src_proj,
                                      height, width, reproj_sigma=1.0, depth_sigma=0.01):
    """Strict ref -> src -> ref consistency using a source-as-reference depth.

    Args:
        ref_depth: [B, H, W] reference depth.
        src_depth: [B, H, W] predicted source depth at the same image scale.

    Returns:
        confidence: [B, 1, H, W] in [0, 1].
    """
    with torch.no_grad():
        B = ref_depth.shape[0]
        dtype = ref_depth.dtype

        src_xy, src_z, ref_xy1 = project_depth(ref_depth, ref_proj, src_proj, height, width)
        grid = _normalize_grid(src_xy, height, width).view(B, height, width, 2)
        sampled_src_depth = F.grid_sample(
            src_depth.unsqueeze(1), grid, mode='bilinear',
            padding_mode='zeros', align_corners=True).view(B, 1, height * width)

        back_proj = torch.matmul(ref_proj, torch.inverse(src_proj))
        back_rot = back_proj[:, :3, :3]
        back_trans = back_proj[:, :3, 3:4]
        src_xy1 = torch.cat((src_xy, torch.ones_like(src_xy[:, :1, :])), dim=1)
        back_xyz = torch.matmul(back_rot, src_xy1) * sampled_src_depth + back_trans
        back_z = back_xyz[:, 2:3, :].clamp(min=1e-6)
        back_xy = back_xyz[:, :2, :] / back_z

        reproj_err = (back_xy - ref_xy1[:, :2, :]).norm(dim=1)
        depth_rel = (back_z - ref_depth.reshape(B, 1, height * width)).abs()
        depth_rel = depth_rel[:, 0, :] / ref_depth.reshape(B, height * width).clamp(min=1e-6)

        in_front = (src_z[:, 0, :] > 1e-6) & (sampled_src_depth[:, 0, :] > 1e-6)
        in_bounds = (
            (grid[..., 0].reshape(B, -1) >= -1.0) &
            (grid[..., 0].reshape(B, -1) <= 1.0) &
            (grid[..., 1].reshape(B, -1) >= -1.0) &
            (grid[..., 1].reshape(B, -1) <= 1.0))

        conf = torch.exp(-reproj_err / reproj_sigma) * torch.exp(-depth_rel / depth_sigma)
        conf = conf * (in_front & in_bounds).to(dtype)
        return conf.reshape(B, 1, height, width).clamp(0.0, 1.0)
