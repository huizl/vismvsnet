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


def signed_occlusion_visibility_confidence(
        ref_depth, src_depth, ref_proj, src_proj, height, width,
        occ_abs_tol=2.0, occ_rel_tol=0.01, temperature=1.0):
    """Visibility from signed source-depth ordering.

    A projected reference point is likely occluded when it lies behind the
    sampled source surface by more than the configured depth tolerance.
    Unlike round-trip consistency, this gate does not penalize every symmetric
    mismatch; it only suppresses the behind-surface ordering associated with
    occlusion.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if occ_abs_tol < 0.0 or occ_rel_tol < 0.0:
        raise ValueError("occlusion tolerances must be non-negative")

    with torch.no_grad():
        B = ref_depth.shape[0]
        dtype = ref_depth.dtype

        src_xy, projected_src_z, _ = project_depth(
            ref_depth, ref_proj, src_proj, height, width)
        grid = _normalize_grid(src_xy, height, width).view(B, height, width, 2)
        sampled_src_depth = F.grid_sample(
            src_depth.unsqueeze(1), grid, mode='bilinear',
            padding_mode='zeros', align_corners=True).view(B, 1, height * width)

        in_bounds = (
            (grid[..., 0].reshape(B, -1) >= -1.0) &
            (grid[..., 0].reshape(B, -1) <= 1.0) &
            (grid[..., 1].reshape(B, -1) >= -1.0) &
            (grid[..., 1].reshape(B, -1) <= 1.0))
        valid = (
            in_bounds &
            (projected_src_z[:, 0, :] > 1e-6) &
            (sampled_src_depth[:, 0, :] > 1e-6))

        projected = projected_src_z[:, 0, :]
        sampled = sampled_src_depth[:, 0, :]
        tolerance = torch.maximum(
            torch.full_like(sampled, occ_abs_tol),
            occ_rel_tol * sampled)

        signed_gap = projected - sampled - tolerance
        transition = (temperature * tolerance).clamp(min=1e-6)
        visibility = torch.sigmoid(-signed_gap / transition)
        visibility = visibility * valid.to(dtype)
        return visibility.reshape(B, 1, height, width).clamp(0.0, 1.0)


def learned_visibility_geometry_features(
        ref_depth, src_depth, ref_proj, src_proj, height, width,
        occ_abs_tol=2.0, occ_rel_tol=0.01, temperature=1.0,
        reproj_sigma=1.0, depth_sigma=0.01):
    """Geometry inputs for the learned pair-wise visibility head.

    The two returned confidence maps deliberately retain different failure
    modes: round-trip confidence is symmetric, whereas signed visibility only
    penalizes a reference point that lies behind the predicted source surface.
    The visibility head can learn when either cue is trustworthy.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if occ_abs_tol < 0.0 or occ_rel_tol < 0.0:
        raise ValueError("occlusion tolerances must be non-negative")

    with torch.no_grad():
        B = ref_depth.shape[0]
        dtype = ref_depth.dtype

        src_xy, projected_src_z, ref_xy1 = project_depth(
            ref_depth, ref_proj, src_proj, height, width)
        grid = _normalize_grid(src_xy, height, width).view(B, height, width, 2)
        sampled_src_depth = F.grid_sample(
            src_depth.unsqueeze(1), grid, mode='bilinear',
            padding_mode='zeros', align_corners=True).view(B, 1, height * width)

        in_bounds = (
            (grid[..., 0].reshape(B, -1) >= -1.0) &
            (grid[..., 0].reshape(B, -1) <= 1.0) &
            (grid[..., 1].reshape(B, -1) >= -1.0) &
            (grid[..., 1].reshape(B, -1) <= 1.0))
        valid = (
            in_bounds &
            (projected_src_z[:, 0, :] > 1e-6) &
            (sampled_src_depth[:, 0, :] > 1e-6))

        projected = projected_src_z[:, 0, :]
        sampled = sampled_src_depth[:, 0, :]
        tolerance = torch.maximum(
            torch.full_like(sampled, occ_abs_tol),
            occ_rel_tol * sampled)
        transition = (temperature * tolerance).clamp(min=1e-6)
        signed_visibility = torch.sigmoid(
            -(projected - sampled - tolerance) / transition)

        back_proj = torch.matmul(ref_proj, torch.inverse(src_proj))
        back_rot = back_proj[:, :3, :3]
        back_trans = back_proj[:, :3, 3:4]
        src_xy1 = torch.cat(
            (src_xy, torch.ones_like(src_xy[:, :1, :])), dim=1)
        back_xyz = torch.matmul(back_rot, src_xy1) * sampled_src_depth + back_trans
        back_z = back_xyz[:, 2:3, :].clamp(min=1e-6)
        back_xy = back_xyz[:, :2, :] / back_z

        reproj_err = (back_xy - ref_xy1[:, :2, :]).norm(dim=1)
        ref_depth_flat = ref_depth.reshape(B, height * width).clamp(min=1e-6)
        depth_rel = (back_z[:, 0, :] - ref_depth_flat).abs() / ref_depth_flat
        roundtrip = (
            torch.exp(-reproj_err / reproj_sigma) *
            torch.exp(-depth_rel / depth_sigma))

        valid_float = valid.to(dtype)
        roundtrip = roundtrip * valid_float
        signed_visibility = signed_visibility * valid_float
        return (
            roundtrip.reshape(B, 1, height, width).clamp(0.0, 1.0),
            signed_visibility.reshape(B, 1, height, width).clamp(0.0, 1.0),
            valid_float.reshape(B, 1, height, width),
        )


def ground_truth_pair_visibility(
        ref_depth, src_depth, ref_valid, src_valid,
        ref_proj, src_proj, height, width,
        occ_abs_tol=2.0, occ_rel_tol=0.01):
    """Build visible/occluded supervision from a pair of GT depth maps.

    A point is visible when its projected source depth agrees with the source
    GT surface. It is occluded when it lies behind that surface. Points that
    project outside the source view, lack valid GT, or lie significantly in
    front of the sampled source surface are excluded from supervision.

    Returns:
        target: [B, 1, H, W], 1 for visible and 0 for occluded.
        supervised: [B, 1, H, W] boolean mask.
    """
    if occ_abs_tol < 0.0 or occ_rel_tol < 0.0:
        raise ValueError("occlusion tolerances must be non-negative")

    with torch.no_grad():
        B = ref_depth.shape[0]
        src_xy, projected_src_z, _ = project_depth(
            ref_depth, ref_proj, src_proj, height, width)
        grid = _normalize_grid(src_xy, height, width).view(B, height, width, 2)

        sampled_src_depth = F.grid_sample(
            src_depth.unsqueeze(1), grid, mode='nearest',
            padding_mode='zeros', align_corners=True).view(B, height * width)
        sampled_src_valid = F.grid_sample(
            src_valid.to(ref_depth.dtype).unsqueeze(1), grid, mode='nearest',
            padding_mode='zeros', align_corners=True).view(B, height * width) > 0.5

        in_bounds = (
            (grid[..., 0].reshape(B, -1) >= -1.0) &
            (grid[..., 0].reshape(B, -1) <= 1.0) &
            (grid[..., 1].reshape(B, -1) >= -1.0) &
            (grid[..., 1].reshape(B, -1) <= 1.0))
        projected = projected_src_z[:, 0, :]
        ref_valid_flat = ref_valid.reshape(B, height * width) > 0.5
        comparable = (
            in_bounds & ref_valid_flat & sampled_src_valid &
            (projected > 1e-6) & (sampled_src_depth > 1e-6))

        tolerance = torch.maximum(
            torch.full_like(sampled_src_depth, occ_abs_tol),
            occ_rel_tol * sampled_src_depth)
        depth_gap = projected - sampled_src_depth
        visible = comparable & (depth_gap.abs() <= tolerance)
        occluded = comparable & (depth_gap > tolerance)
        supervised = visible | occluded

        return (
            visible.to(ref_depth.dtype).reshape(B, 1, height, width),
            supervised.reshape(B, 1, height, width),
        )


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
