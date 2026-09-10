import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """Warp source features to reference view using differentiable homography.

    Args:
        src_fea: [B, C, H, W]
        src_proj: [B, 4, 4]
        ref_proj: [B, 4, 4]
        depth_values: [B, Ndepth] or [B, Ndepth, H, W]

    Returns:
        warped_src_fea: [B, C, Ndepth, H, W]
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    is_per_pixel = depth_values.dim() == 4
    if is_per_pixel:
        num_depth = depth_values.shape[1]
    else:
        num_depth = depth_values.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        if is_per_pixel:
            depth_flat = depth_values.view(batch, num_depth, height * width)  # [B, D, H*W]
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_flat.unsqueeze(1)  # [B, 3, D, H*W]
        else:
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, 1)  # [B, 3, D, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2),
                                   mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def depth_regression(p, depth_values):
    """Soft-argmin: weighted sum of depth hypotheses by probability.

    Args:
        p: probability volume [B, D, H, W]
        depth_values: depth hypotheses [B, D] or [B, D, H, W]
    Returns:
        depth: [B, H, W]
    """
    if depth_values.dim() == 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


def groupwise_correlation(v1, v2, groups, dim):
    """Groupwise correlation (CIDER).

    Splits channels into groups, computes dot product within each group.

    Args:
        v1, v2: feature volumes of same shape [B, C, D, H, W]
        groups: number of groups (C must be divisible by groups)
        dim: the channel dimension (typically 1)
    Returns:
        correlation volume [B, groups, D, H, W]
    """
    size = list(v1.size())
    c = size[dim]
    assert c % groups == 0
    s1 = size[:dim]
    s2 = size[dim + 1:]
    reshaped_size = s1 + [groups, c // groups] + s2
    v1_reshaped = v1.view(*reshaped_size)
    v2_reshaped = v2.view(*reshaped_size)
    vc = (v1_reshaped * v2_reshaped).sum(dim=dim + 1)
    return vc


def prob_map(prob_volume, window=2):
    """Confidence map: sum of probabilities within ±window depth bins of argmin.

    This is Vis-MVSNet's confidence measure (prob_map), distinct from max probability.
    Reference: Vis-MVSNet (IJCV 2022), Sec. 3.3, soft_argmin with window.

    Args:
        prob_volume: [B, D, H, W] probability volume (already softmaxed along dim=1)
        window: number of depth bins on each side of argmin (default 2)

    Returns:
        confidence: [B, H, W] sum of probabilities within the window
    """
    B, D, H, W = prob_volume.shape
    index = torch.arange(D, device=prob_volume.device, dtype=prob_volume.dtype).view(1, D, 1, 1)
    argmin_idx = torch.sum(prob_volume * index, dim=1, keepdim=True)  # [B, 1, H, W]
    mask = (index - argmin_idx).abs() <= window
    confidence = torch.sum(prob_volume * mask.to(prob_volume.dtype), dim=1)  # [B, H, W]
    return confidence


def entropy(volume, dim, keepdim=False):
    """Compute entropy of a probability volume.

    Args:
        volume: probability volume [B, D, H, W]
        dim: dimension to compute entropy over
    Returns:
        entropy map
    """
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)
