import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


# =============================================================================
# Feature Extractor: multi-scale, outputs at 1/8, 1/4, 1/2 resolution
# =============================================================================
class MultiScaleFeatureNet(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        # 1/2 scale
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        # 1/4 scale
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)

        # 1/8 scale
        self.conv7 = ConvBnReLU(32, 32, 5, 2, 2)
        self.conv8 = ConvBnReLU(32, 32, 3, 1, 1)

        # output convolutions — one per scale
        self.feat2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.feat4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.feat8 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))

        x2 = self.conv4(self.conv3(self.conv2(x)))      # 1/2
        x4 = self.conv6(self.conv5(x2))                  # 1/4
        x8 = self.conv8(self.conv7(x4))                  # 1/8

        return self.feat8(x8), self.feat4(x4), self.feat2(x2)


# =============================================================================
# 3D Regularization Networks
# =============================================================================
class RegNet3D(nn.Module):
    """3D encoder-decoder for per-pair cost volume (3-level: 8→16→32→64→32→16→8)."""

    def __init__(self, in_channels=8):
        super(RegNet3D, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(32))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(16))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(8))

    def forward(self, x):
        conv0 = self.conv0(x)                               # [B, 8, D, H, W]
        conv2 = self.conv2(self.conv1(conv0))               # [B, 16, D, H/2, W/2]
        conv4 = self.conv4(self.conv3(conv2))               # [B, 32, D, H/4, W/4]
        conv6 = self.conv6(self.conv5(conv4))               # [B, 64, D, H/8, W/8]
        x = self.deconv5(conv6)
        x = x[:, :, :conv4.shape[2], :conv4.shape[3], :conv4.shape[4]]
        x = F.relu(x + conv4, inplace=True)
        x = self.deconv3(x)
        x = x[:, :, :conv2.shape[2], :conv2.shape[3], :conv2.shape[4]]
        x = F.relu(x + conv2, inplace=True)
        x = self.deconv1(x)
        x = x[:, :, :conv0.shape[2], :conv0.shape[3], :conv0.shape[4]]
        x = F.relu(x + conv0, inplace=True)
        return x


class RegPair(nn.Module):
    """Per-pair score head: 8 → 1."""

    def __init__(self):
        super(RegPair, self).__init__()
        self.conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class RegFuse(nn.Module):
    """Fused cost volume regularization (3-level) + final score head."""

    def __init__(self):
        super(RegFuse, self).__init__()
        self.conv0 = ConvBnReLU3D(8, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(32))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(16))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(8))

        self.final = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        conv6 = self.conv6(self.conv5(conv4))
        x = self.deconv5(conv6)
        x = x[:, :, :conv4.shape[2], :conv4.shape[3], :conv4.shape[4]]
        x = F.relu(x + conv4, inplace=True)
        x = self.deconv3(x)
        x = x[:, :, :conv2.shape[2], :conv2.shape[3], :conv2.shape[4]]
        x = F.relu(x + conv2, inplace=True)
        x = self.deconv1(x)
        x = x[:, :, :conv0.shape[2], :conv0.shape[3], :conv0.shape[4]]
        x = F.relu(x + conv0, inplace=True)
        x = self.final(x)
        return x


# =============================================================================
# Uncertainty Network
# =============================================================================
class UncertNet(nn.Module):
    """2D CNN: entropy map → uncertainty + occlusion logits (two heads)."""

    def __init__(self):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.uncert_head = nn.Conv2d(8, 1, 3, 1, 1, bias=False)
        self.occ_head = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, entropy_map):
        out = self.conv1(entropy_map)
        out = self.conv2(out) + entropy_map
        uncert = self.uncert_head(out)
        occ = self.occ_head(out)
        return uncert, occ


# =============================================================================
# Single Cascaded Stage
# =============================================================================
class SingleStage(nn.Module):
    """One stage of the cascaded pipeline.

    For each source view:
      1. Warp + groupwise correlation → pair cost volume [B, 8, D, H, W]
      2. 3D regularization → intermediate features
      3. Score → softmax → depth + entropy → uncertainty
      4. Uncertainty-weighted accumulation into fused cost volume
    After all pairs: final regularization → fused depth.
    """

    def __init__(self):
        super(SingleStage, self).__init__()
        self.reg = RegNet3D(8)
        self.reg_pair = RegPair()
        self.reg_fuse = RegFuse()
        self.uncert_net = UncertNet()

    def forward(self, ref_feat, ref_proj, src_feats, src_projs, depth_values, mode='soft'):
        """
        Args:
            ref_feat:  [B, C, H, W] reference feature
            ref_proj:  [B, 4, 4]   reference projection matrix
            src_feats: list of [B, C, H, W] source features
            src_projs: list of [B, 4, 4] source projection matrices
            depth_values: [B, D] or [B, D, H, W] depth hypotheses
            mode: fusion mode ('soft' | 'average' | 'maxpool' | 'uwta' | 'hard')

        Returns:
            final_depth:  [B, H, W] fused depth
            confidence:   [B, H, W] Vis-MVSNet prob_map (sum of probs within ±2 of argmin)
            pair_results: list of (est_depth [B,H,W], uncert [B,1,H,W], occ [B,1,H,W])
        """
        B, C, H, W = ref_feat.shape
        D = depth_values.shape[1]

        ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)  # [B, C, D, H, W]

        pair_results = []

        # ---- fusion buffers ----
        if mode in ('soft', 'hard'):
            weight_sum = torch.zeros(B, 1, 1, H, W, device=ref_feat.device, dtype=ref_feat.dtype)
        fused_interm = torch.zeros(B, 8, D, H, W, device=ref_feat.device, dtype=ref_feat.dtype)

        # per-pair early initialisation for uwta / maxpool
        min_weight = None
        maxpool_init = True

        # ---- per source view ----
        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_src = homo_warping(src_feat, src_proj, ref_proj, depth_values)
            cost_volume = groupwise_correlation(ref_volume, warped_src, 8, dim=1)   # [B, 8, D, H, W]

            interm = self.reg(cost_volume)                                           # [B, 8, D, H, W]
            score = self.reg_pair(interm).squeeze(1)                                 # [B, D, H, W]
            prob = F.softmax(score, dim=1)                                           # [B, D, H, W]
            est_depth = depth_regression(prob, depth_values)                         # [B, H, W]

            entropy_map = entropy(prob, dim=1, keepdim=True)                         # [B, 1, H, W]
            uncert, occ = self.uncert_net(entropy_map)                               # [B,1,H,W], [B,1,H,W]
            pair_results.append((est_depth, uncert, occ))

            # ---- fuse intermediate features ----
            if mode == 'soft':
                weight = (-uncert).exp().unsqueeze(2)            # [B, 1, 1, H, W]
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            elif mode == 'hard':
                weight = (uncert < 0).to(interm.dtype).unsqueeze(2) + 1e-4
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            elif mode == 'average':
                fused_interm = fused_interm + interm
            elif mode == 'uwta':
                w = uncert.unsqueeze(2)
                if min_weight is None:
                    min_weight = w
                    mask = torch.ones_like(w).to(interm.dtype)
                else:
                    mask = (w < min_weight).to(interm.dtype)
                    min_weight = w * mask + min_weight * (1 - mask)
                fused_interm = interm * mask + fused_interm * (1 - mask)
            elif mode == 'maxpool':
                if maxpool_init:
                    fused_interm = fused_interm + interm
                    maxpool_init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)

            if not self.training:
                del prob, score, interm, cost_volume, warped_src

        # ---- normalise fused features ----
        if mode == 'soft':
            fused_interm = fused_interm / weight_sum
        elif mode == 'hard':
            fused_interm = fused_interm / weight_sum
        elif mode == 'average':
            fused_interm = fused_interm / len(src_feats)

        # ---- final score after fusion ----
        final_score = self.reg_fuse(fused_interm).squeeze(1)     # [B, D, H, W]
        final_prob = F.softmax(final_score, dim=1)                # [B, D, H, W]
        final_depth = depth_regression(final_prob, depth_values)  # [B, H, W]
        confidence = prob_map(final_prob)                          # [B, H, W]

        return final_depth, confidence, pair_results


# =============================================================================
# Full Cascaded Vis-MVSNet Model
# =============================================================================
class VisMVSModel(nn.Module):
    """3-stage cascaded MVS network with pair-wise uncertainty-aware fusion.

    Stage 1 (1/8 scale): coarse depth, wide range
    Stage 2 (1/4 scale): refined depth, range narrowed by stage 1
    Stage 3 (1/2 scale): finest depth, range narrowed by stage 2
    """

    def __init__(self, mode='soft',
                 stage1_depth_num=48, stage1_interval_scale=4,
                 stage2_depth_num=32, stage2_interval_scale=2,
                 stage3_depth_num=16, stage3_interval_scale=1):
        super(VisMVSModel, self).__init__()
        self.feature = MultiScaleFeatureNet()
        self.stage1 = SingleStage()
        self.stage2 = SingleStage()
        self.stage3 = SingleStage()
        self.mode = mode

        self.s1_dnum = stage1_depth_num
        self.s1_iscale = stage1_interval_scale
        self.s2_dnum = stage2_depth_num
        self.s2_iscale = stage2_interval_scale
        self.s3_dnum = stage3_depth_num
        self.s3_iscale = stage3_interval_scale

    def _build_depth_range(self, depth_start, depth_interval, depth_num, B, H, W, device, dtype):
        """Build spatially-uniform depth hypotheses [B, D]."""
        d = torch.arange(depth_num, device=device, dtype=dtype)
        if depth_start.dim() == 2:
            depth_start = depth_start.view(B, 1)
            depth_interval = depth_interval.view(B, 1)
        return depth_start + depth_interval * d  # [B, D]

    def _build_per_pixel_depth_range(self, pred_depth, depth_num, interval, B, H, W, device, dtype):
        """Build per-pixel depth hypotheses [B, D, H, W] centred on pred_depth."""
        half = depth_num // 2
        d = torch.arange(depth_num, device=device, dtype=dtype).view(1, depth_num, 1, 1)
        # interval is [B, 1]; add trailing dims so it broadcasts with [B,H,W] and [1,D,1,1]
        interval = interval.view(B, 1, 1)
        depth_start = pred_depth - half * interval
        if depth_start.dim() == 3:
            depth_start = depth_start.unsqueeze(1)  # [B, 1, H, W]
        return depth_start + interval.view(B, 1, 1, 1) * d  # [B, D, H, W]

    def forward(self, imgs, proj_matrices, depth_values_orig):
        """
        Args:
            imgs:               [B, V, 3, H, W]
            proj_matrices:      [B, V, 4, 4]
            depth_values_orig:  [B, D_orig]  original depth hypotheses from dataset

        Returns:
            outputs:     list of [stage_depth, pair_results] per stage
            final_depth: [B, 1, H, W]
            conf_maps:   list of [conf_1, conf_2, conf_3] upsampled confidence maps [B, 1, H, W]
        """
        imgs_list = list(torch.unbind(imgs, 1))
        proj_list = list(torch.unbind(proj_matrices, 1))
        ref_img, src_imgs = imgs_list[0], imgs_list[1:]
        ref_proj, src_projs = proj_list[0], proj_list[1:]

        B, _, H_full, W_full = ref_img.shape
        D_orig = depth_values_orig.shape[1]
        device = ref_img.device
        dtype = ref_img.dtype

        # ---- multi-scale feature extraction (all views batched) ----
        V = len(imgs_list)
        all_imgs = torch.cat([ref_img] + src_imgs, dim=0)  # [V*B, 3, H, W]
        feat8_all, feat4_all, feat2_all = self.feature(all_imgs)

        def split_feats(f_all):
            return [f_all[i * B:(i + 1) * B] for i in range(V)]

        ref_feat8, *srcs_feat8 = split_feats(feat8_all)
        ref_feat4, *srcs_feat4 = split_feats(feat4_all)
        ref_feat2, *srcs_feat2 = split_feats(feat2_all)

        # ---- base depth params ----
        base_interval = depth_values_orig[:, 1:2] - depth_values_orig[:, 0:1]  # [B, 1]
        base_start = depth_values_orig[:, 0:1]                                  # [B, 1]

        def _scale_proj(proj, s):
            """Scale the first two rows of a projection matrix by factor s.

            Dataset proj_matrices have intrinsics at 1/4 image scale.
            For a feature map at 1/S scale, multiply by (4/S) to convert.
            """
            p = proj.clone()
            p[:, :2, :] *= s
            return p

        # =====================================================================
        # Stage 1 — 1/8 scale, coarsest
        # =====================================================================
        # intrinsics are at 1/4 scale; features at 1/8 → scale by 0.5
        s1_ref_proj = _scale_proj(ref_proj, 0.5)
        s1_src_projs = [_scale_proj(p, 0.5) for p in src_projs]

        s1_interval = base_interval * self.s1_iscale
        depth_values_1 = self._build_depth_range(
            base_start, s1_interval, self.s1_dnum, B,
            ref_feat8.shape[2], ref_feat8.shape[3], device, dtype)  # [B, D1]

        depth_1, conf_1, pairs_1 = self.stage1(
            ref_feat8, s1_ref_proj, srcs_feat8, s1_src_projs,
            depth_values_1, mode=self.mode)

        conf_1_up = F.interpolate(conf_1.unsqueeze(1), scale_factor=4, mode='bilinear',
                                  align_corners=False)  # [B,1,4H8,4W8]

        # =====================================================================
        # Stage 2 — 1/4 scale, range narrowed by stage 1
        # =====================================================================
        H4, W4 = ref_feat4.shape[2], ref_feat4.shape[3]
        depth_1_up = F.interpolate(depth_1.unsqueeze(1), size=(H4, W4),
                                   mode='bilinear', align_corners=False).squeeze(1)  # [B, H4, W4]

        s2_interval = base_interval * self.s2_iscale       # per-pixel interval [B, 1]
        depth_values_2 = self._build_per_pixel_depth_range(
            depth_1_up, self.s2_dnum, s2_interval, B, H4, W4, device, dtype)  # [B, D2, H4, W4]

        depth_2, conf_2, pairs_2 = self.stage2(
            ref_feat4, ref_proj, srcs_feat4, src_projs,
            depth_values_2, mode=self.mode)

        conf_2_up = F.interpolate(conf_2.unsqueeze(1), scale_factor=2, mode='bilinear',
                                  align_corners=False)  # [B,1,2H4,2W4]

        # =====================================================================
        # Stage 3 — 1/2 scale, finest
        # =====================================================================
        # intrinsics are at 1/4 scale; features at 1/2 → scale by 2.0
        s3_ref_proj = _scale_proj(ref_proj, 2.0)
        s3_src_projs = [_scale_proj(p, 2.0) for p in src_projs]

        H2, W2 = ref_feat2.shape[2], ref_feat2.shape[3]
        depth_2_up = F.interpolate(depth_2.unsqueeze(1), size=(H2, W2),
                                   mode='bilinear', align_corners=False).squeeze(1)  # [B, H2, W2]

        s3_interval = base_interval * self.s3_iscale
        depth_values_3 = self._build_per_pixel_depth_range(
            depth_2_up, self.s3_dnum, s3_interval, B, H2, W2, device, dtype)  # [B, D3, H2, W2]

        depth_3, conf_3, pairs_3 = self.stage3(
            ref_feat2, s3_ref_proj, srcs_feat2, s3_src_projs,
            depth_values_3, mode=self.mode)

        conf_3_up = conf_3.unsqueeze(1)  # [B, 1, H2, W2]

        # ---- final depth at stage 3 scale ----
        final_depth = depth_3.unsqueeze(1)  # [B, 1, H2, W2]

        outputs = [
            [depth_1, pairs_1],
            [depth_2, pairs_2],
            [depth_3, pairs_3],
        ]

        return outputs, final_depth, [conf_1_up, conf_2_up, conf_3_up]


# =============================================================================
# Loss
# =============================================================================
class VisMVSLoss(nn.Module):
    """Multi-component cascaded loss.

    For each stage:
      - L1 loss on fused depth (pixel units, scaled by depth_interval)
      - Pair L1 loss on per-source-view depth
      - Uncertainty loss  (error * exp(-uncert) + uncert)
      - Occlusion logistic loss (optional, when occ_guide=True with per-view masks)
    Final weighted sum: 0.5 * stage1 + 1.0 * stage2 + 2.0 * stage3

    Reference: Vis-MVSNet (BMVC 2020 / IJCV 2022), Sec. 3.4
    """

    def __init__(self, occ_guide=False):
        super(VisMVSLoss, self).__init__()
        self.occ_guide = occ_guide
        self.stage_weights = [0.5, 1.0, 2.0]

    def forward(self, stage_outputs, depth_gt, mask, depth_interval):
        """
        Args:
            stage_outputs: list of [stage_depth, pair_results] per stage
            depth_gt:      [B, 1, H, W]
            mask:          [B, 1, H, W] valid pixel mask  (or [B, V, 1, H, W] for occ_guide)
            depth_interval: base depth interval (scalar or [B])

        Returns:
            total_loss, scalar_stats dict
        """
        if depth_interval.dim() == 0:
            depth_interval = depth_interval.view(1)

        # ensure depth_gt and mask are 4-D [B, 1, H, W]
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        stage_losses = []
        stats = []

        for stage_idx, (est_depth, pair_results) in enumerate(stage_outputs):
            B, H_stage, W_stage = est_depth.shape

            gt_ds = F.interpolate(depth_gt, size=(H_stage, W_stage),
                                  mode='bilinear', align_corners=False).squeeze(1)
            mask_ds = F.interpolate(mask, size=(H_stage, W_stage),
                                    mode='nearest').squeeze(1)

            valid = mask_ds > 0.5

            # ---- fused L1 (pixel-normalised) ----
            abs_err = (est_depth - gt_ds).abs()
            abs_err_scaled = abs_err / depth_interval.view(-1, 1, 1)

            l1 = abs_err_scaled[valid].mean()

            # ---- per-pair losses ----
            pair_abs_err = []
            for p_est, p_uncert, p_occ in pair_results:
                err = (p_est - gt_ds).abs() / depth_interval.view(-1, 1, 1)
                pair_abs_err.append(err)

            pair_l1_losses = [e[valid].mean() for e in pair_abs_err]
            pair_l1 = sum(pair_l1_losses) / len(pair_l1_losses)

            # uncertainty loss  (Vis-MVSNet Eq. 10)
            uncert_losses = []
            for err, (p_est, p_uncert, p_occ) in zip(pair_abs_err, pair_results):
                u = p_uncert.squeeze(1)
                uloss = (err[valid] * (-u[valid]).exp() + u[valid]).mean()
                uncert_losses.append(uloss)
            uncert_loss = sum(uncert_losses) / len(uncert_losses)

            pair_loss = pair_l1 + uncert_loss

            # occlusion logistic loss  (Vis-MVSNet Eq. 11, optional)
            if self.occ_guide:
                logistic_losses = []
                for (p_est, p_uncert, p_occ) in pair_results:
                    occ = p_occ.squeeze(1)
                    # mask > 0.5 → visible → target +1;  else occluded → target -1
                    target = valid.float() * 2 - 1
                    logistic_losses.append(
                        F.soft_margin_loss(occ[valid], target[valid], reduction='mean'))
                logistic_loss = sum(logistic_losses) / len(logistic_losses)
                pair_loss = pair_loss + logistic_loss

            loss = l1 + pair_loss
            stage_losses.append(loss)

            less1 = (abs_err_scaled[valid] < 1.0).float().mean()
            less3 = (abs_err_scaled[valid] < 3.0).float().mean()
            stats.append((l1, less1, less3))

        total_loss = sum(w * l for w, l in zip(self.stage_weights, stage_losses))

        l1_0, less1_0, less3_0 = stats[0]
        l1_1, less1_1, less3_1 = stats[1] if len(stats) > 1 else (l1_0, less1_0, less3_0)
        l1_2, less1_2, less3_2 = stats[2] if len(stats) > 2 else (l1_0, less1_0, less3_0)

        scalar_stats = {
            'loss': total_loss,
            'l1_stage1': l1_0,
            'l1_stage2': l1_1,
            'l1_stage3': l1_2,
            'less1_stage3': less1_2,
            'less3_stage3': less3_2,
        }

        return total_loss, scalar_stats
