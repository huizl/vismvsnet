"""Vis-MVSNet baseline with three independently ablatable research modules.

The default configuration is intended for trinocular training (one reference and
two source views):
  M1 adaptive multi-hypothesis search for coarse-stage error recovery;
  M2 depth-hypothesis-aware visibility fusion w_i(p, z);
  M3 visibility-constrained boundary refinement.

Disabling all three modules recovers the local Vis-MVSNet baseline architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (depth_regression, entropy, groupwise_correlation,
                     homo_warping, prob_map)
from .vismvsnet import (MultiScaleFeatureNet, RegFuse, RegNet3D, RegPair,
                        UncertNet, VisMVSLoss as BaselineLoss)


def _scale_projection(projection, scale):
    result = projection.clone()
    result[:, :2, :] *= scale
    return result


def candidate_projection_validity(ref_proj, src_proj, depth_values, height, width):
    """Return a differentiable-free in-frame mask for every depth hypothesis."""
    batch, num_depth = depth_values.shape[:2]
    device, dtype = depth_values.device, depth_values.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing='ij')
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=0).reshape(1, 3, -1)
    pixels = pixels.expand(batch, -1, -1)
    relative = src_proj @ torch.inverse(ref_proj)
    rotation, translation = relative[:, :3, :3], relative[:, :3, 3:4]

    if depth_values.dim() == 2:
        depths = depth_values[:, :, None].expand(-1, -1, height * width)
    else:
        depths = depth_values.reshape(batch, num_depth, -1)
    points = (rotation @ pixels).unsqueeze(2) * depths[:, None] + translation.unsqueeze(2)
    z = points[:, 2]
    x = points[:, 0] / z.clamp_min(1e-6)
    y = points[:, 1] / z.clamp_min(1e-6)
    valid = ((z > 1e-6) & (x >= 0) & (x <= width - 1) &
             (y >= 0) & (y <= height - 1)).to(dtype)
    return valid.reshape(batch, num_depth, height, width)


def _gather_depth(depth_values, indices):
    if depth_values.dim() == 2:
        expanded = depth_values[:, :, None, None].expand(
            -1, -1, indices.shape[-2], indices.shape[-1])
    else:
        expanded = depth_values
    return torch.gather(expanded, 1, indices.unsqueeze(1)).squeeze(1)


def secondary_mode(probability, depth_values, exclusion_radius=2):
    """Depth of the strongest probability mode outside the primary peak."""
    with torch.no_grad():
        primary = probability.argmax(dim=1, keepdim=True)
        bins = torch.arange(probability.shape[1], device=probability.device).view(1, -1, 1, 1)
        suppressed = probability.masked_fill((bins - primary).abs() <= exclusion_radius, -1.0)
        secondary = suppressed.argmax(dim=1)
        return _gather_depth(depth_values, secondary)


class HypothesisVisibilityNet(nn.Module):
    """Predict visibility separately for each source, pixel, and depth candidate."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(5, 8, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, (3, 1, 1), padding=(1, 0, 0),
                      groups=8, bias=True), nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1, bias=True))
        # A neutral initial gate preserves relative baseline weights.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, score, probability, uncertainty, depth_values,
                pair_depth, projection_validity, interval):
        score_norm = (score - score.mean(1, keepdim=True)) / (
            score.std(1, keepdim=True, unbiased=False) + 1e-6)
        if depth_values.dim() == 2:
            depth_volume = depth_values[:, :, None, None].expand_as(score)
        else:
            depth_volume = depth_values
        interval = interval.view(-1, 1, 1, 1).clamp_min(1e-6)
        distance = ((depth_volume - pair_depth.unsqueeze(1)).abs() / interval).clamp(max=16) / 16
        uncertainty_volume = torch.sigmoid(-uncertainty).unsqueeze(2).expand(
            -1, -1, score.shape[1], -1, -1).squeeze(1)
        features = torch.stack((score_norm, probability, distance,
                                projection_validity, uncertainty_volume), dim=1)
        logits = self.body(features).squeeze(1)
        return logits, torch.sigmoid(logits) * projection_validity


class ResearchStage(nn.Module):
    def __init__(self, use_hypothesis_visibility=True):
        super().__init__()
        self.reg = RegNet3D(8)
        self.reg_pair = RegPair()
        self.reg_fuse = RegFuse()
        self.uncert_net = UncertNet()
        self.use_hypothesis_visibility = use_hypothesis_visibility
        self.visibility = HypothesisVisibilityNet() if use_hypothesis_visibility else None
        self.null_feature = nn.Parameter(torch.zeros(1, 8, 1, 1, 1))

    def forward(self, ref_feat, ref_proj, src_feats, src_projs, depth_values,
                interval, mode='soft'):
        batch, channels, height, width = ref_feat.shape
        num_depth = depth_values.shape[1]
        ref_volume = ref_feat.unsqueeze(2).expand(-1, -1, num_depth, -1, -1)
        fused = torch.zeros(batch, 8, num_depth, height, width,
                            device=ref_feat.device, dtype=ref_feat.dtype)
        weight_sum = torch.zeros(batch, 1, num_depth if self.use_hypothesis_visibility else 1,
                                 height, width, device=ref_feat.device, dtype=ref_feat.dtype)
        pair_results, visibility_volumes = [], []

        for src_feat, src_proj in zip(src_feats, src_projs):
            warped = homo_warping(src_feat, src_proj, ref_proj, depth_values)
            cost = groupwise_correlation(ref_volume, warped, 8, dim=1)
            intermediate = self.reg(cost)
            score = self.reg_pair(intermediate).squeeze(1)
            probability = F.softmax(score, dim=1)
            pair_depth = depth_regression(probability, depth_values)
            uncertainty, occlusion = self.uncert_net(entropy(probability, 1, keepdim=True))
            pair_results.append((pair_depth, uncertainty, occlusion))

            if mode != 'soft':
                raise ValueError("The research model supports soft fusion only")
            base_weight = torch.exp(-uncertainty).unsqueeze(2)
            if self.use_hypothesis_visibility:
                valid = candidate_projection_validity(
                    ref_proj, src_proj, depth_values, height, width)
                _, visibility = self.visibility(
                    score, probability, uncertainty, depth_values,
                    pair_depth, valid, interval)
                weight = base_weight * visibility.unsqueeze(1)
                visibility_volumes.append(visibility)
            else:
                weight = base_weight
            fused = fused + intermediate * weight
            weight_sum = weight_sum + weight

        if self.use_hypothesis_visibility:
            # A learnable no-evidence state prevents forced selection of an invalid source.
            null_weight = torch.exp(-weight_sum)
            fused = (fused + self.null_feature * null_weight) / (weight_sum + null_weight + 1e-6)
            support_volume = (1.0 - null_weight).squeeze(1)
        else:
            fused = fused / weight_sum.clamp_min(1e-6)
            support_volume = torch.ones(batch, num_depth, height, width,
                                        device=ref_feat.device, dtype=ref_feat.dtype)

        final_score = self.reg_fuse(fused).squeeze(1)
        final_probability = F.softmax(final_score, dim=1)
        final_depth = depth_regression(final_probability, depth_values)
        confidence = prob_map(final_probability)
        auxiliary = {
            'probability': final_probability,
            'support_volume': support_volume,
            'visibility_volumes': visibility_volumes,
            'depth_values': depth_values,
        }
        return final_depth, confidence, pair_results, auxiliary


class VisibilityBoundaryRefiner(nn.Module):
    """Refine boundaries only where cross-view evidence supports an update."""
    def __init__(self, feature_channels=32):
        super().__init__()
        self.edge_head = nn.Sequential(
            nn.Conv2d(feature_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
        self.update = nn.Sequential(
            nn.Conv2d(feature_channels + 4, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1))

    def forward(self, feature, depth, confidence, support, interval):
        interval_map = interval.view(-1, 1, 1, 1).clamp_min(1e-6)
        local_mean = F.avg_pool2d(depth, 3, stride=1, padding=1)
        relative_depth = (depth - local_mean) / interval_map
        edge_logits = self.edge_head(feature)
        edge_probability = torch.sigmoid(edge_logits)
        x = torch.cat((feature, relative_depth, confidence, support, edge_probability), dim=1)
        residual, gate = self.update(x).chunk(2, dim=1)
        gate = torch.sigmoid(gate) * support * (0.25 + 0.75 * edge_probability)
        refined = depth + 2.0 * interval_map * torch.tanh(residual) * gate
        return refined, edge_logits, gate


class VisMVSResearchModel(nn.Module):
    def __init__(self, mode='soft', stage1_depth_num=48, stage1_interval_scale=4,
                 stage2_depth_num=32, stage2_interval_scale=2,
                 stage3_depth_num=16, stage3_interval_scale=1,
                 use_adaptive_search=True, use_hypothesis_visibility=True,
                 use_boundary_refine=True, global_candidate_ratio=0.25,
                 secondary_candidate_ratio=0.25):
        super().__init__()
        if mode != 'soft':
            raise ValueError("VisMVSResearchModel currently requires --vismode soft")
        self.feature = MultiScaleFeatureNet()
        self.stage1 = ResearchStage(use_hypothesis_visibility)
        self.stage2 = ResearchStage(use_hypothesis_visibility)
        self.stage3 = ResearchStage(use_hypothesis_visibility)
        self.boundary_refiner = VisibilityBoundaryRefiner() if use_boundary_refine else None
        self.mode = mode
        self.depth_nums = (stage1_depth_num, stage2_depth_num, stage3_depth_num)
        self.interval_scales = (stage1_interval_scale, stage2_interval_scale, stage3_interval_scale)
        self.use_adaptive_search = use_adaptive_search
        self.global_ratio = global_candidate_ratio
        self.secondary_ratio = secondary_candidate_ratio

    @staticmethod
    def _uniform_range(start, interval, count):
        bins = torch.arange(count, device=start.device, dtype=start.dtype).view(1, count)
        return start.view(-1, 1) + interval.view(-1, 1) * bins

    @staticmethod
    def _local_range(center, interval, count):
        bins = torch.arange(count, device=center.device, dtype=center.dtype)
        # Match the baseline cascade exactly when M1 is disabled.
        bins = bins.view(1, count, 1, 1) - count // 2
        if interval.dim() == 1:
            interval = interval.view(-1, 1, 1, 1)
        elif interval.dim() == 3:
            interval = interval.unsqueeze(1)
        return center.unsqueeze(1) + interval * bins

    def _adaptive_range(self, previous_aux, previous_depth, previous_confidence,
                        target_size, count, interval, global_start, global_end):
        center = F.interpolate(previous_depth.unsqueeze(1), size=target_size,
                               mode='bilinear', align_corners=False).squeeze(1)
        confidence = F.interpolate(previous_confidence.unsqueeze(1), size=target_size,
                                   mode='bilinear', align_corners=False).squeeze(1)
        n_global = round(count * self.global_ratio)
        n_secondary = round(count * self.secondary_ratio)
        n_local = count - n_global - n_secondary
        if n_local < 1:
            raise ValueError("Depth count is too small for the configured candidate ratios")
        # Low-confidence pixels receive a wider local interval.
        spread = interval.view(-1, 1, 1) * (1.0 + 2.0 * (1.0 - confidence))
        local = self._local_range(center, spread, n_local)
        candidates = [local]
        if n_secondary:
            secondary = secondary_mode(
                previous_aux['probability'], previous_aux['depth_values'])
            secondary = F.interpolate(secondary.unsqueeze(1), size=target_size,
                                      mode='nearest').squeeze(1)
            candidates.append(self._local_range(secondary, interval * 2.0, n_secondary))
        if n_global:
            fractions = torch.linspace(
                0, 1, n_global, device=center.device, dtype=center.dtype)
            global_values = (global_start.view(-1, 1) * (1 - fractions) +
                             global_end.view(-1, 1) * fractions)
            candidates.append(global_values[:, :, None, None].expand(-1, -1, *target_size))
        hypotheses = torch.cat(candidates, dim=1)
        hypotheses = torch.max(hypotheses, global_start.view(-1, 1, 1, 1))
        hypotheses = torch.min(hypotheses, global_end.view(-1, 1, 1, 1))
        return torch.sort(hypotheses, dim=1).values

    @staticmethod
    def _predicted_support(auxiliary, depth):
        values = auxiliary['depth_values']
        if values.dim() == 2:
            values = values[:, :, None, None].expand_as(auxiliary['support_volume'])
        index = (values - depth.unsqueeze(1)).abs().argmin(dim=1, keepdim=True)
        return torch.gather(auxiliary['support_volume'], 1, index)

    def forward(self, imgs, proj_matrices, depth_values_orig):
        views = list(torch.unbind(imgs, 1))
        projections = list(torch.unbind(proj_matrices, 1))
        ref_img, src_imgs = views[0], views[1:]
        ref_proj, src_projs = projections[0], projections[1:]
        batch = ref_img.shape[0]
        packed = torch.cat(views, dim=0)
        feat8, feat4, feat2 = self.feature(packed)

        def split(features):
            return [features[i * batch:(i + 1) * batch] for i in range(len(views))]

        ref8, *src8 = split(feat8)
        ref4, *src4 = split(feat4)
        ref2, *src2 = split(feat2)
        base_interval = depth_values_orig[:, 1] - depth_values_orig[:, 0]
        global_start, global_end = depth_values_orig[:, 0], depth_values_orig[:, -1]

        d1_values = self._uniform_range(
            global_start, base_interval * self.interval_scales[0], self.depth_nums[0])
        depth1, conf1, pairs1, aux1 = self.stage1(
            ref8, _scale_projection(ref_proj, .5), src8,
            [_scale_projection(p, .5) for p in src_projs], d1_values,
            base_interval * self.interval_scales[0], self.mode)

        size4 = ref4.shape[-2:]
        if self.use_adaptive_search:
            d2_values = self._adaptive_range(
                aux1, depth1, conf1, size4, self.depth_nums[1],
                base_interval * self.interval_scales[1], global_start, global_end)
        else:
            center2 = F.interpolate(depth1.unsqueeze(1), size=size4,
                                    mode='bilinear', align_corners=False).squeeze(1)
            d2_values = self._local_range(
                center2, base_interval * self.interval_scales[1], self.depth_nums[1])
        depth2, conf2, pairs2, aux2 = self.stage2(
            ref4, ref_proj, src4, src_projs, d2_values,
            base_interval * self.interval_scales[1], self.mode)

        size2 = ref2.shape[-2:]
        if self.use_adaptive_search:
            d3_values = self._adaptive_range(
                aux2, depth2, conf2, size2, self.depth_nums[2],
                base_interval * self.interval_scales[2], global_start, global_end)
        else:
            center3 = F.interpolate(depth2.unsqueeze(1), size=size2,
                                    mode='bilinear', align_corners=False).squeeze(1)
            d3_values = self._local_range(
                center3, base_interval * self.interval_scales[2], self.depth_nums[2])
        depth3, conf3, pairs3, aux3 = self.stage3(
            ref2, _scale_projection(ref_proj, 2.0), src2,
            [_scale_projection(p, 2.0) for p in src_projs], d3_values,
            base_interval * self.interval_scales[2], self.mode)

        final_depth = depth3.unsqueeze(1)
        boundary_logits = refine_gate = None
        support = self._predicted_support(aux3, depth3)
        if self.boundary_refiner is not None:
            final_depth, boundary_logits, refine_gate = self.boundary_refiner(
                ref2, final_depth, conf3.unsqueeze(1), support,
                base_interval * self.interval_scales[2])
        auxiliary = {'stages': (aux1, aux2, aux3), 'support': support,
                     'boundary_logits': boundary_logits, 'refine_gate': refine_gate}
        outputs = [[depth1, pairs1], [depth2, pairs2], [depth3, pairs3]]
        confidence_maps = [
            F.interpolate(conf1.unsqueeze(1), size=size2, mode='bilinear', align_corners=False),
            F.interpolate(conf2.unsqueeze(1), size=size2, mode='bilinear', align_corners=False),
            conf3.unsqueeze(1)]
        return outputs, final_depth, confidence_maps, auxiliary


class VisMVSResearchLoss(nn.Module):
    def __init__(self, refined_weight=1.0, boundary_weight=0.1, occ_guide=False):
        super().__init__()
        self.baseline = BaselineLoss(occ_guide=occ_guide)
        self.refined_weight = refined_weight
        self.boundary_weight = boundary_weight

    @staticmethod
    def _boundary_target(depth):
        dx = F.pad((depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        gradient = torch.sqrt(dx.square() + dy.square() + 1e-6)
        flattened = gradient.reshape(gradient.shape[0], -1)
        kth = max(1, int(0.9 * flattened.shape[1]))
        scale = flattened.kthvalue(kth, dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-6)
        return (gradient >= scale).to(depth.dtype)

    def forward(self, stage_outputs, final_depth, auxiliary, depth_gt, mask, depth_interval):
        total, stats = self.baseline(stage_outputs, depth_gt, mask, depth_interval)
        target_size = final_depth.shape[-2:]
        gt = F.interpolate(depth_gt, size=target_size, mode='bilinear', align_corners=False)
        valid = F.interpolate(mask, size=target_size, mode='nearest') > .5
        logits = auxiliary.get('boundary_logits')
        if logits is not None:
            refined = F.smooth_l1_loss(
                (final_depth / depth_interval.view(-1, 1, 1, 1))[valid],
                (gt / depth_interval.view(-1, 1, 1, 1))[valid])
            total = total + self.refined_weight * refined
            stats['refined_loss'] = refined.detach()
            target = self._boundary_target(gt)
            boundary = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
            total = total + self.boundary_weight * boundary
            stats['boundary_loss'] = boundary.detach()
        stats['loss'] = total.detach()
        return total, stats


# Stable names for the training and evaluation entry points.
VisMVSModel = VisMVSResearchModel
VisMVSLoss = VisMVSResearchLoss
