"""Quick verification script for Vis-MVSNet model.

Usage:
    cd D:\seu\code\triple\vismvsnet
    python verify.py
"""
import torch
import sys
sys.path.insert(0, '.')

from models import VisMVSModel, VisMVSLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Use smaller spatial size for CPU test
B, V, C, H, W = 1, 3, 3, 64, 80
D_orig = 48

imgs = torch.randn(B, V, C, H, W, device=device)
proj_matrices = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
depth_values_orig = torch.arange(425.0, 425.0 + D_orig * 2.5, 2.5, device=device).unsqueeze(0).repeat(B, 1)
depth_gt = torch.rand(B, 1, H, W, device=device) * 500 + 425
mask = torch.ones(B, 1, H, W, device=device)

print("=== Testing VisMVSModel ===")
model = VisMVSModel(
    mode='soft',
    stage1_depth_num=8, stage2_depth_num=8, stage3_depth_num=8,
).to(device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

print("Forward pass...")
with torch.no_grad():
    outputs, final_depth, prob_maps, auxiliary = model(
        imgs, proj_matrices, depth_values_orig)

print(f"Stage outputs: {len(outputs)} stages")
for i, (depth, pairs) in enumerate(outputs):
    print(f"  Stage {i+1}: depth={depth.shape}, num_pairs={len(pairs)}")
    for j, (p_d, p_u, p_o) in enumerate(pairs):
        print(f"    Pair {j}: depth={p_d.shape}, uncert={p_u.shape}, occ={p_o.shape}")

print(f"Final depth: {final_depth.shape}")
print(f"Conf maps: {[p.shape for p in prob_maps]}")
print(f"Support: {auxiliary['support'].shape}")

# Regression test for the training crash observed with batch_size=4 and D=16.
dummy_aux = {
    'depth_values': torch.randn(4, 16, 2, 3, device=device),
    'support_volume': torch.rand(4, 16, 2, 3, device=device),
}
dummy_depth = torch.randn(4, 2, 3, device=device)
dummy_support = model._predicted_support(dummy_aux, dummy_depth)
assert dummy_support.shape == (4, 1, 2, 3)
print("Batch-4 support lookup: OK")

print("\n=== Testing VisMVSLoss ===")
depth_interval = depth_values_orig[:, 1] - depth_values_orig[:, 0]
loss_fn = VisMVSLoss(occ_guide=False)
loss, stats = loss_fn(
    outputs, final_depth, auxiliary, depth_gt, mask, depth_interval)

print(f"Loss: {loss.item():.6f}")
for k, v in stats.items():
    print(f"  {k}: {v.item():.6f}")

print("\n=== Testing backward pass ===")
# Re-run forward with gradients enabled for backward test
outputs2, final_depth2, _, auxiliary2 = model(imgs, proj_matrices, depth_values_orig)
loss2, _ = loss_fn(
    outputs2, final_depth2, auxiliary2, depth_gt, mask, depth_interval)
loss2.backward()
print("Backward pass OK")

# Test baseline and single-module ablation paths.
for label, switches in [
        ('baseline', (False, False, False)),
        ('M1-search', (True, False, False)),
        ('M2-visibility', (False, True, False)),
        ('M3-boundary', (False, False, True))]:
    print(f"\n=== Testing {label} ===")
    m = VisMVSModel(
        mode='soft', stage1_depth_num=8, stage2_depth_num=8, stage3_depth_num=8,
        use_adaptive_search=switches[0],
        use_hypothesis_visibility=switches[1],
        use_boundary_refine=switches[2]).to(device).eval()
    with torch.no_grad():
        outputs, final_depth, prob_maps, _ = m(imgs, proj_matrices, depth_values_orig)
    print(f"  OK - final_depth: {final_depth.shape}")

print("\n=== All tests passed! ===")
