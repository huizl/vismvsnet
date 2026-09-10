"""Quick verification script for Vis-MVSNet model.

Usage:
    cd D:\seu\code\triple\vismvsnet
    python verify.py
"""
import torch
import sys
sys.path.insert(0, '.')

from models.vismvsnet import VisMVSModel, VisMVSLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Use smaller spatial size for CPU test
B, V, C, H, W = 1, 2, 3, 128, 160
D_orig = 48

imgs = torch.randn(B, V, C, H, W, device=device)
proj_matrices = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
depth_values_orig = torch.arange(425.0, 425.0 + D_orig * 2.5, 2.5, device=device).unsqueeze(0).repeat(B, 1)
depth_gt = torch.rand(B, 1, H, W, device=device) * 500 + 425
mask = torch.ones(B, 1, H, W, device=device)

print("=== Testing VisMVSModel ===")
model = VisMVSModel(
    mode='soft',
    stage1_depth_num=12, stage2_depth_num=8, stage3_depth_num=4,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

print("Forward pass...")
with torch.no_grad():
    outputs, final_depth, prob_maps = model(imgs, proj_matrices, depth_values_orig)

print(f"Stage outputs: {len(outputs)} stages")
for i, (depth, pairs) in enumerate(outputs):
    print(f"  Stage {i+1}: depth={depth.shape}, num_pairs={len(pairs)}")
    for j, (p_d, p_u, p_o) in enumerate(pairs):
        print(f"    Pair {j}: depth={p_d.shape}, uncert={p_u.shape}, occ={p_o.shape}")

print(f"Final depth: {final_depth.shape}")
print(f"Conf maps: {[p.shape for p in prob_maps]}")

print("\n=== Testing VisMVSLoss ===")
depth_interval = depth_values_orig[:, 1] - depth_values_orig[:, 0]
loss_fn = VisMVSLoss(occ_guide=False)
loss, stats = loss_fn(outputs, depth_gt, mask, depth_interval)

print(f"Loss: {loss.item():.6f}")
for k, v in stats.items():
    print(f"  {k}: {v.item():.6f}")

print("\n=== Testing backward pass ===")
# Re-run forward with gradients enabled for backward test
outputs2, _, _ = model(imgs, proj_matrices, depth_values_orig)
loss2, _ = loss_fn(outputs2, depth_gt, mask, depth_interval)
loss2.backward()
print("Backward pass OK")

# Test with different fusion modes
for mode in ['soft', 'average', 'maxpool', 'uwta']:
    print(f"\n=== Testing mode='{mode}' ===")
    m = VisMVSModel(
        mode=mode,
        stage1_depth_num=12, stage2_depth_num=8, stage3_depth_num=4,
    ).to(device)
    with torch.no_grad():
        outputs, final_depth, prob_maps = m(imgs, proj_matrices, depth_values_orig)
    print(f"  OK - final_depth: {final_depth.shape}")

print("\n=== All tests passed! ===")
