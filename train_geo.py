import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models.vismvsnet_geo import VisMVSModel, VisMVSLoss
from utils import *
import gc
import sys
import datetime


# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


cudnn.benchmark = True

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser(description='Vis-MVSNet PyTorch Implementation')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='vismvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='optional smoke-test limit; 0 runs the full training loader')
parser.add_argument('--max_test_batches', type=int, default=0,
                    help='optional smoke-test limit; 0 runs the full test loader')

# Vis-MVSNet specific
parser.add_argument('--vismode', type=str, default='soft',
                    choices=['soft', 'hard', 'average', 'uwta', 'maxpool'],
                    help='multi-view fusion mode')
parser.add_argument('--nviews', type=int, default=3, help='number of views')
parser.add_argument('--geo_alpha', type=float, default=0.3, help='geo fusion weight')
parser.add_argument('--stage1_geo_alpha', type=float, default=None,
                    help='optional stage 1 geo fusion weight')
parser.add_argument('--stage2_geo_alpha', type=float, default=None,
                    help='optional stage 2 geo fusion weight')
parser.add_argument('--stage3_geo_alpha', type=float, default=None,
                    help='optional stage 3 geo fusion weight')
parser.add_argument('--use_visibility_head', action='store_true',
                    help='enable the supervised pair-wise visibility head at stage 1')
parser.add_argument('--use_visibility_propagation', action='store_true',
                    help='propagate stage-1 visibility with residual corrections at stages 2 and 3')
parser.add_argument('--visibility_beta', type=float, default=0.5,
                    help='residual visibility gate strength in [0, 1]')
parser.add_argument('--visibility_stage2_beta', type=float, default=0.25)
parser.add_argument('--visibility_stage3_beta', type=float, default=0.1)
parser.add_argument('--visibility_stage2_residual_scale', type=float, default=0.25)
parser.add_argument('--visibility_stage3_residual_scale', type=float, default=0.1)
parser.add_argument('--visibility_stage2_loss_weight', type=float, default=0.5)
parser.add_argument('--visibility_stage3_loss_weight', type=float, default=0.25)
parser.add_argument('--visibility_loss_weight', type=float, default=0.2,
                    help='weight of the balanced focal visibility loss')
parser.add_argument('--visibility_focal_gamma', type=float, default=2.0)
parser.add_argument('--visibility_occ_abs_tol', type=float, default=2.0,
                    help='absolute GT depth tolerance in mm for visibility labels')
parser.add_argument('--visibility_occ_rel_tol', type=float, default=0.01,
                    help='relative GT depth tolerance for visibility labels')
parser.add_argument('--visibility_gt_downsample', type=int, default=8,
                    help='dataset downsample factor for pair-wise GT depth maps')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='train only the new visibility head')
parser.add_argument('--freeze_stage1_visibility', action='store_true',
                    help='keep a loaded stage-1 visibility head fixed')
parser.add_argument('--reset_visibility_residual_heads', action='store_true',
                    help='zero stage-2/3 occlusion heads after loading a checkpoint')
parser.add_argument('--stage1_dnum', type=int, default=48, help='stage 1 depth num')
parser.add_argument('--stage2_dnum', type=int, default=32, help='stage 2 depth num')
parser.add_argument('--stage3_dnum', type=int, default=16, help='stage 3 depth num')
parser.add_argument('--stage1_iscale', type=int, default=4, help='stage 1 interval scale')
parser.add_argument('--stage2_iscale', type=int, default=2, help='stage 2 interval scale')
parser.add_argument('--stage3_iscale', type=int, default=1, help='stage 3 interval scale')

args = parser.parse_args()

if args.freeze_backbone and not args.use_visibility_head:
    parser.error('--freeze_backbone requires --use_visibility_head')
if args.use_visibility_propagation and not args.use_visibility_head:
    parser.error('--use_visibility_propagation requires --use_visibility_head')
if args.freeze_stage1_visibility and not args.use_visibility_propagation:
    parser.error('--freeze_stage1_visibility requires --use_visibility_propagation')
if not 0.0 <= args.visibility_beta <= 1.0:
    parser.error('--visibility_beta must be in [0, 1]')
if args.visibility_loss_weight < 0.0:
    parser.error('--visibility_loss_weight must be non-negative')
if args.visibility_focal_gamma < 0.0:
    parser.error('--visibility_focal_gamma must be non-negative')
if args.visibility_occ_abs_tol < 0.0 or args.visibility_occ_rel_tol < 0.0:
    parser.error('visibility tolerances must be non-negative')
for name, value in (
        ('visibility_stage2_beta', args.visibility_stage2_beta),
        ('visibility_stage3_beta', args.visibility_stage3_beta)):
    if not 0.0 <= value <= 1.0:
        parser.error('--{} must be in [0, 1]'.format(name))
for name, value in (
        ('visibility_stage2_residual_scale', args.visibility_stage2_residual_scale),
        ('visibility_stage3_residual_scale', args.visibility_stage3_residual_scale),
        ('visibility_stage2_loss_weight', args.visibility_stage2_loss_weight),
        ('visibility_stage3_loss_weight', args.visibility_stage3_loss_weight)):
    if value < 0.0:
        parser.error('--{} must be non-negative'.format(name))

os.makedirs(args.logdir, exist_ok=True)
sys.stdout = Logger(os.path.join(args.logdir, "logs.txt"))

if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# =============================================================================
# Dataset, Model, Optimizer
# =============================================================================
MVSDataset = find_dataset_def(args.dataset)
visibility_dataset_kwargs = dict(
    return_visibility_gt=args.use_visibility_head,
    visibility_downsample=args.visibility_gt_downsample,
)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train",
                           args.nviews, args.numdepth, args.interval_scale,
                           **visibility_dataset_kwargs)
test_dataset = MVSDataset(args.testpath, args.testlist, "test",
                          5, args.numdepth, args.interval_scale,
                          **visibility_dataset_kwargs)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                            num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                           num_workers=4, drop_last=False)

model = VisMVSModel(
    mode=args.vismode, use_geo=True, geo_alpha=args.geo_alpha,
    use_visibility_head=args.use_visibility_head,
    visibility_beta=args.visibility_beta,
    visibility_stages=(1,),
    use_visibility_propagation=args.use_visibility_propagation,
    visibility_stage_betas=(
        args.visibility_beta,
        args.visibility_stage2_beta,
        args.visibility_stage3_beta,
    ),
    visibility_residual_scales=(
        0.0,
        args.visibility_stage2_residual_scale,
        args.visibility_stage3_residual_scale,
    ),
    geo_occ_abs_tol=args.visibility_occ_abs_tol,
    geo_occ_rel_tol=args.visibility_occ_rel_tol,
    stage1_depth_num=args.stage1_dnum,
    stage1_interval_scale=args.stage1_iscale,
    stage2_depth_num=args.stage2_dnum,
    stage2_interval_scale=args.stage2_iscale,
    stage3_depth_num=args.stage3_dnum,
    stage3_interval_scale=args.stage3_iscale,
)

for stage_name, stage_alpha in zip(
        ('stage1', 'stage2', 'stage3'),
        (args.stage1_geo_alpha, args.stage2_geo_alpha, args.stage3_geo_alpha)):
    if stage_alpha is not None:
        if not 0.0 <= stage_alpha <= 1.0:
            raise ValueError('{} geo alpha must be in [0, 1]'.format(stage_name))
        getattr(model, stage_name).geo_alpha = stage_alpha

if args.freeze_backbone:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not args.freeze_stage1_visibility:
        for parameter in model.stage1.visibility_head.parameters():
            parameter.requires_grad = True
    if args.use_visibility_propagation:
        for stage in (model.stage2, model.stage3):
            for parameter in stage.uncert_net.occ_head.parameters():
                parameter.requires_grad = True

if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()
model_loss = VisMVSLoss(
    occ_guide=False,
    visibility_loss_weight=args.visibility_loss_weight if args.use_visibility_head else 0.0,
    visibility_focal_gamma=args.visibility_focal_gamma,
    visibility_occ_abs_tol=args.visibility_occ_abs_tol,
    visibility_occ_rel_tol=args.visibility_occ_rel_tol,
    visibility_stage_loss_weights=(
        1.0,
        args.visibility_stage2_loss_weight,
        args.visibility_stage3_loss_weight,
    ),
)
trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
if not trainable_parameters:
    raise RuntimeError('No trainable model parameters were selected')
optimizer = optim.Adam(trainable_parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# =============================================================================
# Load checkpoint
# =============================================================================
start_epoch = 0


def load_model_state(state_dict, allow_visibility_init=False):
    target_has_module = next(iter(model.state_dict())).startswith('module.')
    source_has_module = next(iter(state_dict)).startswith('module.')
    if target_has_module and not source_has_module:
        state_dict = {'module.' + key: value for key, value in state_dict.items()}
    elif source_has_module and not target_has_module:
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=not allow_visibility_init)
    if allow_visibility_init:
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        invalid_missing = [key for key in missing if '.visibility_head.' not in key]
        if unexpected or invalid_missing:
            raise RuntimeError(
                'Checkpoint mismatch. unexpected={}, missing={}'.format(
                    unexpected, invalid_missing))
        if missing:
            print('Initialized new visibility head parameters:', missing)


def reset_visibility_residual_heads():
    if not args.use_visibility_propagation:
        raise ValueError(
            '--reset_visibility_residual_heads requires visibility propagation')
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    for stage_name in ('stage2', 'stage3'):
        occ_head = getattr(target_model, stage_name).uncert_net.occ_head
        nn.init.zeros_(occ_head.weight)
        if occ_head.bias is not None:
            nn.init.zeros_(occ_head.bias)
    print('Reset stage2/stage3 visibility residual heads to zero.')


if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    load_model_state(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    load_model_state(
        state_dict['model'],
        allow_visibility_init=args.use_visibility_head)
if args.reset_visibility_residual_heads:
    reset_visibility_residual_heads()
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
print('Number of trainable parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters() if p.requires_grad])))


# =============================================================================
# Best-metric tracking
# =============================================================================
best_metrics = {
    "abs_depth_error": 1e9,
    "thres2mm_error": 1e9,
    "thres4mm_error": 1e9,
    "thres8mm_error": 1e9,
}


def save_best_models(avg_test, epoch_idx):
    abs_err = avg_test["abs_depth_error"]
    th2 = avg_test["thres2mm_error"]
    th4 = avg_test["thres4mm_error"]
    th8 = avg_test["thres8mm_error"]

    if abs_err < best_metrics["abs_depth_error"]:
        best_metrics["abs_depth_error"] = abs_err
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_abs.ckpt"))
        print(f"  Best ABS updated: {abs_err:.4f}")

    if th2 < best_metrics["thres2mm_error"]:
        best_metrics["thres2mm_error"] = th2
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_2mm.ckpt"))
        print(f"  Best 2mm updated: {th2:.4f}")

    if th4 < best_metrics["thres4mm_error"]:
        best_metrics["thres4mm_error"] = th4
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_4mm.ckpt"))
        print(f"  Best 4mm updated: {th4:.4f}")

    if th8 < best_metrics["thres8mm_error"]:
        best_metrics["thres8mm_error"] = th8
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_8mm.ckpt"))
        print(f"  Best 8mm updated: {th8:.4f}")


# =============================================================================
# Train / Test loops
# =============================================================================
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # ---- training ----
        for batch_idx, sample in enumerate(TrainImgLoader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(
                epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))

        # ---- save checkpoint ----
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                       "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # ---- testing ----
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
                break
            start_time = time.time()
            loss, scalar_outputs = test_sample(sample, detailed_summary=True)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
                epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss,
                time.time() - start_time))

        avg_test = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test, global_step)
        print("avg_test_scalars:", avg_test)

        save_best_models(avg_test, epoch_idx)

        if epoch_idx == args.epochs - 1:
            torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                       os.path.join(args.logdir, "latest.ckpt"))

        gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
            break
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
            batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(
                batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    if args.freeze_backbone:
        model.eval()
        if not args.freeze_stage1_visibility:
            model.module.stage1.visibility_head.train()
        if args.use_visibility_propagation:
            model.module.stage2.uncert_net.occ_head.train()
            model.module.stage3.uncert_net.occ_head.train()
    else:
        model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"].unsqueeze(1)   # [B, 1, H, W]
    mask = sample_cuda["mask"].unsqueeze(1)         # [B, 1, H, W]
    depth_values = sample_cuda["depth_values"]

    # extract base depth interval for loss normalisation
    depth_interval = depth_values[:, 1] - depth_values[:, 0]  # [B]

    outputs, final_depth, prob_maps = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_values)

    visibility_kwargs = {}
    if args.use_visibility_head:
        visibility_kwargs = dict(
            visibility_depths=sample_cuda["visibility_depths"],
            visibility_masks=sample_cuda["visibility_masks"],
            proj_matrices=sample_cuda["proj_matrices"],
        )
    loss, scalar_outputs = model_loss(
        outputs, depth_gt, mask, depth_interval, **visibility_kwargs)
    loss.backward()
    optimizer.step()

    if detailed_summary:
        # use stage 3 (final) depth for metrics
        depth_est = outputs[-1][0]  # stage 3 depth [B, H_stage, W_stage]
        # upsample to gt resolution
        depth_est_full = F.interpolate(
            depth_est.unsqueeze(1),
            size=(depth_gt.shape[2], depth_gt.shape[3]),
            mode='bilinear', align_corners=False).squeeze(1)

        valid = mask.squeeze(1) > 0.5
        scalar_outputs["abs_depth_error"] = tensor2float(
            AbsDepthError_metrics(depth_est_full, depth_gt.squeeze(1), valid))
        e2 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 2))
        e4 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 4))
        e8 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 8))
        scalar_outputs["thres2mm_error"] = e2
        scalar_outputs["thres4mm_error"] = e4
        scalar_outputs["thres8mm_error"] = e8
        scalar_outputs["thres2mm_acc"] = 1.0 - e2
        scalar_outputs["thres4mm_acc"] = 1.0 - e4
        scalar_outputs["thres8mm_acc"] = 1.0 - e8

    return tensor2float(loss), tensor2float(scalar_outputs)


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"].unsqueeze(1)
    mask = sample_cuda["mask"].unsqueeze(1)
    depth_values = sample_cuda["depth_values"]

    depth_interval = depth_values[:, 1] - depth_values[:, 0]

    outputs, final_depth, prob_maps = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_values)

    visibility_kwargs = {}
    if args.use_visibility_head:
        visibility_kwargs = dict(
            visibility_depths=sample_cuda["visibility_depths"],
            visibility_masks=sample_cuda["visibility_masks"],
            proj_matrices=sample_cuda["proj_matrices"],
        )
    loss, scalar_outputs = model_loss(
        outputs, depth_gt, mask, depth_interval, **visibility_kwargs)

    if detailed_summary:
        depth_est = outputs[-1][0]
        depth_est_full = F.interpolate(
            depth_est.unsqueeze(1),
            size=(depth_gt.shape[2], depth_gt.shape[3]),
            mode='bilinear', align_corners=False).squeeze(1)

        valid = mask.squeeze(1) > 0.5
        scalar_outputs["abs_depth_error"] = tensor2float(
            AbsDepthError_metrics(depth_est_full, depth_gt.squeeze(1), valid))
        e2 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 2))
        e4 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 4))
        e8 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 8))
        scalar_outputs["thres2mm_error"] = e2
        scalar_outputs["thres4mm_error"] = e4
        scalar_outputs["thres8mm_error"] = e8
        scalar_outputs["thres2mm_acc"] = 1.0 - e2
        scalar_outputs["thres4mm_acc"] = 1.0 - e4
        scalar_outputs["thres8mm_acc"] = 1.0 - e8

    return tensor2float(loss), tensor2float(scalar_outputs)


# =============================================================================
# Entry
# =============================================================================
if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
