# 三目 Vis-MVSNet 深度估计研究代码

本项目以 Vis-MVSNet 为基线，当前阶段固定使用三张输入图像：1 张参考图像和
2 张源图像，即 `nviews=3`。代码只保留基线、新方案、数据加载、训练、评估和
点云融合所需内容。

## 模型结构

新模型位于 `models/vismvsnet_research.py`，包含三个可以独立关闭的改进模块。

1. **M1：自适应多假设深度搜索**

   在第二、第三阶段同时保留局部精细候选、次峰候选和稀疏全局恢复候选。
   低置信度像素使用更大的局部搜索间隔，从而降低粗阶段估计错误后真值深度
   被排除出后续搜索范围的概率。

2. **M2：深度假设级可见性融合**

   为每个源视图、参考像素和候选深度预测独立的可见性权重
   `w_i(p,z)`。投影到源图像范围外的候选会被屏蔽；当两个源视图都缺少有效
   证据时，模型使用可学习的无证据状态，避免强制选择错误源视图。

3. **M3：可见性约束的边界细化**

   使用参考特征、深度、匹配置信度、跨视图支持度和边界概率预测有界深度
   残差。修正门控受跨视图支持度约束，主要处理遮挡边缘的前景扩张、背景污染
   和细结构缺失。

三个模块默认全部启用。`models/vismvsnet.py` 保留原 Vis-MVSNet 基础结构，
为新模型提供特征提取、代价体正则化和基线损失等公共组件。

## 三视图训练

在服务器进入项目目录后执行：

```bash
DATAPATH=/path/to/dtu GPU=0 BATCH_SIZE=4 bash train.sh
```

也可以直接执行：

```bash
python train.py \
  --dataset dtu_yao \
  --trainpath /path/to/dtu \
  --testpath /path/to/dtu \
  --trainlist lists/dtu/train.txt \
  --testlist lists/dtu/test.txt \
  --nviews 3 \
  --test_nviews 3 \
  --logdir checkpoints/dtu/vis_research_view3
```

训练和验证都会严格使用三张图像。默认检查点目录为：

```text
checkpoints/dtu/vis_research_view3
```

`train.sh` 中的服务器数据路径可以通过 `DATAPATH` 环境变量覆盖，不需要修改
脚本。

## 消融实验

所有消融实验应使用相同数据划分、随机种子、三视图输入、深度范围和训练轮数。

| 实验 | 追加参数 |
|---|---|
| Vis-MVSNet 基线 | `--disable_adaptive_search --disable_hypothesis_visibility --disable_boundary_refine` |
| 仅 M1 | `--disable_hypothesis_visibility --disable_boundary_refine` |
| 仅 M2 | `--disable_adaptive_search --disable_boundary_refine` |
| 仅 M3 | `--disable_adaptive_search --disable_hypothesis_visibility` |
| M1＋M2 | `--disable_boundary_refine` |
| M1＋M3 | `--disable_hypothesis_visibility` |
| M2＋M3 | `--disable_adaptive_search` |
| M1＋M2＋M3 | 不追加关闭参数 |

其中：

- `--global_candidate_ratio` 控制全局恢复候选比例，默认 `0.25`。
- `--secondary_candidate_ratio` 控制次峰候选比例，默认 `0.25`。
- `--refined_loss_weight` 控制细化深度损失权重，默认 `1.0`。
- `--boundary_loss_weight` 控制边界监督损失权重，默认 `0.1`。

## 模型评估

```bash
DATAPATH=/path/to/dtu_test \
GTPATH=/path/to/dtu/Depths \
CKPT=checkpoints/dtu/vis_research_view3/best_2mm.ckpt \
bash eval.sh
```

评估脚本默认同样使用三视图。若要评估某一消融模型，需要在 `eval.sh` 的命令
末尾添加与训练时完全相同的模块关闭参数。

## 上传服务器前的快速检查

```bash
python verify.py
```

该脚本不需要数据集，会使用小尺寸随机数据完成以下检查：

- 三视图完整模型前向传播；
- 损失计算和反向传播；
- Vis-MVSNet 基线路径；
- M1、M2、M3 单模块路径。

## 当前保留文件

- `models/vismvsnet_research.py`：三个改进点和新损失函数。
- `models/vismvsnet.py`：Vis-MVSNet 基线公共组件。
- `models/module.py`：单应性变换、相关性、深度回归和置信度计算。
- `train.py`、`train.sh`：三视图训练入口。
- `eval.py`、`eval.sh`：三视图深度评估入口。
- `fusion.py`：深度过滤和点云融合。
- `datasets/`：DTU 训练与评估数据加载。
- `lists/`：DTU 数据划分。
- `verify.py`：无需数据集的快速验证脚本。

