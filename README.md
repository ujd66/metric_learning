# Point Cloud Metric Learning

工业点云分类项目，基于自定义 PointNet baseline，支持 19 个 known class + 1 个 negative class。

## 环境要求

- Python >= 3.11
- PyTorch >= 2.7.0+cu128
- uv (包管理)

## 安装

```bash
cd pointcloud_metric_learning
uv sync
```

## 数据准备

### 方式一：PCD 转 NPY（推荐）

将原始 PCD 数据按以下结构放置：

```
raw_pcd_dataset/
  train/
    class_000/
      xxx.pcd
    class_001/
      xxx.pcd
    ...
    negative/
      xxx.pcd
  val/
    ...
  test/
    ...
```

每个 PCD 样本目录可包含伴随文件（`xxx_crop.png`, `xxx_meta.json`），转换时会自动记录路径。

运行转换：

```bash
python scripts/prepare_pcd_dataset.py \
  --input-dir raw_pcd_dataset \
  --output-dir dataset \
  --splits train val test

# 如需包含 intensity 通道 (xyz + intensity -> 4 channels)：
python scripts/prepare_pcd_dataset.py \
  --input-dir raw_pcd_dataset \
  --output-dir dataset \
  --use_intensity
```

转换后生成 `dataset/` 目录，每个 `.npy` 文件为 dict 格式：

```python
{
  "points": numpy.ndarray,          # shape [N, 3] 或 [N, 4]
  "label": int,
  "class_name": str,
  "sample_id": str,
  "source_pcd": str,
  "heightmap_path": str 或 None,
  "info_path": str 或 None,
  "transform_path": str 或 None
}
```

### 方式二：直接使用 NPY

将数据集按以下结构放置：

```
dataset/
  train/
    class_000/
      xxx.npy
    class_001/
      xxx.npy
    ...
    negative/
      xxx.npy
  val/
    ...
  test/
    ...
```

每个 `.npy` 文件支持两种格式：

1. dict 格式（含 `points`, `label`, `class_name`, `sample_id`）
2. 纯 ndarray 格式（shape `[N, 3]` 或 `[N, 6]`，label 从目录名推断）

### 查看 PCD 样本信息

```bash
python scripts/inspect_pcd_sample.py --input raw_pcd_dataset/train/class_000/000003.pcd
```

## 推荐数据流程

### 1. PCD 检查

确认 PCD 文件能正常读取，点数、坐标范围合理：

```bash
python scripts/inspect_pcd_sample.py --input label/cewenqi/000018/000018_0001_crop.pcd
```

### 2. 生成统一数据池（dataset_all）

将所有 PCD 转成 NPY 格式到一个统一目录，不做 split。自动将 negative 类（qita/other 等）映射到 `negative/` 目录。

```bash
python scripts/prepare_pcd_dataset.py \
  --flat \
  --input-root raw_pcd_dataset \
  --output-root dataset_all \
  --config configs/config.yaml
```

输出结构：

```
dataset_all/
  class_000/
  class_001/
  ...
  class_018/
  negative/
```

### 3. 重新分层划分（train/val/test）

用 `rebuild_split.py` 自动按类别分层划分，确保每个类别都进入 train/val/test：

```bash
python scripts/rebuild_split.py \
  --input-root dataset_all \
  --output-root dataset \
  --config configs/config.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

划分策略：
- 样本充足的类别：按比例划分
- count >= 3：至少 train 1, val 1, test 1
- count == 2：train 1, val 1（warning）
- count == 1：仅 train（warning）
- **不会复制样本到 val/test**，避免数据泄漏

### 4. 验证数据划分

```bash
python scripts/validate_dataset.py \
  --dataset-root dataset \
  --config configs/config.yaml
```

输出：
- 每个 split 的总样本数
- 每个类别在 train/val/test 的样本数
- negative 类是否存在
- 缺失类别
- 类别不均衡比例
- 样本过少警告

报告保存到 `outputs/reports/dataset_validation.json`。

## 配置

编辑 `configs/config.yaml` 修改超参数。编辑 `configs/class_names.json` 修改类别名称。

### 类别映射

`config.yaml` 中的 `label_mapping` 控制哪些类被视为 negative：

```yaml
label_mapping:
  negative_names: ["negative", "qita", "other", "others", "其他"]
  force_negative_label: 19
```

- 如果原始数据的类名匹配 `negative_names`，自动映射到 label 19
- 输出目录统一为 `negative/`
- 标签编号：0-18 为 19 个 known class，19 为 negative

### 类别权重

训练时可启用 class_weight 来应对不均衡：

```yaml
train:
  use_class_weight: true
  class_weight_max: 10.0
```

权重按逆频率计算并 clip 到 `[1.0, class_weight_max]`，防止极小类权重爆炸。

## Baseline 诊断流程

在引入 metric learning 之前，需要确认 CrossEntropy baseline 能在真实数据上正常学习。推荐流程：

### 1. 验证数据集

```bash
python scripts/validate_dataset.py \
  --dataset-root dataset \
  --config configs/config.yaml
```

检查类别覆盖、不均衡、metric learning suitability。

### 2. 过拟合小实验

用 32 个样本验证模型能正常收敛：

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --overfit-small-batch
```

- 如果 acc 接近 100%：PASS，代码和数据基本正常
- 如果 acc > 50% 但不到 100%：PARTIAL，检查 lr / augmentation
- 如果 acc < 50%：FAIL，检查标签、预处理、模型代码

### 3. 完整训练

```bash
python scripts/train.py --config configs/config.yaml
```

每个 epoch 输出：
- Train Loss / Train Acc
- Val Loss / Val Acc
- Val Macro Precision / Recall / F1
- Val Known-class Accuracy
- Val Negative Accuracy（无 negative 时显示 N/A）
- Learning Rate

训练曲线保存到 `outputs/reports/training_curves.png`。

### 4. 评价

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

输出文件：
- `outputs/reports/evaluation.json` — 全部指标
- `outputs/reports/confusion_matrix.png` — 混淆矩阵
- `outputs/reports/confusion_matrix_normalized.png` — 归一化混淆矩阵
- `outputs/reports/per_class_metrics.csv` — 每类指标

### 5. Embedding 可视化

```bash
python scripts/visualize_embeddings.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

输出：
- `outputs/reports/embedding_tsne_test.png` — t-SNE 可视化
- `outputs/reports/embedding_features.npy` — embedding 向量
- `outputs/reports/embedding_labels.npy` — 对应标签

### 判断 baseline 是否正常

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| Overfit small batch acc | > 90% | < 50% 说明代码/标签有问题 |
| Val macro F1 | > 0.3（20 类不均衡） | 始终 < 0.1 |
| Train loss | 持续下降 | 不下降或 NaN |
| Val loss | 先降后稳 | 一直上升（严重过拟合） |
| t-SNE 可视化 | 同类有一定聚集 | 完全随机分布 |

### 满足什么条件可以进入 metric learning

1. Overfit small batch 能到 90%+
2. 完整训练 val macro F1 合理（至少 > 0.2）
3. t-SNE 显示同类有一定聚集趋势
4. 没有训练异常（loss NaN、acc 不动等）

## 2.1 轮：Metric Learning 训练（CE + SupCon）

在 baseline 确认正常后，加入 Supervised Contrastive Loss 进行 metric learning 增强。

### 原理

训练总 loss：

```
total_loss = ce_weight * CE_loss + metric_weight * SupCon_loss
```

- `CE_loss`：标准交叉熵，包含所有类别（含 negative）
- `SupCon_loss`：Supervised Contrastive Loss，默认排除 negative 类
- `warmup_epochs_for_metric_loss`：前 N 个 epoch 只用 CE，之后加入 metric loss
- `temperature`：控制 contrastive loss 对 hard negatives 的关注程度

### 配置

`configs/config.yaml` 中控制 metric learning 的部分：

```yaml
experiment:
  name: "pointnet_ce_supcon"
  output_subdir: "ce_supcon"

loss:
  ce_weight: 1.0
  metric_type: "supcon"
  metric_weight: 0.05
  temperature: 0.07
  include_negative_in_metric_loss: false
  warmup_epochs_for_metric_loss: 10

metric_learning:
  enabled: true
  normalize_embedding: true
  exclude_negative: true

sampler:
  use_pk_sampler: true
  classes_per_batch: 8
  samples_per_class: 4
  include_negative: false
  drop_singleton_classes: true
```

关闭 metric learning（恢复 baseline 行为）：

```yaml
metric_learning:
  enabled: false
sampler:
  use_pk_sampler: false
```

### 1. 运行 CE + SupCon 训练

```bash
python scripts/train.py --config configs/config.yaml
```

输出到 `outputs/runs/ce_supcon/`：
- `training_curves.png` — 训练曲线（含 metric loss 曲线）
- `training_history.json` — 每 epoch 详细指标
- `checkpoints/best.pt` — best model
- `checkpoints/last.pt` — last model
- `best_val_per_class_metrics.json` — best checkpoint 的每类指标

每个 epoch 输出：
- Train Total Loss (CE Loss) Acc
- Metric Loss (w=current_weight)
- Val Loss / Acc
- Macro P / R / F1
- Known Acc / Neg Acc
- LR

### 2. 评价分类效果

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/runs/ce_supcon/checkpoints/best.pt \
  --split test \
  --output-dir outputs/runs/ce_supcon/eval_test
```

### 3. 生成 Embedding 评估

```bash
# 先提取 embedding
python scripts/visualize_embeddings.py \
  --config configs/config.yaml \
  --checkpoint outputs/runs/ce_supcon/checkpoints/best.pt \
  --split test \
  --output-dir outputs/runs/ce_supcon/embedding_test

# 然后评估 embedding 质量
python scripts/evaluate_embeddings.py \
  --input outputs/runs/ce_supcon/embedding_test/embedding_features.npy \
  --output-dir outputs/runs/ce_supcon/embedding_eval_test \
  --normalize true \
  --exclude-negative true \
  --negative-label 19 \
  --topk 1 3 5 10 \
  --class-names configs/class_names.json
```

注意：evaluate_embeddings.py 的输入需要同时包含 embedding 和 labels。更方便的方式是手动组装 npz：

```python
import numpy as np
features = np.load("outputs/runs/ce_supcon/embedding_test/embedding_features.npy")
labels = np.load("outputs/runs/ce_supcon/embedding_test/embedding_labels.npy")
np.savez("outputs/runs/ce_supcon/embedding_eval_test/embeddings.npz",
    embeddings=features, labels=labels)
```

```bash
python scripts/evaluate_embeddings.py \
  --input outputs/runs/ce_supcon/embedding_eval_test/embeddings.npz \
  --output-dir outputs/runs/ce_supcon/embedding_eval_test \
  --normalize true \
  --exclude-negative true \
  --negative-label 19 \
  --topk 1 3 5 10 \
  --class-names configs/class_names.json \
  --config configs/config.yaml
```

### 4. 和 Baseline 对比

```bash
python scripts/compare_embedding_reports.py \
  --baseline-json outputs/reports/embedding_eval/metrics_summary.json \
  --new-json outputs/runs/ce_supcon/embedding_eval_test/metrics_summary.json \
  --output outputs/runs/ce_supcon/comparison_report.html
```

输出：
- `comparison_report.html` — 可视化对比报告（提升/下降/不变标记）
- `comparison_report_metrics.csv` — 对比数据 CSV

### 5. 建议尝试的超参数

| 实验 | metric_weight | temperature | warmup_epochs |
|------|---------------|-------------|---------------|
| CE baseline | 0 (disabled) | - | - |
| CE + SupCon (保守) | 0.05 | 0.07 | 10 |
| CE + SupCon (激进) | 0.1 | 0.07 | 10 |

### PKSampler 说明

PKSampler 确保每个 batch 包含 P 个类别、每类 K 个样本：

- `classes_per_batch = 8`，`samples_per_class = 4` → batch_size = 32
- 少样本类会使用有放回采样
- 默认排除 singleton 类和 negative 类
- 如果可用类别不足 P，自动降低 P 并给出 warning

### Negative 类处理

- CrossEntropyLoss 始终包含 negative label（分类需要识别 negative）
- SupConLoss 默认排除 negative 样本（negative 内部多样性大，不应强制聚成一团）
- 设置 `include_negative_in_metric_loss: true` 可让 negative 参与 metric loss

### Focus Class Pairs

在 `config.yaml` 中配置关注的混淆类别对：

```yaml
analysis:
  focus_class_pairs:
    - ["changtiaofalan", "teshujiaqiangtieduantou"]
```

evaluate_embeddings.py 会输出这些类别对的详细分析：
- 平均/最大/最小相似度
- 最相似的样本对 (CSV)
- 对比前后的相似度变化

### 进入下一阶段的标准

满足以下所有条件时，可以进入 prototype / unknown rejection 阶段：

1. Val Macro F1 不低于 baseline 超过 3 个百分点
2. Recall@1 不低于 baseline
3. Similarity gap 不低于 baseline
4. Top confusing pair similarity 有下降，或者至少不升高
5. Negative accuracy 不明显恶化
6. 训练稳定，无 NaN

## 2.2 轮：Prototype 构建与 OOD / Unknown Rejection

基于已训练模型的 embedding 构建 19 个 known class 的 prototype，支持推理时根据 prototype similarity 做 unknown rejection。

**当前使用 baseline checkpoint（不是 CE+SupCon），因为 baseline embedding 质量更好。**

### 数据限制说明

> **重要：当前 negative 样本非常少，因此：**
> - negative accuracy 不稳定
> - negative_reject_rate 不稳定
> - threshold 对 negative 的校准不可靠
> - unknown rejection 更适合看 known_accept_rate 和 nearest_similarity 分布
> - 后续应补充更多 negative / unknown 样本

### 1. 构建 Prototypes

从 train split 提取 embedding，为每个 known class 计算均值 prototype：

```bash
python scripts/build_prototypes.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split train \
  --output outputs/prototypes/baseline_prototypes.pt
```

输出 `baseline_prototypes.pt` 包含：
- `prototypes`: Tensor [19, 256]，每个 known class 的 L2-normalized prototype
- `class_names`: 19 个类别名称
- `class_support`: 每个类别的样本数
- 不包含 negative 类的 prototype

### 2. 搜索 Similarity Threshold

在 val split 上搜索最优 similarity threshold：

```bash
python scripts/search_threshold.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --split val \
  --output outputs/prototypes/baseline_threshold.json
```

搜索策略：
- 默认选择 `known_accept_rate >= 0.95` 的 threshold 中，`negative_reject_rate` 最高的
- 如果 val 中没有 negative 样本，选择保证 95% known 被接受的最高 threshold
- threshold 范围 0.1 到 0.99，步长 0.01

### 3. 评估 OOD / Unknown Rejection

在 test split 上评估完整的推理逻辑：

```bash
python scripts/evaluate_ood.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold.json \
  --split test \
  --output-dir outputs/reports/ood_eval_baseline_test
```

输出指标：

| 指标 | 含义 |
|------|------|
| known_accept_rate | known 样本被 prototype 接受的比例 |
| known_reject_rate | known 样本被误判为 unknown 的比例 |
| known_classification_accuracy_after_accept | 被接受的 known 样本中分类正确的比例 |
| negative_reject_rate | negative 样本被拒绝的比例 |
| false_known_rate | negative 样本被误判为 known 的比例 |
| AUROC | known/negative 的区分度 |

输出文件：
- `ood_metrics.json` — 全部指标
- `per_sample_predictions.csv` — 每个样本的预测详情
- `per_class_ood_metrics.csv` — 每类指标
- `nearest_similarity_histogram.png` — known vs negative 相似度分布
- `threshold_curve.png` — threshold 搜索曲线
- `final_confusion_matrix.png` — 最终混淆矩阵（含 unknown）
- `report.html` — 完整 HTML 报告

### 4. 单样本推理

支持 prototype-based unknown rejection 的推理：

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold.json \
  --input raw_pcd_dataset/test/class_000/000003.pcd
```

推理逻辑：
1. 分类头预测 `pred_cls`
2. 如果 `pred_cls == negative_label (19)` → `"negative"`
3. 否则计算 embedding 与所有 known prototype 的 cosine similarity
4. 如果 `max(similarity) < threshold` → `"unknown"`
5. 否则 → `"known"`，使用最近 prototype 的类别

输出 JSON：
- `final_type`: `"known"` / `"negative"` / `"unknown"`
- `final_label`: 最终预测类别名称
- `reason`: 判定原因
- `classifier_pred`: 分类头预测
- `nearest_known_class`: 最近 prototype 类别
- `nearest_similarity`: 与最近 prototype 的相似度
- `similarity_threshold`: 使用的阈值
- `top5_classifier_probs`: 分类头 top-5
- `top5_prototype_similarities`: prototype top-5

不带 `--prototypes` 时退化为纯分类器推理。

### 配置

```yaml
prototype:
  enabled: true
  build_split: "train"
  normalize_embeddings: true
  use_known_only: true

ood:
  enabled: true
  similarity_threshold: 0.65
  threshold_min: 0.1
  threshold_max: 0.99
  threshold_step: 0.01
  target_known_accept_rate: 0.95
  threshold_selection: "max_negative_rejection_under_known_accept_constraint"
```

### 如何解读 OOD 结果

- **known_accept_rate >= 0.95**：大部分 known 样本能正确通过 prototype 匹配
- **known_reject_rate < 0.05**：prototype threshold 不会过度拒绝已知类
- **negative_reject_rate**：当前因样本不足仅供参考，不作为主要判断依据
- **AUROC**：如果可计算，> 0.9 表示 embedding 空间能有效区分 known/negative
- **相似度分布直方图**：查看 known 和 negative 的 nearest similarity 是否有明显分离

## 2.3 轮：改进 Prototype/OOD Threshold 搜索和评估

在 2.2 基础上，改进 threshold 搜索策略、增加多阈值评估、相似度分位数报告和风险分析。

### Threshold 搜索策略

支持 4 种策略（通过 `config.yaml` 的 `ood.threshold_selection` 配置）：

| 策略 | 说明 |
|------|------|
| `known_quantile` | 使用 val known similarity 的分位数作为 threshold |
| `max_negative_rejection_under_known_accept_constraint` | 在 known_accept >= target 的 threshold 中选 negative_reject 最高的 |
| `best_balanced_score` | 选 known_accept + negative_reject 最大的 |
| `manual` | 直接使用 `ood.similarity_threshold` |

### 当前 negative 很少时的推荐

当 negative 样本不足时（如当前只有 3 个 test negative），不应只依赖 `negative_reject_rate`。推荐使用 `known_quantile`：

| 分位数 | 含义 | 适用场景 |
|--------|------|----------|
| `p05` | 保留 ~95% known | 业务更怕误收 unknown |
| `p01` | 保留 ~99% known | 业务更怕误拒 known |
| `p10` | 保留 ~90% known | 对 known 接受率要求不高 |

### 配置

```yaml
ood:
  threshold_selection: "known_quantile"
  known_quantile: 0.05
  candidate_thresholds: [0.68, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96]
  min_known_accept_rate: 0.95
```

### 1. 运行 known_quantile threshold 搜索

```bash
python scripts/search_threshold.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --split val \
  --output outputs/prototypes/baseline_threshold_p05.json
```

输出 JSON 包含：
- `selected_threshold`：选定的 threshold
- `selection_strategy`：使用的策略名
- `known_similarity_quantiles`：known 样本的 p01/p05/p10/p50/p90/p95/p99
- `negative_similarity_quantiles`：negative 样本分位数（如果有）
- `candidate_threshold_results`：候选 threshold 的指标对比
- `threshold_curve`：全范围搜索曲线
- `warnings`：negative 样本不足时的提醒

### 2. 运行多 threshold sweep 评估

```bash
python scripts/evaluate_ood.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --split test \
  --output-dir outputs/reports/ood_eval_baseline_test_p05 \
  --thresholds 0.68,0.75,0.80,0.85,0.90,0.92,0.94,0.96
```

输出额外文件：
- `threshold_sweep_metrics.csv` — 每个 threshold 的完整指标
- `threshold_sweep_metrics.json` — JSON 格式
- `threshold_sweep_report.html` — 可视化对比报告（高亮推荐/best/balanced）
- `threshold_sweep_plot.png` — sweep 曲线图

### 3. 风险分析输出

每个评估都会输出：

| 文件 | 内容 |
|------|------|
| `per_sample_predictions.csv` | 增加 `nearest_similarity`, `margin_to_threshold`, `risk_level` |
| `hard_negative_samples.csv` | 所有 negative 但被误判为 known 的样本 |
| `near_boundary_known_samples.csv` | margin 最小的前 50 个 known 样本 |

风险等级规则：

| 风险等级 | 条件 |
|----------|------|
| `safe_known` | known 样本，margin >= 0.03 |
| `near_boundary_known` | known 样本，0 <= margin < 0.03 |
| `rejected_known` | known 样本，similarity < threshold |
| `safe_rejected_negative` | negative 样本，margin <= -0.03 |
| `near_boundary_negative` | negative 样本，-0.03 < margin < 0 |
| `false_known_negative` | negative 样本，similarity >= threshold |

### 4. 改进的图表

- **threshold_curve.png**：增加 known_reject_rate、false_known_rate、P05 线、manual threshold 线
- **nearest_similarity_histogram.png**：增加 threshold 竖线
- **report.html**：增加相似度分位数卡片

### 如何选择最终 threshold

1. 先看 `known_similarity_quantiles`：p05 ~0.94，p01 ~0.93
2. 如果业务更怕误拒 known → 用 p01（~0.93），保证 99% known 被接受
3. 如果业务更怕误收 unknown → 用 p05（~0.94），只保证 95% known
4. 用 `--thresholds` sweep 对比不同 threshold 在 test 上的实际表现
5. 重点关注 `known_accept_rate` 和 `known_classification_accuracy_after_accept`

## 2.4 轮：Query-Gallery Retrieval 评估与部署包导出

在 prototype/OOD 基础上，增加更接近实际部署的 query-gallery retrieval 评估，并导出可部署的 inference bundle。

### 1. 构建 Gallery

从 train split 提取 embedding 构建检索库：

```bash
python scripts/build_gallery.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split train \
  --output outputs/gallery/baseline_train_gallery.pt
```

输出 `baseline_train_gallery.pt` 包含：
- `embeddings`: Tensor [N, 256]，L2-normalized
- `labels`: Tensor [N]
- `class_names`, `sample_ids`, `source_paths`
- 默认只包含 known classes，排除 negative
- 加 `--include-negative` 可包含 negative

### 2. Query-Gallery Retrieval 评估

```bash
python scripts/evaluate_retrieval.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --gallery outputs/gallery/baseline_train_gallery.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --split test \
  --output-dir outputs/reports/retrieval_eval_baseline_test
```

评估逻辑：对每个 query 样本，计算与 gallery 所有样本的 cosine similarity，找 top-K 最近邻。如果最近邻 similarity < threshold → "unknown"，否则使用最近邻的类别。

输出指标：

| 指标 | 含义 |
|------|------|
| known_accept_rate | known query 被 gallery 接受的比例 |
| top1_accuracy_on_accepted | 被接受的 known query 中 top-1 正确的比例 |
| recall@K / precision@K | top-K 检索指标 |
| negative_reject_rate | negative query 被拒绝的比例 |
| final_accuracy_with_unknown | 综合准确率（含 unknown 判定） |
| AUROC | known/negative 区分度 |

输出文件：
- `retrieval_metrics.json` — 全部指标
- `per_sample_retrieval.csv` — 每个 query 的检索详情（含 top1/top5 邻居信息）
- `per_class_retrieval_metrics.csv` — 每类指标
- `topk_metrics.csv` — Recall@K / Precision@K
- `nearest_similarity_histogram.png` — 相似度分布
- `retrieval_confusion_matrix.png` — 检索混淆矩阵
- `report.html` — 完整 HTML 报告

### 3. 多阈值 Sweep

```bash
python scripts/evaluate_retrieval.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --gallery outputs/gallery/baseline_train_gallery.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --split test \
  --output-dir outputs/reports/retrieval_eval_baseline_test \
  --thresholds 0.85,0.90,0.91,0.92,0.94
```

输出额外文件：
- `threshold_sweep_retrieval.csv` — 每个 threshold 的指标
- `threshold_sweep_retrieval.html` — 可视化对比报告（高亮推荐/best/balanced）

### 4. 导出部署包

```bash
python scripts/export_inference_bundle.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --class-names configs/class_names.json \
  --output outputs/deploy/pointnet_baseline_bundle
```

输出目录：
- `model.pt` — 模型 checkpoint
- `prototypes.pt` — 类别 prototype
- `threshold.json` — threshold 配置
- `class_names.json` — 类别名称映射
- `config.yaml` — 模型配置
- `version.json` — 版本信息（含 git commit、创建时间）
- `README_INFERENCE.md` — 推理使用说明

### 5. 风险分析

`per_sample_retrieval.csv` 中包含 `risk_level` 字段：

| 风险等级 | 含义 |
|----------|------|
| `safe` | 正确判定，离 threshold 较远 |
| `near_boundary` | 距 threshold 在 0.03 以内 |
| `rejected_known` | known query 被误判为 unknown |
| `misclassified_known` | known query 被接受但 top1 错误 |
| `false_known_negative` | negative query 被误判为 known |

### Threshold 使用建议

基于当前数据（threshold 0.90~0.91）：

- **推荐 threshold=0.90**: known_accept ≈ 97%, negative_reject = 100%, balanced_score 最高
- **保守 threshold=0.85**: known_accept ≈ 99%, 但 negative_reject 下降
- **激进 threshold=0.94**: known_accept ≈ 92%, 但分类准确率更高

## 2.5 轮：统一推理决策逻辑与类别完整性检查

统一最终推理逻辑为 classifier negative → prototype OOD rejection → known class → gallery evidence。增加类别完整性检查，确保部署前所有 known class 都有训练数据。

### 1. 类别完整性检查

```bash
python scripts/check_class_coverage.py \
  --dataset-root dataset \
  --config configs/config.yaml \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --gallery outputs/gallery/baseline_train_gallery.pt
```

检查内容：
- config 中 `num_known_classes` 与 `class_names.json` 长度一致
- `class_names.json` 的 key 覆盖所有 label
- train/val/test 每个 known class 的样本数
- prototypes 中每个 known class 是否存在
- gallery 中每个 known class 是否存在
- **无 train 样本的 known class 标记为 ERROR**（当前 class_014/shenggaozuofalan 无任何样本）

输出：
- `outputs/reports/class_coverage_report.json` — JSON 报告
- `outputs/reports/class_coverage_report.html` — HTML 报告

### 2. 最终推理逻辑（final_infer.py）

```bash
python scripts/final_infer.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --gallery outputs/gallery/baseline_train_gallery.pt \
  --input path/to/sample.pcd
```

决策优先级：

| 优先级 | 步骤 | 条件 | 结果 |
|--------|------|------|------|
| 1 | 分类头 negative | `pred == negative_label` | `"negative"` |
| 2 | Prototype OOD rejection | `nearest_sim < threshold` | `"unknown"` |
| 3 | Known class | 通过 prototype threshold | `"known"` (prototype class) |
| 4 | Gallery evidence | 可选，不覆盖 final_label | top-K 相似样本 |

**关键原则：**
- classifier negative 是第一优先级
- prototype threshold 是 unknown rejection 的主机制
- gallery retrieval 只作为证据展示，**不作为主 OOD 判定**
- 如果 gallery top1 class 与 prototype class 不一致 → `risk_level = "prototype_gallery_conflict"`
- 如果 prototype_similarity - threshold < 0.03 → `risk_level = "near_threshold_boundary"`

输出 JSON 包含三个完整模块：
- `classifier`: 分类头预测、confidence、top-5
- `prototype`: 最近 prototype 类别、similarity、threshold、top-5
- `gallery`: top-1/top-5 相似样本（sample_id、label、similarity、source_path）

### 3. Gallery 的正确用途

- **用途**: 为已知样本提供相似样本检索和证据展示
- **不应用于**: unknown/negative rejection（gallery similarity 对 negative 太宽松）
- **原因**: retrieval-based nearest gallery similarity mean = 0.958，无法有效区分 known 和 negative
- **推荐**: 使用 prototype similarity 做 OOD rejection，gallery 做证据展示

### 4. Threshold 推荐

| Threshold | known_accept | 特点 |
|-----------|-------------|------|
| 0.90 | ~97% | 推荐，balanced_score 最高 |
| 0.91 | ~95% | P05 分位数，保守 |
| 0.85 | ~99% | 宽松，可能放过 negative |
| 0.94 | ~92% | 激进，会误拒部分 known |

### 5. 当前 class_014 问题

`class_014` (shenggaozuofalan) 在所有 split 中均无样本：
- train: 0, val: 0, test: 0
- prototypes 中不存在该 class 的 prototype
- gallery 中不存在该 class 的 gallery entry
- **这是一个数据缺失问题，不是模型问题**
- **建议**: 补充 class_014 的训练数据后重新构建 prototypes 和 gallery

### 6. 部署前检查清单

1. 运行 `check_class_coverage.py` 确认无 ERROR
2. 确认 threshold 设置正确（推荐 0.90~0.91）
3. 确认 prototypes 包含所有 known class
4. 如使用 gallery，确认 gallery 包含所有 known class
5. 用各 class 的代表性样本测试 final_infer.py
6. 确认 class_014 数据缺失是否可接受

### 7. 更新的部署包导出

```bash
python scripts/export_inference_bundle.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --class-names configs/class_names.json \
  --gallery outputs/gallery/baseline_train_gallery.pt \
  --coverage-report outputs/reports/class_coverage_report.json \
  --output outputs/deploy/pointnet_baseline_bundle
```

新增 bundle 文件：
- `gallery.pt` — gallery embeddings（可选）
- `class_coverage_report.json` — 类别完整性报告（可选）
- `final_infer.py` — 最终推理脚本

## 2.6-lite 轮：受限版本推理、评估与交付报告

在暂时无法补充 class_014 和 negative 样本的情况下，完善受限版本的推理、评估和交付报告。

### 为什么暂时不能称为完整 19 类模型

- class_014 (shenggaozuofalan) 无任何训练数据，无法构建 prototype 或 gallery entry
- negative 样本不足（train 仅 12 个），无法充分验证 OOD rejection
- 当前只支持 **18 个 known class**，不是完整的 19 类

### 配置 unsupported_known_labels

在 `configs/config.yaml` 中新增：

```yaml
supported_classes:
  enabled: true
  supported_known_labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
  unsupported_known_labels: [14]
  unsupported_policy: "return_unknown_with_warning"
```

影响：
- `build_prototypes.py` 只为 supported class 构建 prototype
- `build_gallery.py` 只把 supported class 放入 gallery
- `final_infer.py` 如果 classifier 预测 unsupported class → `unsupported_known_class`

### 1. 运行 pseudo-OOD 评估

用已有 known 类模拟 unknown，评估 prototype OOD rejection 的能力：

```bash
python scripts/evaluate_pseudo_ood.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --output-dir outputs/reports/pseudo_ood_eval
```

方法：Leave-One-Class-Out — 临时移除某个 known class 的 prototype，将该 class 的 test 样本当作 pseudo-unknown。

输出：
- `pseudo_ood_metrics.json` — 总体和每类指标
- `per_class_pseudo_ood.csv` — 每类详细结果
- `pseudo_ood_reject_rate_bar.png` — 拒绝率柱状图
- `pseudo_ood_similarity_heatmap.png` — 相似度热力图
- `report.html` — 完整 HTML 报告

### 2. 生成 limited deployment report

```bash
python scripts/generate_limited_deployment_report.py \
  --class-coverage outputs/reports/class_coverage_report.json \
  --ood outputs/reports/ood_eval_baseline_test_p05/ood_metrics.json \
  --retrieval outputs/reports/retrieval_eval_baseline_test/retrieval_metrics.json \
  --pseudo-ood outputs/reports/pseudo_ood_eval/pseudo_ood_metrics.json \
  --threshold outputs/prototypes/baseline_threshold_p05.json \
  --output-dir outputs/reports/limited_deployment
```

报告包含：
- 当前支持范围（18 supported + 1 unsupported）
- 不支持范围（class_014 无法可靠识别）
- 关键指标（OOD、retrieval、pseudo-OOD）
- 推荐推理逻辑
- 交付结论：**limited_internal_prototype**（不是 production ready）
- 下一步必须补充的数据

### 3. 导出受限部署包

```bash
python scripts/export_inference_bundle.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --prototypes outputs/prototypes/baseline_prototypes.pt \
  --threshold-json outputs/prototypes/baseline_threshold_p05.json \
  --class-names configs/class_names.json \
  --gallery outputs/gallery/baseline_train_gallery.pt \
  --coverage-report outputs/reports/class_coverage_report.json \
  --deployment-report outputs/reports/limited_deployment/limited_deployment_report.html \
  --output outputs/deploy/pointnet_baseline_bundle
```

### 4. 当前 supported / unsupported 类

| 类别 | 标签 | 状态 |
|------|------|------|
| 18 个 known class | 0-13, 15-18 | **supported** |
| shenggaozuofalan | 14 | **unsupported**（无训练数据） |
| qita (negative) | 19 | limited evaluation |

### 5. 后续补数据后如何升级为完整版本

1. 补充 class_014 训练数据到 `dataset/train/class_014/`
2. 运行 `rebuild_split.py` 重新划分
3. 重新训练模型（或使用当前模型 fine-tune）
4. 重建 prototypes：`python scripts/build_prototypes.py`
5. 重建 gallery：`python scripts/build_gallery.py`
6. 重新搜索 threshold：`python scripts/search_threshold.py`
7. 运行完整评估链
8. 从 `config.yaml` 移除 `unsupported_known_labels: [14]`
9. 导出完整部署包

## 训练（baseline）

```bash
python scripts/train.py --config configs/config.yaml
```

训练日志输出到 `outputs/logs/train.log`，checkpoint 保存到 `outputs/checkpoints/`。

输出文件：
- `outputs/reports/training_curves.png` — 训练曲线
- `outputs/reports/training_history.json` — 每 epoch 指标
- `outputs/reports/best_val_per_class_metrics.json` — best checkpoint 的每类指标

## 评价

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

结果保存到 `outputs/reports/evaluation.json`、`outputs/reports/confusion_matrix.png`、`outputs/reports/confusion_matrix_normalized.png` 和 `outputs/reports/per_class_metrics.csv`。

## 推理

```bash
# 支持 .npy 和 .pcd 格式
python scripts/infer.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt --input /path/to/sample.npy
python scripts/infer.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt --input /path/to/sample.pcd
```

输出 JSON 格式的预测结果。

## Embedding 评估指标

`scripts/evaluate_embeddings.py` 用于评估 metric learning 的 embedding 质量，输入是已提取的向量 embedding 和对应标签，不需要模型代码。

### 输入格式

支持三种格式：

**格式 1：.npz 文件**

```python
np.savez("embeddings.npz",
    embeddings=embeddings,   # [N, D]
    labels=labels,           # [N]
    sample_ids=ids,          # [N], 可选
    class_names=names,       # list, 可选
)
```

**格式 2：.npy dict 文件**

```python
np.save("embeddings.npy", {
    "embeddings": embeddings,  # [N, D]
    "labels": labels,          # [N]
    "sample_ids": [...],       # 可选
    "class_names": [...],      # 可选
})
```

**格式 3：.csv 文件**

CSV 至少包含 `label` 列和 `feat_0` 到 `feat_D` 列，可选 `sample_id` 列。

### 从模型导出 embedding

在 evaluate.py 或 infer.py 中，模型 forward 返回 `embedding` 字段，可直接收集：

```python
# 伪代码：遍历 test dataloader 收集 embedding
all_embeddings, all_labels = [], []
for batch in dataloader:
    out = model(batch["points"])
    all_embeddings.append(out["embedding"].cpu().numpy())
    all_labels.extend(batch["label"])
np.savez("test_embeddings.npz",
    embeddings=np.concatenate(all_embeddings),
    labels=np.array(all_labels),
)
```

### 运行

```bash
python scripts/evaluate_embeddings.py \
    --input outputs/embeddings/test_embeddings.npz \
    --output-dir outputs/reports/embedding_eval \
    --normalize true \
    --exclude-negative true \
    --negative-label -1 \
    --topk 1 3 5 10 \
    --class-names configs/class_names.json \
    --config configs/config.yaml
```

参数说明：
- `--normalize true`：对 embedding 做 L2 归一化后再计算 cosine similarity
- `--exclude-negative true`：排除 negative 类样本（不参与指标计算）
- `--negative-label`：negative 类的 label id
- `--topk`：Recall@K / Precision@K 的 K 值列表
- `--class-names`：类别名称映射文件路径
- `--config`：config.yaml 路径（用于读取 focus_class_pairs 配置）

### 评价指标说明

| 指标 | 含义 | 理想值 |
|------|------|--------|
| Intra-class similarity | 同类样本两两 cosine similarity 的平均值 | 越高越好 |
| Inter-class similarity | 不同类样本之间的 cosine similarity | 越低越好 |
| Similarity gap | intra - inter，衡量 embedding 区分度 | 越大越好 |
| Recall@K | 以每个样本为 query，top-K 检索结果中至少有一个同类样本的比例 | 越高越好 |
| Precision@K | top-K 检索结果中同类样本的占比 | 越高越好 |
| 1-NN accuracy | 用最近邻样本的 label 作为预测的准确率 | 越高越好 |

### Negative 类处理

如果数据集中包含 negative 类（如"其他"类别），建议设置 `--exclude-negative true`。原因：negative 类内部样本差异大，不适合计算类内聚合度。排除后仅影响 metric learning 指标计算，不影响其他类的评估。

### 输出文件

| 文件 | 内容 |
|------|------|
| `metrics.json` | 所有数值指标汇总 |
| `metrics_summary.json` | 扁平化指标摘要（用于对比脚本） |
| `per_class_metrics.csv` | 每个类别的详细指标 |
| `class_similarity_matrix.csv` | 类别间 cosine similarity 矩阵 |
| `class_similarity_matrix.png` | 类别间相似度热力图 |
| `confusion_matrix.png` | 1-NN 分类混淆矩阵 |
| `top_confusing_pairs.csv` | 最容易混淆的类别对 |
| `embedding_tsne.png` | t-SNE 可视化 |
| `focus_pair_analysis.json` | 关注类别对的详细分析 |
| `focus_pair_*_top_similar.csv` | 关注类别对最相似的样本对 |

### 如何解读结果

- **Similarity gap > 0.3**：embedding 区分度好，同类紧密、异类分离
- **Recall@1 > 0.9**：最近邻检索效果优秀
- **Top confusing pairs**：帮助发现哪些类别容易混淆，指导数据增强或类别合并

## Phase 3.0：新增数据后的完整回归验证

在新增 5208 个 PCD（含 class_014 44 个、qita/negative 256 个）后，用现有 PointNet baseline 跑完整回归。

### 关键变更

- **class_014 (shenggaozuofalan) 从 unsupported 变为 supported**：所有 19 个 known class 均有训练数据
- **negative (qita) 样本从 ~16 增至 256**：OOD 评估更可信
- **config.yaml**：`unsupported_known_labels: []`

### 1. 一键回归

```bash
python scripts/run_full_regression.py \
    --config configs/config.yaml \
    --raw-root label \
    --run-name newdata_pointnet_baseline_v1
```

执行 15 步管道：
1. `prepare_pcd_dataset.py` — PCD → NPY
2. `rebuild_split.py` — 分层划分 train/val/test
3. `validate_dataset.py` — 数据验证
4. `check_class_coverage.py` — 类别完整性检查（训练前）
5. `train.py` — 训练
6. `evaluate.py` — 分类评估
7. `extract_embeddings.py` — 提取 test embedding
8. `evaluate_embeddings.py` — embedding 质量评估
9. `build_prototypes.py` — 构建 prototype
10. `search_threshold.py` — threshold 搜索
11. `evaluate_ood.py` — OOD 评估（含 threshold sweep）
12. `build_gallery.py` — 构建 gallery
13. `evaluate_retrieval.py` — retrieval 评估（含 threshold sweep）
14. `check_class_coverage.py` — 类别完整性检查（训练后，含 prototypes/gallery）
15. `generate_final_report.py` — 最终报告 + 对比

所有输出在 `outputs/runs/{run_name}/`。

可选参数：
- `--skip-train`：跳过训练，复用已有 checkpoint
- `--skip-data-prep`：跳过数据准备（prepare + rebuild）
- `--train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`：自定义划分比例

### 2. 最终报告

```bash
python scripts/generate_final_report.py \
    --config configs/config.yaml \
    --run-name newdata_pointnet_baseline_v1 \
    --run-dir outputs/runs/newdata_pointnet_baseline_v1 \
    --checkpoint outputs/checkpoints/best.pt \
    --prototypes outputs/runs/newdata_pointnet_baseline_v1/prototypes/baseline_prototypes.pt \
    --threshold-json outputs/runs/newdata_pointnet_baseline_v1/prototypes/baseline_threshold.json \
    --gallery outputs/runs/newdata_pointnet_baseline_v1/gallery/baseline_train_gallery.pt
```

报告内容：
1. 数据分布（每类 train/val/test 数量）
2. 类别完整性（class_014 是否有 train/prototype/gallery）
3. 分类指标（overall acc、macro F1、per-class P/R/F1）
4. Embedding 指标（NN acc、similarity gap、top confusing pairs）
5. OOD 指标（AUROC、known_accept、negative_reject）
6. Threshold sweep
7. Retrieval 指标
8. class_014 单独指标
9. Negative 单独指标
10. 是否可替代旧版本

输出：
- `final_report.html` / `final_report.json`
- `comparison_to_previous.html` — 与旧版本对比

### 3. class_014 验证确认

回归完成后确认：
- `check_class_coverage.py` 输出 class_014 status = OK（有 train 样本）
- `prototypes.pt` 中 class_014 的 class_support > 0
- `gallery.pt` 中包含 class_014 的样本
- `final_report.html` 显示 class_014 为 SUPPORTED

### 4. 进入 PointNet++ / PointNeXt 对比阶段

回归完成后：
1. 确认 19 类全部 supported，coverage status = PASS 或 WARNING
2. 确认 macro F1、AUROC 等指标合理
3. 用此轮结果作为 PointNet baseline 基准
4. 引入 PointNet++ / PointNeXt，跑相同管道，对比指标

## Phase 3.1：PointNet 下的 CE-only vs CE+SupCon 受控对比

Phase 3.0 的 baseline 实际使用了 CE + SupCon（metric_weight=0.05, warmup=10），
因此它不是纯 CE baseline。为了科学评估 SupCon 的实际贡献，
本轮在 **完全相同的数据划分** 上补跑一个纯 CE 实验。

### 1. 为什么要补跑 CE-only

- Phase 3.0 的 "baseline" 包含了 SupCon 损失，无法判断 SupCon 是否真的带来了提升
- 受控对比需要：相同数据、相同模型结构（PointNet）、仅损失函数不同
- 结论将指导 PointNet++ / PointNeXt 阶段是否继续使用 SupCon

### 2. 实验配置

两个配置文件位于 `configs/experiments/`：

| 配置 | metric_learning.enabled | metric_weight | warmup |
|------|------------------------|---------------|--------|
| `pointnet_ce_only_newdata.yaml` | false | 0.0 | 0 |
| `pointnet_ce_supcon_newdata.yaml` | true | 0.05 | 10 |

其他所有参数完全相同（seed=42, epochs=100, class_weight_max=10.0, etc.）。

### 3. 如何运行

**重要**：不要重建 split。直接复用 `dataset/` 目录（Phase 3.0 已建好的 train/val/test 划分）。

#### 3a. 跑 CE-only

```bash
python scripts/run_full_regression.py \
    --config configs/experiments/pointnet_ce_only_newdata.yaml \
    --raw-root label \
    --run-name newdata_pointnet_ce_only_v1 \
    --skip-data-prep
```

#### 3b. 跑 CE+SupCon

```bash
python scripts/run_full_regression.py \
    --config configs/experiments/pointnet_ce_supcon_newdata.yaml \
    --raw-root label \
    --run-name newdata_pointnet_ce_supcon_v1 \
    --skip-data-prep
```

> **注意**：两个实验使用 `--skip-data-prep` 以复用相同的 `dataset/` 划分。
> 输出分别保存在 `outputs/runs/newdata_pointnet_ce_only_v1/` 和
> `outputs/runs/newdata_pointnet_ce_supcon_v1/`。

### 4. 如何生成对比报告

```bash
python scripts/compare_runs.py \
    --runs outputs/runs/newdata_pointnet_ce_only_v1 \
           outputs/runs/newdata_pointnet_ce_supcon_v1 \
    --labels "CE-only" "CE+SupCon" \
    --output-dir outputs/reports/newdata_pointnet_controlled_comparison
```

输出文件：
| 文件 | 内容 |
|------|------|
| `comparison.csv` | 所有指标对比表格（含 delta） |
| `comparison.json` | 完整对比数据 + 自动结论 |
| `comparison_report.html` | 可视化对比报告 |

### 5. 对比指标

| 维度 | 指标 |
|------|------|
| 分类 | Overall Acc, Macro F1, Known Acc |
| class_014 | Precision / Recall / F1 |
| Negative | Precision / Recall / F1 |
| 最差类别 | F1 最低的 5 个类 |
| Embedding | Intra/Inter Sim, Gap, NN Acc, Recall@1/5 |
| OOD | Threshold, Known Accept, Negative Reject, AUROC |
| Retrieval | Top1 Acc, AUROC, Recall@1/5 |
| Confusing Pairs | Top 5 最容易混淆的类别对 |

### 6. 自动结论

`compare_runs.py` 会自动判断：

1. **CE+SupCon 是否优于 CE-only**：比较 Macro F1, Similarity Gap, OOD AUROC
2. **是否存在 embedding 退化**：SupCon 的 similarity gap 是否下降
3. **是否存在 negative recall 下降**：SupCon 是否导致 negative 检出率降低
4. **是否推荐在 PointNet++ 中继续使用 SupCon**

### 7. 什么时候进入 PointNet++ / PointNeXt

1. 完成本轮对比，确定 SupCon 是否有效
2. 如果 CE+SupCon 优于 CE-only → 在 PointNet++ 中继续使用 SupCon
3. 如果 CE-only 优于 CE+SupCon → 尝试调参（metric_weight, temperature）或放弃 SupCon
4. 用当前最佳 PointNet 结果作为基准，引入新 backbone 进行对比
5. **不要在 PointNet++ 测试之前引入 OpenPoints**
