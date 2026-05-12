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

### 5. 训练

```bash
python scripts/train.py --config configs/config.yaml
```

### 两种数据划分方式

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| **自动划分**（推荐） | `prepare_pcd_dataset.py --flat` + `rebuild_split.py` | 从原始数据开始，需要确保每类都有 val/test |
| **手动 mapping** | 按 `sample_mapping.json` 复制到 `raw_pcd_dataset/` | 已有固定划分方案 |

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

## 训练

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
    --class-names configs/class_names.json
```

参数说明：
- `--normalize true`：对 embedding 做 L2 归一化后再计算 cosine similarity
- `--exclude-negative true`：排除 negative 类样本（不参与指标计算）
- `--negative-label`：negative 类的 label id
- `--topk`：Recall@K / Precision@K 的 K 值列表
- `--class-names`：类别名称映射文件路径

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
| `per_class_metrics.csv` | 每个类别的详细指标 |
| `class_similarity_matrix.csv` | 类别间 cosine similarity 矩阵 |
| `class_similarity_matrix.png` | 类别间相似度热力图 |
| `confusion_matrix.png` | 1-NN 分类混淆矩阵 |
| `top_confusing_pairs.csv` | 最容易混淆的类别对 |
| `embedding_tsne.png` | t-SNE 可视化 |

### 如何解读结果

- **Similarity gap > 0.3**：embedding 区分度好，同类紧密、异类分离
- **Recall@1 > 0.9**：最近邻检索效果优秀
- **Top confusing pairs**：帮助发现哪些类别容易混淆，指导数据增强或类别合并
