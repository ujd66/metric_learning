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

每个 PCD 样本目录可包含伴随文件（`xxx_heightmap.png`, `xxx_info.txt`, `xxx_transform.json`），转换时会自动记录路径。

运行转换：

```bash
python scripts/prepare_pcd_dataset.py \
  --input_dir raw_pcd_dataset \
  --output_dir dataset \
  --splits train val test

# 如需包含 intensity 通道 (xyz + intensity -> 4 channels)：
python scripts/prepare_pcd_dataset.py \
  --input_dir raw_pcd_dataset \
  --output_dir dataset \
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

## 配置

编辑 `configs/config.yaml` 修改超参数。编辑 `configs/class_names.json` 修改类别名称。

## 训练

```bash
python scripts/train.py --config configs/config.yaml
```

训练日志输出到 `outputs/logs/train.log`，checkpoint 保存到 `outputs/checkpoints/`。

## 评价

```bash
python scripts/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

结果保存到 `outputs/reports/evaluation.json` 和 `outputs/reports/confusion_matrix.png`。

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
