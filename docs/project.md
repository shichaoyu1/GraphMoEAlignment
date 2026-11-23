# Glioma 项目介绍与使用说明

## 项目定位

`glioma` 是一个面向脑胶质瘤多模态 MRI 的语义单元对齐项目。项目以病人级 ROI 为输入，将 MRI 病灶区域、病理信息和分子标志物映射到统一的 shared semantic space，用于验证不同医学语义单元之间是否能够形成稳定、可解释的跨模态对齐关系。

本说明只基于当前 `glioma` 项目本身编写。运行、配置、评估和可视化都以 `glioma/` 包内模块为边界，项目可作为独立 Python 包使用。

项目按研究路线提供独立配置。Paper 2 的修订入口、Paper 4 的几何与 authority 实验，以及 Idea4.5 的协议见[文档索引](README.md)。

## 目录结构

当前项目主要模块如下：

```text
glioma/
  anchors/                 # 语义锚点导出入口
  cli/                     # 命令行入口
  config/                  # 实验配置和论文组合预设
  data/                    # 病例发现、标签解析、数据加载
  eval/                    # 语义对齐评估
  io/                      # 实验产物保存
  models/                  # 主模型、编码器、图、扩散、原型库等模型模块
  modules/                 # graph / diffusion / MoE 核心模块
  objectives/              # 对比学习、MedCLIP-style、DCCA、prototype bank
  semantic/                # 语义锚点、目标策略、指标、词表
  training/                # 训练循环和工具函数
  validation/              # 已保存的验证实验产物
  visualization/           # 语义空间和语义图可视化
```

## 核心功能

### 语义节点

项目将 MRI 输入拆成更贴近医学语义的节点：

| 来源 | 节点 |
|---|---|
| MRI 区域 | `Necrotic/Core`、`Edema`、`Enhancing` |
| 病理锚点 | `Tumor Grade`、`Tumor Type` |
| 分子锚点 | `IDH`、`MGMT`、`1p19Q CODEL` |
| 临床锚点 | 默认关闭，可通过 `--include_clinical_anchors` 加入 |

### 模型思想

每个病人样本会被编码为 shared representation 和 private representation：

- shared representation：承载跨区域、跨病人、跨锚点的公共语义。
- private representation：保留区域或个体私有信息。
- graph 模块：在 shared space 中构建语义节点关系，并用 Laplacian consistency 约束结构一致性。
- diffusion 模块：在 private latent 上做互补语义恢复。
- prototype bank：维护病理、分子、临床锚点原型，用于多正样本语义对齐。
- MoE 模块：可选的 semantic/graph/diffusion 专家路由组件，用于模块组合实验。

### 对齐目标

项目支持三类对齐目标：

| 目标 | 说明 |
|---|---|
| `clip` | 多正样本 contrastive alignment |
| `medclip` | MedCLIP-style 同字段负样本屏蔽策略 |
| `dcca` | DCCA 风格相关性对齐，样本不足时回退到 contrastive 辅助项 |

## 环境准备

按[项目首页](../README.md#setup)安装统一依赖。从包含 `glioma/` 的父目录执行以下 Python 模块命令。

## 数据准备

推荐使用 UTSW-Glioma 风格的数据组织。数据根目录下应包含多个病人文件夹，每个病人文件夹中包含四个 MRI 模态和一个分割 mask。

示例：

```text
UTSW-Glioma/
  Patient_001/
    brain_t1.nii.gz
    brain_t1ce.nii.gz
    brain_t2.nii.gz
    brain_flair.nii.gz
    tumorseg_manual_correction.nii.gz
  Patient_002/
    ...
```

代码兼容的常见文件名包括：

| 类型 | 示例 |
|---|---|
| T1 | `brain_t1.nii.gz`、`brain_t1_ants.nii.gz`、`*_t1.nii.gz` |
| T1ce | `brain_t1ce.nii.gz`、`brain_t1ce_ants.nii.gz`、`*_t1ce.nii.gz`、`*_t1gd.nii.gz` |
| T2 | `brain_t2.nii.gz`、`brain_t2_ants.nii.gz`、`*_t2.nii.gz` |
| FLAIR | `brain_flair.nii.gz`、`brain_fl_ants.nii.gz`、`*_flair.nii.gz` |
| Segmentation | `rtumorseg_manual_correction.nii.gz`、`tumorseg_manual_correction.nii.gz`、`tumorseg_FeTS.nii.gz`、`*_seg.nii.gz` |

metadata 建议使用 TSV 文件，并通过 `--metadata_tsv` 显式传入。常用字段：

```text
Subject ID
Tumor Grade
Tumor Type
IDH
MGMT
1p19Q CODEL
Age at Histological Diagnosis
Gender
```

`Subject ID` 需要和病人文件夹名对应，方便样本与病理/分子信息匹配。

## 快速开始

建议从项目父目录运行，使 Python 能识别 `glioma` 包：

```powershell
cd /path/to/workspace
```

先用实际数据路径执行 smoke test：

```powershell
python -m glioma.cli.train_semantic_alignment `
  --data_root "/path/to/UTSW-Glioma" `
  --metadata_tsv "/path/to/metadata.tsv" `
  --variant full `
  --max_cases 8 `
  --epochs 1 `
  --batch_size 2 `
  --roi_size 32 `
  --z_slices 3 `
  --cpu
```

smoke test 通过后，可运行推荐配置：

```powershell
python -m glioma.cli.train_semantic_alignment `
  --data_root "/path/to/UTSW-Glioma" `
  --metadata_tsv "/path/to/metadata.tsv" `
  --variant full `
  --graph_type learnable `
  --epochs 30 `
  --batch_size 4 `
  --augment
```

如果数据中存在配准后的 `_ants` 文件，可添加：

```powershell
--prefer_registered
```

## 输出结果

默认输出目录：

```text
output/semantic_alignment_experiment
```

Paper 1/3 保留原语义对齐与 graph/diffusion 产物。Paper 2 v2 使用独立的 TopoMoE 产物协议：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次实验的完整参数 |
| `anchor_vocab.json` | 语义锚点词表 |
| `splits.json` | train/val/test 病人划分 |
| `history.json` | 训练和验证历史 |
| `best_direct_alignment.pt` | direct validation mAP 最佳权重（Paper 2） |
| `best_routed_topomoe.pt` | routed validation mAP 最佳权重（Paper 2 主 checkpoint） |
| `checkpoint_manifest.json` | 双 checkpoint 的 epoch、指标与 fallback 状态 |
| `test_metrics.json` | routed-best 上的 direct/routed 测试指标 |
| `test_metrics_direct_checkpoint.json` | direct-best 对照指标 |
| `patient_level_records.json` | 病人级 query、target、prototype 记录 |
| `routing_records.json` / `routing_spectrum.json` | 病例级路由和分层路由统计 |
| `topomoe_topology.json` / `topomoe_diagnostics.json` | prior、initial、effective topology 与学习诊断 |
| `intervention_metrics.json` | remove expert、拓扑替换、wrong routing 和 node mask 结果 |
| `topomoe_*.png` | TopoMoE 机制、路由、拓扑、checkpoint 和干预图 |

Paper 2 不再生成 `semantic_unit_alignment_space.png`、`semantic_unit_graph_50patients.png`、`semantic_unit_adjacency.png` 或 `semantic_unit_laplacian.png`。这些旧图只属于 Paper 1/3 的 legacy visualization path。

`glioma/validation/` 中已经保存了 `paper1`、`paper2`、`paper3` 三组验证产物，可作为输出格式参考。里面的 `config.json` 是历史运行快照，若包含旧机器上的绝对路径，以新实验实际传入的 `--out_dir` 和 `--validation_output_root` 为准。

## 评估指标

常用指标包括：

| 指标 | 含义 |
|---|---|
| `recall@1` / `recall@5` / `recall@10` | query 能否在 top-k 中检索到正确锚点 |
| `mrr` | mean reciprocal rank |
| `map` | mean average precision |
| `pair_auc` | 正负 query-anchor pair 的区分能力 |
| `positive_negative_distance_gap` | 正样本与负样本距离差 |
| `anchor_consistency` | query 到目标锚点的平均一致性 |
| `edge_precision@10/25/50` | 语义图 top edge 的目标匹配精度 |
| `pathology_unavailable` | 移除病理锚点后的检索结果 |
| `molecular_unavailable` | 移除分子锚点后的检索结果 |

## 消融实验

`--variant` 控制主要消融路线：

| 变体 | 作用 |
|---|---|
| `full` | 完整模型：语义 graph + semantic anchors + shared/private + diffusion |
| `clip` | CLIP-style 对比学习基线 |
| `medclip_style` | MedCLIP-style 多正样本对齐 |
| `dcca` | DCCA 风格对齐目标 |
| `graph_shared_only` | 只保留 shared graph，不使用 private/diffusion |
| `hgt` | 当前等价映射到 `graph_shared_only` |
| `no_anchor` | 去除病理锚点 |
| `graph_only` | 保留 graph，关闭 diffusion |
| `modality_vector` | 使用模态级节点而不是区域级节点 |
| `no_private` | 去除 private branch 和 diffusion |
| `no_graph` | 去除图结构约束 |

示例：

```powershell
python -m glioma.cli.train_semantic_alignment --data_root "/path/to/UTSW-Glioma" --metadata_tsv "/path/to/metadata.tsv" --variant clip
python -m glioma.cli.train_semantic_alignment --data_root "/path/to/UTSW-Glioma" --metadata_tsv "/path/to/metadata.tsv" --variant graph_only
python -m glioma.cli.train_semantic_alignment --data_root "/path/to/UTSW-Glioma" --metadata_tsv "/path/to/metadata.tsv" --variant no_graph
```

`--graph_type` 控制图构建方式：

| 参数 | 说明 |
|---|---|
| `no_graph` | 不使用图结构 |
| `fixed` | 固定连接图 |
| `similarity` | 基于 shared representation 相似度建图 |
| `learnable` | 使用可学习 edge MLP 建图 |
| `random` | 随机图对照 |

## 论文组合预设

`glioma/config/paper_profiles.py` 提供以下论文组合预设：

| 参数 | 模块组合 |
|---|---|
| `--paper_config paper1` | graph + diffusion |
| `--paper_config paper2` | TopoMoE v2：shared units + disease-anchor topology；默认无 MRI graph/private/diffusion |
| `--paper_config paper3` | diffusion + MoE |
| `--paper_config paper4` | 测地或 SPD 融合，见[证据与参数](paper4/evidence.md) |

示例：

```powershell
python -m glioma.cli.train_semantic_alignment `
  --data_root "/path/to/UTSW-Glioma" `
  --metadata_tsv "/path/to/metadata.tsv" `
  --variant full `
  --paper_config paper1 `
  --epochs 30 `
  --batch_size 4
```

当 `--paper_config` 不为 `none` 且没有手动指定 `--out_dir` 时，结果会写入对应的 validation 输出目录。

## 服务器后台运行

从包含 glioma/ 的父目录执行。下面启动一次 Paper 2 TopoMoE 训练，输出目录和启动日志共用本次运行名。

```bash
run_dir="glioma/output/server_runs/paper2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup python -u -m glioma.cli.train_semantic_alignment \
  --paper_config paper2 --data_root /path/to/UTSW-Glioma \
  --metadata_tsv /path/to/metadata.tsv --seed 42 --epochs 50 \
  --out_dir "$run_dir/seed_42" > "$run_dir/launcher.log" 2>&1 < /dev/null &
echo "PID=$!  LOG=$run_dir/launcher.log"
```

多折修订实验使用[Paper 2 修订入口](paper2/experiments.md)。

## 常用参数

| 参数 | 说明 |
|---|---|
| `--data_root` | 数据根目录 |
| `--metadata_tsv` | metadata TSV 路径 |
| `--out_dir` | 输出目录 |
| `--validation_output_root` | paper config 的验证输出根目录 |
| `--max_cases` | 限制病例数量，适合 smoke test |
| `--train_ratio` | 训练集比例 |
| `--val_ratio` | 验证集比例 |
| `--roi_size` | ROI 裁剪后的二维尺寸 |
| `--z_slices` | 2.5D 输入的 z 方向切片数 |
| `--node_mode` | `regions` 使用病灶区域节点，`modalities` 使用模态节点 |
| `--graph_type` | 图构建方式 |
| `--moe_module` | `none`、`semantic_moe`、`graph_moe`、`diffusion_moe`、`topo_moe` |
| `--topo_mode` | `prior_only`、`learned_only`、`prior_plus_learned` |
| `--target_policy` | `region_rules` 或 `all_patient_anchors` |
| `--include_clinical_anchors` | 加入临床锚点 |
| `--exclude_pathology_anchors` | 排除病理锚点 |
| `--exclude_molecular_anchors` | 排除分子锚点 |
| `--augment` | 启用训练增强 |
| `--cache` | 缓存样本读取结果 |
| `--num_workers` | DataLoader worker 数量 |
| `--cpu` | 强制使用 CPU |
| `--seed` | 随机种子 |

## 推荐流程

1. 先用 `--max_cases 8 --epochs 1 --cpu` 做 smoke test，确认数据路径、metadata 字段和 NIfTI 命名能被识别。
2. smoke test 通过后，去掉 `--cpu`，将 `--epochs` 调到 30 或更高。
3. Paper 2 主实验使用 `--paper_config paper2`，其默认只保留 disease-anchor topology；MRI `learnable` graph 消融使用 `--variant graph_shared_only`。
4. Paper 2 消融优先比较 v1/v2、三种 topology mode、prototype refinement、family-balanced route 和 product/log-prob mixture；显式 `--topomoe_version v1` 恢复首轮 `learnable MRI graph + private + no diffusion` 基线组合。
5. 论文组合实验可使用 `--paper_config paper1/paper2/paper3` 统一落盘。
6. Paper 2 结果优先查看 `test_metrics.json`、`checkpoint_manifest.json`、`topomoe_diagnostics.json`、`intervention_metrics.json` 和 `topomoe_*.png`。

## 注意事项

- 本项目的主任务是语义单元对齐与检索评估，不是直接临床诊断。
- 病理/分子锚点用于构建可解释的 shared semantic space，不能作为临床决策依据。
- 病例数过少或训练集锚点不足时，对比学习目标无法稳定构建，脚本会提示需要更多 semantic cases 或 anchors。
- metadata 中未知、空值或 `NA` 字段不会进入 semantic anchor vocabulary。
- 本项目用于科研和教学演示，不能用于真实临床诊断。
