# Paper 4 诊断测地模态图融合

Paper 4 使用独立的 `GliomaGeodesicFusionNet`。每个区域内的 T1、T1ce、T2、FLAIR 分别编码，四个模态作为节点，六条模态对路径作为图边。当前实现是判别式 metric-geodesic approximation，不是生成式 Metric Flow Matching。

方法依据：[Metric Flow Matching 论文](https://arxiv.org/abs/2405.14780)与[官方 MIT 代码](https://github.com/kkapusniak/metric-flow-matching)。本仓库使用纯 PyTorch 重新实现任务所需公式，不复制其 Lightning 或 `torchcfm` 训练框架。

## 单次运行

```bash
python -m glioma.cli.train_semantic_alignment \
  --data_root /path/to/UTSW-Glioma \
  --metadata_tsv /path/to/metadata.tsv \
  --paper_config paper4 \
  --fusion_mode geodesic \
  --epochs 50 \
  --out_dir output/paper4_single
```

主要参数：

- `--fusion_mode concat|euclidean|geodesic`
- `--geo_metric_support case_and_anchors|case_only`
- `--disable_fusion_graph`
- `--geo_path_steps 5`
- `--geo_gamma 0.5`
- `--geo_rho 0.001`
- `--lambda_geo_energy 0.1`
- `--lambda_path_semantic 0.1`

## AutoDL 全量协议

先检查五组配置、三个 seeds 和完整测试参数：

```bash
cd /root/autodl-tmp/glioma

DRY_RUN=1 \
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
bash run_server_paper4_geodesic_full.sh
```

后台正式运行：

```bash
mkdir -p logs

nohup env \
  DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
  METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
  GROUP_NAME=paper4_geodesic_full \
  bash run_server_paper4_geodesic_full.sh \
  > logs/paper4_geodesic_full_launcher.log 2>&1 &
```

查看总日志：

```bash
tail -f logs/paper4_geodesic_full_launcher.log
```

launcher 顺序运行：

1. `full_geodesic_graph`
2. `euclidean_graph`
3. `case_only_metric`
4. `geodesic_no_graph`
5. `latent_concat`

默认每组使用 seeds `42 43 44`、50 epoch、完整 test split。15 个训练全部成功后自动执行 `glioma.cli.aggregate_geodesic_runs`。

## 输出目录

```text
output/server_runs/<GROUP_NAME>/<variant>/paper4/seed_*/
logs/server_runs/<GROUP_NAME>/<variant>/
output/server_runs/<GROUP_NAME>/aggregate/
```

单 seed 重点产物：

```text
best_paper4_alignment.pt
checkpoint_manifest.json
history.json
test_metrics.json
geodesic_diagnostics.json
fusion_graph.json
fusion_figure_manifest.json
geodesic_path_projection.png
modality_geodesic_graph.png
geodesic_energy_comparison.png
```

聚合目录包含 `aggregate_geodesic.json`、`paper4_ablation_summary.png`、`paper4_edge_stability.png` 和聚合 manifest。
