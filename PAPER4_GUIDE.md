# Paper 4 诊断测地模态图融合：证据总账

> **工作结论**：下游任务对某张模态图敏感，并不证明该图恢复了真实模态关系。Paper 4 必须分别诊断表征几何（geometry）、是否通信（communication）和边权如何分配（edge allocation）。

本文档是 Paper 4 的唯一主证据笔记。旧的运行细节分别保留在 [PAPER4_MANIFOLD_GUIDE.md](PAPER4_MANIFOLD_GUIDE.md) 和 [PAPER4_GRAPH_EVIDENCE.md](PAPER4_GRAPH_EVIDENCE.md)。ICLR 论文结构见 [PAPER4_ICLR_NARRATIVE.md](PAPER4_ICLR_NARRATIVE.md)。

## 1. 当前判定

### 可以写进正文的观察

1. 向量测地模型学到了更低能量、非线性的路径，但没有改善检索性能。
2. 协方差 SPD 表征相对 matched Euclidean 表征带来稳定提升。
3. legacy hierarchical SPD 的局部图几乎完全退化为单位阵，因此该轮结果不能证明局部模态通信有效。
4. 固定通信预算后，均匀分配优于学习边权；当前学习图没有显示出 edge-allocation advantage。
5. 测试时把学习图替换为均匀图会掉点，但从头训练均匀图又优于学习图。这是 co-adaptation，而不是 topology recovery 的证据。
6. SPD 融合头比 latent concatenation 更小，当前实现约有 `34,582` 对 `298,500` 个融合参数。

### 暂时不能写成结论的内容

- “模型发现了真实 MRI 模态关系”。没有真值图，且局部边在不同数据划分下不稳定。
- “学习图普遍无用”。目前只有 UTSW，公开数据和已知真值的合成数据尚未完成。
- “均匀局部图贡献了性能”。在当前线性传播与中心性/均值池化下，它与局部单位图严格等价。
- “锚点图全面有效”。锚点对 molecular 子组有益，但 pathology 子组存在相反趋势。
- 第三轮是最终多种子结果。该目录缺少三个 published baselines 及确认阶段种子，只能标记为 `prototype_incomplete`。

## 2. 三轮实验审计

### 2.1 Vector geodesic：路径更像测地线，但任务没有受益

来源：`output/sever_paper4_geodesic_full_20260721_093159/aggregate/aggregate_geodesic.json`

| Variant | mAP（mean ± sd） | R@1 | MRR | 关键诊断 |
|---|---:|---:|---:|---|
| `euclidean_graph` | **0.627638 ± 0.009129** | 0.739247 | 0.843836 | 当前轮最佳 mAP |
| `full_geodesic_graph` | 0.623814 ± 0.009213 | 0.739247 | 0.843836 | energy ratio 0.874411；path deviation 0.079689 |
| `latent_concat` | 0.619044 ± 0.014709 | 0.741935 | 0.849171 | 强但更大的融合头 |
| `case_only_metric` | 0.617455 ± 0.018380 | — | — | energy ratio 0.968570 |
| `geodesic_no_graph` | 0.615970 ± 0.017477 | — | — | energy ratio 0.986527 |

**解释边界**：`full_geodesic_graph` 的路径确实偏离直线并降低能量，但 mAP 比 Euclidean 低 `0.003824`。这支持 **geodesicity is not utility**，不支持“测地学习失败”这一更强说法，因为路径目标可能与下游检索目标错配。

### 2.2 Legacy hierarchical SPD：几何提升成立，图贡献未成立

来源：`output/sever_paper4_hierarchical_spd_v1_20260724_084121/aggregate/aggregate_manifold.json`

| Variant | mAP（mean ± sd） | 对结果的含义 |
|---|---:|---|
| `hierarchical_spd_graph` | **0.697527 ± 0.009694** | SPD 主路径 |
| `spd_local_only` | 0.697255 ± 0.011066 | 去掉上层图几乎不变 |
| `spd_no_anchor_family` | 0.696298 ± 0.011143 | 去掉 anchor family 几乎不变 |
| `euclidean_hierarchical_graph` | 0.679162 ± 0.001587 | matched geometry control |
| `latent_concat` | 0.616950 ± 0.009486 | 参数更多但明显更低 |

拓扑审计显示：

- local adjacency 的 diagonal mean 为 `1.0`；off-diagonal mass 约 `4.5e-15`，即数值上的单位阵。
- upper adjacency 的 diagonal mean 为 `0.968207`；off-diagonal mass 只有 `0.031793`。
- SPD 相对 Euclidean 提升约 `0.0184`，但 local-only 和 no-anchor 消融几乎没有变化。

因此该轮只能支持 **SPD representation gain**。它不能支持 hierarchical graph discovery。

### 2.3 Fixed-budget graph evidence：通信有效，学习分配没有优势

来源：`output/sever_paper4_graph_evidence_v220260803`

当前只完成七个 variants × seeds 42/43/44。缺少 `hemis`、`gmu`、`mbt_style`，也未完成 seeds 45/46。以下数字是原型观察，不进入最终主表。

| Variant | mAP（mean ± sd） | 相对 learned SPD |
|---|---:|---:|
| `spd_uniform_graph` | **0.729236 ± 0.013267** | +0.025588 |
| `spd_cross_graph` | 0.703647 ± 0.012958 | reference |
| `spd_identity_graph` | 0.700153 ± 0.019501 | -0.003494 |
| `spd_local_only` | 0.692458 ± 0.015858 | -0.011189 |
| `spd_no_anchor_family` | 0.687632 ± 0.017070 | -0.016016 |
| `euclidean_cross_graph` | 0.654002 ± 0.005134 | -0.049645 |
| `latent_concat` | 0.638257 ± 0.012452 | -0.065390 |

三个 split seeds 上，`uniform - learned` 的 mAP 差分别为 `0.025407 / 0.025412 / 0.025946`，方向完全一致。结构诊断同时确认固定预算确实进入计算：local off-diagonal mass 为 `0.35`，upper off-diagonal mass 为 `0.40`，region-family mass 为 `0.25`。

#### Co-adaptation reversal

在 `spd_cross_graph` 的 checkpoint 上执行测试时干预：

| Intervention | 平均 ΔmAP |
|---|---:|
| uniform | -0.018008 |
| no upper | -0.009821 |
| no region-family | -0.006946 |
| identity | -0.006117 |
| shuffle | -0.005220 |
| no local | +0.002349 |

令 `S(A,B)` 表示在拓扑 A 下训练、在拓扑 B 下测试的得分：

- intervention cost：`S(learned, learned) - S(learned, uniform) = 0.018008`。
- retrained advantage：`S(uniform, uniform) - S(learned, learned) = 0.025588`。
- co-adaptation gap：`S(uniform, uniform) - S(learned, uniform) ≈ 0.043596`。

测试时替换图导致下降，只说明参数与训练图共同适应；它不说明训练图正确或最优。

#### 边稳定性与子组差异

- 不同 split seeds 的 local off-diagonal Pearson correlations 为 `-0.3573 / 0.4706 / -0.1400`，局部边没有稳定解释。
- upper graph correlations 为 `0.4553 / 0.7672 / 0.8720`，较稳定但仍混合了 split change 与 initialization change。
- molecular mAP：learned cross `0.7422`，no-anchor `0.6886`，说明 semantic support 对分子目标可能有效。
- pathology mAP：learned cross `0.7407`，no-anchor `0.7484`，说明锚点作用不是跨目标一致的。
- uniform 在 grade 4、edema、enhancing、core、pathology、molecular 等多数子组上均为当前最佳，但该观察仍需完整协议确认。

## 3. 一个必须写清的代数事实

对 M 个可用模态，固定 cross mass `α` 的均匀图为

```text
Aα = (1 - α)I + α/(M - 1) (11ᵀ - I).
```

`Aα` 对称且双随机，因此对任意模态表示 `H`：

```text
mean(Aα H) = (1/M) 1ᵀ Aα H = (1/M) 1ᵀ H = mean(H).
```

当前 local layer 是线性消息传播，随后按列中心性池化。对称双随机矩阵的列和恒为 1，所以 local uniform 与 local identity 的 pooled region representation 相同。代码中已有数值单元测试。

**直接后果**：第三轮 `spd_uniform_graph` 相对 `spd_identity_graph` 的收益来自 upper uniform communication，而不是 local uniform mixing。简化模型 SPD-UB 因而采用 local identity pooling，省去无效的局部图传播。

## 4. 主张—证据—限定矩阵

| 主张 | 当前强度 | 支持证据 | 必须保留的限定 |
|---|---|---|---|
| 测地性不保证任务效用 | 强 | 更低 energy ratio，但 mAP 低于 Euclidean | 只针对当前路径目标与任务 |
| SPD moment geometry 有效 | 强 | 两轮 matched Euclidean 对照均明显落后 | 尚需公开数据复现 |
| 强制通信预算可防止 self-collapse | 强 | off-diagonal mass 精确达到预算 | 结构参与不等于性能提升 |
| learned edge allocation 优于固定分配 | 不支持 | uniform 比 learned 高 0.0256 | 第三轮尚未完成完整协议 |
| 测试干预可证明图正确 | 被反例否定 | intervention cost 与 retrained advantage 同时为正 | 需要 topology supervision 或额外可识别性假设 |
| semantic support 对所有目标有益 | 不支持 | molecular 与 pathology 趋势不同 | 需报告 target-family subgroup |
| SPD-UB 是更简单的候选方法 | 中等 | 当前性能、代数等价性和参数效率 | 需要 AV-MNIST、MOSEI 和完整 UTSW 确认 |

## 5. 协议风险与修复

### 已发现风险

1. 旧 `--seed` 同时控制数据划分和模型初始化，无法区分 split variance 与 optimization variance。
2. seeds 42/43/44 的 test 集重叠较低，Jaccard 约 `0.107–0.138`；跨 seed 边相关性主要反映换数据，不是固定数据上的图稳定性。
3. 第三轮少了三项 published baselines 与确认种子。
4. 当前 uniform intervention 同时替换 local 和 upper graph，旧结果不能直接定位是哪一级产生变化。
5. UTSW 是单中心医学数据，不足以支撑普遍多模态融合结论。

### 新协议

- `--split_seed` 只控制样本发现与划分；`--model_seed` 只控制初始化和训练随机性；`--seed` 仅作兼容别名。
- `--spd_local_topology` 与 `--spd_upper_topology` 分别取 `learned|identity|uniform`。
- `--paper4_graph_intervention` 只在评估态生效。
- UTSW screening：split seeds `42/43/44` × model seed `101`。
- UTSW confirmation：固定三个 splits，再使用 model seeds `101/102/103`。
- 合成实验：5 data seeds × 5 model seeds；公开数据使用官方 split × 5 model seeds。
- 主指标使用 paired contrast 与 hierarchical bootstrap 95% CI，所有方法只按 validation 选择。

## 6. 新实验入口

### UTSW factorial protocol

```bash
DRY_RUN=1 \
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
STAGE=screen \
bash run_server_paper4_iclr_evidence.sh
```

screening 后按 mean validation mAP 确定最佳 published baseline，再执行：

```bash
BEST_PUBLISHED_BASELINE=hemis \
STAGE=confirm \
DATA_ROOT=/root/autodl-tmp/dataset/UTSW-Glioma \
METADATA_TSV=/root/autodl-tmp/dataset/UTSW_Glioma_Metadata-2-1.tsv \
bash run_server_paper4_iclr_evidence.sh
```

### Synthetic / AV-MNIST / CMU-MOSEI

```bash
DATASET=synthetic \
SPLIT_SEEDS="42 43 44 45 46" \
MODEL_SEEDS="101 102 103 104 105" \
GRAPH_TYPES="chain star two_community" \
REGIMES="geometry_only exchangeable topology_relevant" \
MISSING_RATES="0.0 0.25 0.50" \
bash run_server_paper4_public_benchmarks.sh
```

AV-MNIST 和 MOSEI 使用同一 NPZ token 协议：每个 split 必须包含 `<split>_tokens` 与 `<split>_labels`，可选 `<split>_modality_mask` 和 `<split>_token_mask`。`tokens` 形状为 `[N,G,M,T,C]`；当前公开基准设 `G=1`。

## 7. 投稿前证据门槛

- 若 learned graph 在 topology-relevant synthetic 上不能恢复真值边，只能声称当前参数化无法识别拓扑。
- 若公开数据不复现 `uniform > learned`，中心结论改为“干预敏感不足以证明拓扑恢复”，不写“学习图普遍无用”。
- 只有固定 split、多初始化的 edge stability 足够高，才讨论边的可解释性。
- 第三轮旧目录永远保留 `prototype_incomplete` 标签；最终表只使用新 factorial protocol。
- 医学数据放在通用实验之后，作为真实、少样本、带 semantic supports 的 stress test。

最后更新：2026-08-05。
