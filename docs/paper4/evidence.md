# Idea 4：诊断测地模态图融合——研究叙事与证据总账

> **核心结论**：下游任务对某张模态图敏感，并不证明该图恢复了真实模态关系。表征几何（geometry）、通信支持与预算（communication）和边权分配（edge allocation）必须被分别诊断。

本文档是 Paper 4 的中文主笔记，涵盖研究命题、证据边界、实验决策和写作口径。当前 Python 入口见第 10 节；英文论文结构见[写作蓝图](manuscript.md)。

## 2026-09-08 更新：UTSW confirmation 已完成

本节是最新证据入口；下文旧三轮数值保留为历史发现过程。完整结果、新增种子对照、配对区间、子任务和审计记录见 本轮结果报告（本地产物 `reports/paper4_confirmation_20260908/RESULTS.md`）。

- 实际来源：`output/sever_paper2+paper4_20260824alloutput/server_runs/paper4_iclr_factorial_v1`。
- 完成 **81/81**：原 screening 33 次保留不变，新增 confirmation 48 次；8 个确认模型各有 split `42/43/44` × model `101/102/103`，另外 HeMIS/GMU/MBT-style 各 3 次为筛选阶段设计。
- baseline 继续使用先前按 screening validation 选定的 `latent_concat`，不按 test 重新选择。独立复算全部 81 次患者级 mAP 与存档一致；同 split 的患者划分、anchor vocabulary、50-epoch 训练预算已核对。

| 确认模型（各 n=9） | Test mAP，均值 ± 样本 SD |
|---|---|
| SPD-UB | **0.726674 ± 0.009419** |
| SPD uniform-both | 0.725569 ± 0.010437 |
| SPD upper-only | 0.712739 ± 0.010528 |
| SPD learned-both | 0.705391 ± 0.014021 |
| SPD identity-both | 0.700747 ± 0.009741 |
| SPD no-anchor-family | 0.686718 ± 0.013127 |
| Euclidean learned-both | 0.640545 ± 0.020208 |
| latent_concat | 0.629807 ± 0.010780 |

本轮 geometry gain **+6.485 mAP 点**、communication gain **+2.482 点**、allocation gain（learned−uniform）**−2.018 点**，方向均为 9/9 一致；这些是不同条件对比，不能相加为独立因果贡献。SPD-UB 较 learned 高 **2.128 点**、较 concat 高 **9.687 点**，但较 uniform 仅 **0.111 点**，描述性配对 95% CI **[−0.076,+0.289] 点**，因此只能称性能接近，不能称显著更优或已证明等价。uniform 的 validation 均值略高于 SPD-UB，SPD-UB 的定位来自预定简化方案，不能用 test 排名替代 validation 选型。

干预方面，S(A,A)=0.705391、S(A,B)=0.670648、S(B,B)=0.725569，A=learned、B=uniform。换图平均成本 **+3.474 点**，但只有 **8/9** 组掉点；重新训练 uniform **9/9** 更好，co-adaptation gap **+5.492 点**。这与协同适应解释相容，不是唯一机制的证明。

Anchor-family 对 molecular **+5.372 点**（9/9），pathology **−0.114 点**且区间跨零。固定 split 的局部边 Spearman 分别 **0.505/0.333/−0.048**，上层为 **0.513/0.783/0.448**；上层更稳定，仍不能等同图真值恢复。

统计边界：3 个 split 来自同一队列，测试患者有重叠；bootstrap 区间仅描述 split/model 运行变异。原始逐患者干预分数未导出，本轮效率记录为空。**UTSW 阶段完成，整篇论文证据尚未完成**；synthetic、AV-MNIST、MOSEI 与效率仍待补证据。旧第三轮继续保留 `prototype_incomplete`，本轮新目录为 `factorial_complete`。

## 1. Idea 4 研究卡片

| 项目 | 当前定义 |
|---|---|
| 工作标题 | **Topology Is Not the Message: Disentangling Geometry, Communication, and Co-Adaptation in Multimodal Fusion** |
| 一句话问题 | learned modality graph 带来的下游收益，是否足以证明它恢复了可解释的模态拓扑？ |
| 一句话回答 | 不足以；几何、通信和边权分配都能独立改变性能，测试时换图掉点还可能只来自参数—图的协同适应。 |
| 论文类型 | 诊断研究 + 由诊断导出的简化方法，而不是医学应用论文或单纯的新图模块论文。 |
| 方法产物 | **SPD-UB**：SPD moment representation + local identity pooling + uniform-budget upper communication。 |
| 医学数据角色 | UTSW 是少样本、高风险、带 semantic supports 的真实压力测试，不能承担论文的一般性来源。 |
| 最稳健主张 | **Graph sensitivity is not topology identification.** |
| 需要新实验才能升级的主张 | learned allocation 是否普遍不如 uniform；何种生成机制下能够恢复 planted topology。 |

这篇论文不再问“怎样学习一张更复杂的医学模态图”，而是问：**当图只是端到端网络中的潜变量时，我们究竟凭什么把它解释成结构？** 这是 ICLR 所需的一般 representation learning / latent structure 问题。

## 2. 从旧故事到新故事

### 2.1 旧故事为什么不能继续

旧叙事大致是“测地路径比欧氏路径合理 → SPD 比向量表示合理 → 层次图学到医学模态关系 → 在胶质瘤检索上有效”。三轮实验已经破坏了这条链：

1. 更低的路径能量没有带来更高的 mAP，因此几何上的“更像测地线”不能自动转化为任务效用。
2. SPD 的提升与局部图退化同时出现，说明此前归给层次图的收益至少有相当部分其实来自表征几何。
3. uniform 重新训练后优于 learned，但 learned checkpoint 在测试时换成 uniform 又掉点，说明干预敏感性会被 co-adaptation 混淆。
4. MRI 没有真实模态图标签，不能用下游性能或一张漂亮的邻接矩阵声称“发现了真实关系”。

因此，旧故事中最危险的等号是：

```text
downstream gain = useful communication = correct edge allocation = recovered topology
```

新论文的任务就是拆掉这些等号，并给出分别检验它们的协议。

### 2.2 ICLR 主叙事链

正文应按以下因果顺序推进：

1. learned modality graphs 常以任务分数和测试时换图掉点作为结构证据，但这两种证据都混合了多个机制。
2. 把融合分解为 **Geometry / Communication / Edge Allocation**，并要求 matched retraining，而不是只做 checkpoint intervention。
3. 给出两个明确限制：局部 uniform 传播在当前池化下可以严格无效；测试时敏感可以与替代图重新训练后更优同时成立。
4. 在 planted synthetic 上先判断 topology 何时可识别，再到 AV-MNIST、CMU-MOSEI 检验任务广度和缺失模态鲁棒性。
5. 最后用 UTSW 展示真实场景中的 geodesic utility gap、local graph collapse 和 co-adaptation reversal。
6. 诊断自然导出更简单的 **SPD-UB**；它是分析结果，不是先提出模块再寻找理由。

论文开篇、Figure 1 和 Table 1 都不能以胶质瘤为中心。推荐结果顺序固定为：

```text
formal diagnosis → synthetic identifiability → public multimodal benchmarks
→ co-adaptation/efficiency → UTSW stress test
```

## 3. 诊断框架：究竟在估计什么

令 `S(g,c,a)` 表示使用 geometry `g`、communication policy `c` 和 allocation `a` 从头训练后的验证/测试得分：

- **Geometry gain**：`S(SPD,c,a) - S(Euclidean,c,a)`。
- **Communication gain**：`S(g,communicate,fixed) - S(g,identity,none)`。
- **Allocation gain**：`S(g,communicate,learned) - S(g,communicate,uniform)`。

对训练拓扑 A 与替换拓扑 B，再区分：

- **Intervention cost**：`S(A,A) - S(A,B)`。
- **Retrained advantage**：`S(B,B) - S(A,A)`。
- **Co-adaptation gap**：`S(B,B) - S(A,B)`。

其中 `S(A,B)` 只回答“按 A 训练的参数是否依赖 A”，不能回答“B 是否更差”或“A 是否正确”。后两个问题必须至少加入 `S(B,B)`，真实结构恢复还必须在有 planted graph 的数据上直接测 edge AUROC/AUPRC、排序相关性和稳定性。

### 四个研究问题

1. **RQ1 — Source of gain：** 下游增益分别来自 geometry、communication 还是 allocation？
2. **RQ2 — Evidence validity：** test-time intervention sensitivity 是否能推出 learned topology 正确或可识别？
3. **RQ3 — Identifiability conditions：** 在哪些数据生成机制、缺失率和图结构下，下游训练能够恢复 planted modality graph？
4. **RQ4 — Simplification：** 诊断能否导出一个更小、更稳健且不牺牲性能的融合规则？

## 4. 两条理论抓手与方法结果

### 命题 1：Uniform-budget pooling invariance

当前局部层使用线性消息传播后做均值/中心性池化。对称双随机 uniform-budget 矩阵保持节点均值，因此 local uniform 与 local identity 的 pooled representation 严格相同。这个结论只适用于当前算子；非线性、边特异变换、多层传播或非双随机 mask 都可能打破等价性。

**叙事作用**：第三轮 uniform 的收益不能归给局部 uniform mixing，只能进一步定位到 upper communication；同时给出了删除冗余局部传播的直接理由。

### 命题 2：Intervention sensitivity does not imply identifiability

存在两模态构造，使 `S(A,A) > S(A,B)` 且 `S(B,B) > S(A,A)`。第一项表现为测试时换图掉点，第二项却说明 B 从头训练后更优。因此 sensitivity 是依赖性的证据，不是 topology correctness 的充分证据。

**叙事作用**：UTSW 已经出现这一经验反转；正文仍需一个显式最小反例和 synthetic 数值复现。措辞应为“common evidence is insufficient”，不能夸大为一般不可能性定理。

### 由诊断导出的 SPD-UB

SPD-UB 的最小定义为：

1. 从有效 token 估计 trace-normalized covariance，并映射到 Log-Euclidean tangent space。
2. 局部模态采用 identity/mean pooling，删除代数上冗余的 local uniform propagation。
3. 存在 group 或 semantic support nodes 时，在 upper level 按允许的边类型分配固定、均匀的跨节点通信预算。
4. 使用 symmetric vectorization 和小型 readout 输出融合表示。

`G=1` 的公开数据不能强行制造 semantic anchors，此时方法诚实退化为 SPD moment pooling。现有参数审计为 SPD 融合头 `34,582`，latent concat `298,500`，约小 `8.6×`；正式稿还要补 FLOPs、训练时间和显存。

## 5. 当前判定

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

## 6. 三轮实验审计

### 6.1 Vector geodesic：路径更像测地线，但任务没有受益

来源：`output/sever_paper4_geodesic_full_20260721_093159/aggregate/aggregate_geodesic.json`

| Variant | mAP（mean ± sd） | R@1 | MRR | 关键诊断 |
|---|---:|---:|---:|---|
| `euclidean_graph` | **0.627638 ± 0.009129** | 0.739247 | 0.843836 | 当前轮最佳 mAP |
| `full_geodesic_graph` | 0.623814 ± 0.009213 | 0.739247 | 0.843836 | energy ratio 0.874411；path deviation 0.079689 |
| `latent_concat` | 0.619044 ± 0.014709 | 0.741935 | 0.849171 | 强但更大的融合头 |
| `case_only_metric` | 0.617455 ± 0.018380 | — | — | energy ratio 0.968570 |
| `geodesic_no_graph` | 0.615970 ± 0.017477 | — | — | energy ratio 0.986527 |

**解释边界**：`full_geodesic_graph` 的路径确实偏离直线并降低能量，但 mAP 比 Euclidean 低 `0.003824`。这支持 **geodesicity is not utility**，不支持“测地学习失败”这一更强说法，因为路径目标可能与下游检索目标错配。

### 6.2 Legacy hierarchical SPD：几何提升成立，图贡献未成立

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

### 6.3 Fixed-budget graph evidence：通信有效，学习分配没有优势

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

## 7. 命题 1 的代数展开

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

## 8. 主张—证据—限定矩阵

| 主张 | 当前强度 | 支持证据 | 必须保留的限定 |
|---|---|---|---|
| 测地性不保证任务效用 | 强 | 更低 energy ratio，但 mAP 低于 Euclidean | 只针对当前路径目标与任务 |
| SPD moment geometry 有效 | 强 | 两轮 matched Euclidean 对照均明显落后 | 尚需公开数据复现 |
| 强制通信预算可防止 self-collapse | 强 | off-diagonal mass 精确达到预算 | 结构参与不等于性能提升 |
| learned edge allocation 优于固定分配 | 不支持 | 本轮 confirmation uniform 比 learned 高 0.020177，9/9 | 仅适用于当前 UTSW 协议，非普遍结论 |
| 测试干预可证明图正确 | 被反例否定 | intervention cost 与 retrained advantage 同时为正 | 需要 topology supervision 或额外可识别性假设 |
| semantic support 对所有目标有益 | 不支持 | molecular 与 pathology 趋势不同 | 需报告 target-family subgroup |
| SPD-UB 是更简单的候选方法 | UTSW 性能已确认 | 0.726674 ± 0.009419，与 uniform 接近 | 公开数据与本轮效率未确认；不宣称显著优于 uniform |

## 9. 协议风险与修复

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

## 10. Python 实验入口

从包含 glioma/ 的父目录运行。以下是 UTSW 阶乘协议中一个 SPD-UB 单元，完整矩阵和 seed 口径见本节表格。

```bash
python -m glioma.cli.train_semantic_alignment \
  --paper_config paper4 --data_root /path/to/UTSW-Glioma \
  --metadata_tsv /path/to/metadata.tsv \
  --paper4_fusion_backend spd_hierarchical --spd_graph_policy cross_budget \
  --spd_local_topology identity --spd_upper_topology uniform \
  --split_seed 42 --model_seed 101 --epochs 50 --num_workers 4 --augment \
  --skip_interventions --out_dir /path/to/run_root/spd_ub/paper4/split_42/model_101
```

| 变体 | 几何 | 局部 / 上层拓扑 | 额外参数 |
|---|---|---|---|
| spd_cross_graph | spd | learned / learned | 执行图干预 |
| spd_identity_graph | spd | identity / identity | |
| spd_uniform_graph | spd | uniform / uniform | |
| spd_upper_only | spd | identity / learned | |
| spd_ub | spd | identity / uniform | |
| spd_no_anchor_family | spd | learned / learned | --disable_spd_anchor_families |
| euclidean_cross_graph | euclidean | learned / learned | --spd_geometry euclidean |
| latent_concat / hemis / gmu / mbt_style | published | 由基线定义 | --paper4_fusion_backend published_baseline --paper4_baseline 对应名称 |

历史 screening 使用 split 42/43/44、model 101，运行全部 11 组。按验证 mAP 选择 published baseline 后，confirmation 为 7 个核心组加选定基线，补 model 102/103。组内保留相同 split、训练预算和输出布局。

```bash
python -m glioma.cli.aggregate_paper4_diagnostics \
  --run_root /path/to/run_root --metric map \
  --expected_split_seeds 42 43 44 --expected_model_seeds 101 102 103 \
  --expected_variants spd_cross_graph spd_identity_graph spd_uniform_graph \
    spd_upper_only spd_ub spd_no_anchor_family euclidean_cross_graph latent_concat
```

### Synthetic / AV-MNIST / CMU-MOSEI

```bash
python -m glioma.cli.train_paper4_benchmark \
  --dataset synthetic --planted_graph chain --synthetic_regime topology_relevant \
  --split_seed 42 --model_seed 101 --out_dir /path/to/new_synthetic_run
```

AV-MNIST 和 MOSEI 使用 --dataset avmnist 或 --dataset mosei，并提供 --data_path。每个 NPZ split 包含 <split>_tokens 与 <split>_labels，可选 <split>_modality_mask 和 <split>_token_mask。token 形状为 [N,G,M,T,C]，公开基准设 G=1。

## 11. 投稿前证据门槛

- 若 learned graph 在 topology-relevant synthetic 上不能恢复真值边，只能声称当前参数化无法识别拓扑。
- 若公开数据不复现 `uniform > learned`，中心结论改为“干预敏感不足以证明拓扑恢复”，不写“学习图普遍无用”。
- 只有固定 split、多初始化的 edge stability 足够高，才讨论边的可解释性。
- 第三轮旧目录永远保留 `prototype_incomplete` 标签；最终表只使用新 factorial protocol。
- 医学数据放在通用实验之后，作为真实、少样本、带 semantic supports 的 stress test。

## 12. 结果解释决策树

正式实验完成后，按“结构恢复”和“任务效用”两个轴冻结结论：

| Planted edge recovery | Downstream result | 允许的解释 |
|---|---|---|
| 高且稳定 | learned > uniform | 在明确生成条件和当前参数化下，学习分配既可识别又有任务价值；不能外推到真实 MRI 图。 |
| 高且稳定 | learned ≈/＜ uniform | 拓扑可以恢复，但结构正确不等于对当前任务更有用。 |
| 低或不稳定 | learned > uniform | 学习图可作为预测组件，但边不能作结构解释；重点报告 predictive utility–identifiability gap。 |
| 低或不稳定 | learned ≈/＜ uniform | 当前 learned allocation 既未表现出结构恢复，也未表现出任务优势；优先采用 SPD-UB。 |

跨真实数据集再按以下规则收缩或升级主张：

1. **若 AV-MNIST、MOSEI、UTSW 均出现 uniform ≥ learned：** 可以把“learned allocation often adds no value under matched budgets”作为经验主发现，但仍不写“图学习普遍无用”。
2. **若公开数据 learned 与 uniform 各有胜负：** 主发现固定为“收益来源依任务而异，必须分解诊断”；SPD-UB 改为强基线而非普遍替代方案。
3. **若公开数据普遍 learned > uniform：** 保留 Proposition 2 和 matched-retraining protocol，把 UTSW 写成 co-adaptation 个案，不把 uniform 胜出放进标题或摘要首要结果。
4. **若 SPD 在公开数据无稳定优势：** 将 SPD-UB 降级为 UTSW-specific consequence，论文的一般贡献只保留诊断框架、理论限制与 benchmark evidence。

## 13. 写作冻结规则

### 正文可以使用

- “Downstream performance conflates representation geometry, communication, and edge allocation.”
- “Test-time graph sensitivity establishes parameter dependence, not topology identification.”
- “In the current UTSW prototype, uniform-budget retraining outperforms learned allocation while a test-time replacement hurts performance.”
- “The diagnosis motivates SPD-UB by removing a locally invariant propagation step.”

### 完成确认实验前禁止使用

- “We recover the true modality graph.”
- “Learned graphs are universally unnecessary.”
- “Uniform communication is always optimal.”
- “SPD-UB is state of the art.”
- “The learned MRI edges are biologically meaningful.”

### 摘要与引言的证据顺序

1. 一般问题：latent graph 被当作结构解释时，现有证据缺什么。
2. 诊断贡献：三因素分解、matched retraining、stability 与 planted recovery。
3. 理论贡献：pooling invariance 与 co-adaptation counterexample。
4. 通用实验：synthetic、AV-MNIST、MOSEI。
5. 医学压力测试与效率：UTSW、SPD-UB、参数/FLOPs。

摘要中的 `[PUBLIC-DATA RESULT]` 继续保留；UTSW `[CONFIRMED VALUE]` 现可使用本轮 uniform−learned **2.02 mAP points**，不能用第三轮原型数值替代。最终写作统一使用 `topology recovery`、`communication budget`、`edge allocation`、`co-adaptation` 四组术语，不用“真实图”“发现关系”等更强同义替换。

证据更新：2026-09-08。
